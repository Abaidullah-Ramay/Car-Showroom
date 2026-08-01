"""A LangGraph ReAct agent that plays the salesperson over the car inventory.

Design notes:

* Tools close over the already-loaded dataframe and embedding matrix, so the agent
  reuses the exact same search path as the UI — one ranking implementation, not two.
* Every search the agent runs is written to a `sink` dict, which the Streamlit layer
  reads to repaint the results grid. That is how "show me something cheaper" updates
  what is on screen instead of only being described in prose.
* The agent may only speak about rows a tool returned. Listing text is seller-written
  and demonstrably wrong in places (there is a "Camry" in here tagged electric), so
  the prompt tells it to trust the structured fields and hedge on the prose.
"""
import json

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from showroom.config import CHAT_MODEL
from showroom.search_engine import CATEGORICAL_COLS, SearchResult, search_cars

load_dotenv()

MAX_ROWS = 12

SALES_PROMPT = """You are Sam, a salesperson at an online used-car dealership. You \
are talking to a customer browsing the lot.

HOW YOU WORK
- Be warm, brief and concrete. Two or three sentences per turn unless the customer \
asks for detail. Sound like a person who knows the lot, not a brochure.
- Write plain prose. No markdown: no **bold**, no bullet points, no headings, no \
tables. You are speaking to someone across a desk, not writing a document. When you \
list two or three cars, do it in a sentence.
- Write prices as plain numbers with a dollar sign and commas, e.g. $14,995. Never \
wrap them in asterisks or backticks.
- Never use em dashes or long hyphens in your replies. Use a comma, a full stop, or \
brackets instead. Ordinary hyphens inside a word such as "four-wheel drive" are fine.
- When the customer describes what they want, call `search_inventory` and tell them \
what actually came back.
- When they ask about a specific car on screen, call `get_car_details` for the full \
listing before answering.
- For questions about the lot as a whole ("how many electric SUVs do you have?"), \
call `inventory_stats`.
- To weigh two cars against each other, call `compare_cars`.

GROUNDING. THIS MATTERS MORE THAN SOUNDING HELPFUL
- Only ever discuss cars a tool returned in this conversation. Never invent a \
listing, a price, a mileage or a spec.
- The structured fields (price, year, mileage, fuel, transmission, drivetrain, body \
type, condition) are the reliable data. Lead with those.
- The description text is written by sellers and contains mistakes and marketing \
noise. Quote it as "the listing says", never as established fact.
- A field reading "unknown" means the seller never stated it. Say it is not listed. \
Never fill it in from the model name or your own assumptions.
- You do not have service history, accident records, inspection results, or true \
range/MPG figures. If asked, say so plainly and point at what you do have.
- If a search returns nothing matching a requirement, say that outright and offer \
the nearest alternative. Never quietly substitute something that does not match \
what they asked for.
- Do not invent financing, warranties, delivery, discounts or trade-in offers. If \
asked, say you would need to hand them to a colleague for that.

Prices are asking prices. Mileage is odometer reading at listing time."""


def _fmt_row(row):
    """One compact line per car. The agent gets facts, not raw ad copy."""
    make = row["manufacturer"] if row["manufacturer"] != "unknown" else ""
    name = " ".join(str(p) for p in [int(row["year"]), make, row["model"]] if p != "")
    return (
        f"[id {row['id']}] {name} | ${int(row['price']):,} | {row['type']} | "
        f"{row['fuel']} | {row['transmission']} | {row['drive']} | "
        f"{row['odometer']:,.0f} mi | condition {row['condition']}"
    )


def _fmt_rows(df):
    if df.empty:
        return "No matching cars in inventory."
    return "\n".join(_fmt_row(r) for _, r in df.iterrows())


def describe_grid(sink):
    """The cars currently on the customer's screen, injected each turn."""
    df = sink.get("results")
    if df is None:
        return "\n\nNothing is on the customer's screen yet."

    if df.empty:
        ctx = (f"\n\nThe last search ({sink.get('label')!r}) returned NOTHING. The "
               "customer's screen is empty.")
    else:
        ctx = ("\n\nCURRENTLY ON THE CUSTOMER'S SCREEN "
               f"(from the search {sink.get('label', 'they ran')!r}):\n"
               f"{_fmt_rows(df)}")

    # What the search engine had to bend to produce these rows. Sam must pass this
    # on — results that silently violate a stated requirement are the exact failure
    # this whole system was built to stop.
    meta = sink.get("meta")
    if meta is not None:
        if getattr(meta, "constraints", None):
            terms = ", ".join(f"{k}={v}" for k, v in meta.constraints.items())
            ctx += f"\nEvery car above is guaranteed to match: {terms}."
        if getattr(meta, "price_from_query", False):
            low, high = meta.price_range
            bounds = []
            if low is not None:
                bounds.append(f"at least ${low:,}")
            if high is not None:
                bounds.append(f"no more than ${high:,}")
            ctx += (f"\nThey stated a budget and it was applied: every car above is "
                    f"{' and '.join(bounds)}.")
        if getattr(meta, "relaxed", None):
            terms = ", ".join(f"{k}={v}" for k, v in meta.relaxed.items())
            fields = ", ".join(meta.relaxed)
            ctx += (f"\nIMPORTANT: the lot has nothing matching {terms} alongside "
                    f"their other requirements, so that requirement was dropped to "
                    f"produce these results. You must tell them this outright. The "
                    f"cars above are NOT guaranteed to match {fields}. Check each "
                    f"car's own values above before describing it, and never "
                    f"summarise the group as though it satisfies {terms}.")

    return ctx


def build_tools(cars, embeddings, embedder, sink):
    """Create the toolset bound to this inventory. `sink` captures search results."""

    # The dataset mixes cases ("SUV" but "sedan", lowercase manufacturers), and the
    # model has no reason to guess right. Map whatever it sends onto the real value.
    canonical = {
        col: {v.lower(): v for v in cars[col].dropna().unique()}
        for col in CATEGORICAL_COLS
    }

    def _canon(col, val):
        if val is None or val == "":
            return None
        return canonical[col].get(str(val).strip().lower(), str(val).strip())

    def _spec(fuel, transmission, body_type, drive, manufacturer):
        return {"fuel": _canon("fuel", fuel),
                "transmission": _canon("transmission", transmission),
                "type": _canon("type", body_type),
                "drive": _canon("drive", drive),
                "manufacturer": _canon("manufacturer", manufacturer)}

    def _filtered(fuel, transmission, body_type, drive, manufacturer,
                  min_price, max_price):
        df = cars
        for col, val in _spec(fuel, transmission, body_type, drive,
                              manufacturer).items():
            if val:
                df = df[df[col] == val]
        if min_price is not None:
            df = df[df["price"] >= min_price]
        if max_price is not None:
            df = df[df["price"] <= max_price]
        return df

    @tool
    def search_inventory(
        query: str = "",
        fuel: str = None,
        transmission: str = None,
        body_type: str = None,
        drive: str = None,
        manufacturer: str = None,
        min_price: int = None,
        max_price: int = None,
        sort_by: str = "relevance",
        limit: int = 8,
    ) -> str:
        """Search the lot and update what the customer sees on screen.

        Use this whenever the customer describes what they are after, or asks to see
        something different. The results replace the grid on their screen.

        Args:
            query: Plain-language description of the wanted car, e.g. "roomy and
                comfortable for long highway drives". Leave empty for a pure
                attribute lookup like "cheapest Tesla".
            fuel: One of gas, diesel, hybrid, electric, other.
            transmission: One of automatic, manual, other.
            body_type: One of sedan, SUV, coupe, hatchback, wagon, truck, pickup,
                mini-van, van, convertible, bus, offroad.
            drive: One of 4wd, fwd, rwd.
            manufacturer: Brand name, e.g. tesla, toyota, ford.
            min_price: Minimum asking price in dollars.
            max_price: Maximum asking price in dollars.
            sort_by: relevance (default), price_asc, price_desc, year_desc,
                mileage_asc.
            limit: How many cars to return, 1-12.
        """
        limit = max(1, min(int(limit or 8), MAX_ROWS))

        if sort_by != "relevance" or not (query or "").strip():
            df = _filtered(fuel, transmission, body_type, drive, manufacturer,
                           min_price, max_price)
            order = {
                "price_asc": ("price", True), "price_desc": ("price", False),
                "year_desc": ("year", False), "mileage_asc": ("odometer", True),
            }.get(sort_by)
            if order:
                df = df.sort_values(order[0], ascending=order[1])
            results, relaxed = df.head(limit), {}
            found = SearchResult(results, price_range=(min_price, max_price))
        else:
            # Reuse the app's ranking so chat and grid never disagree.
            found = search_cars(
                query=query, cars=cars, embeddings=embeddings, embedder=embedder,
                filters={**_spec(fuel, transmission, body_type, drive, manufacturer),
                         "price_range": (min_price or 0, max_price or 10**9)},
                top_k=limit, parse_constraints=True,
            )
            results, relaxed = found.results, found.relaxed

        # The UI reads `meta` to show what was enforced and what had to be relaxed.
        # search_cars treats explicit tool arguments as deliberate choices and keeps
        # them out of `constraints`, so fold them back in or the banner under-reports
        # exactly the requirements the customer stated most plainly.
        explicit = {k: v for k, v in
                    _spec(fuel, transmission, body_type, drive, manufacturer).items()
                    if v}
        found.constraints = {**explicit, **found.constraints}
        if min_price is not None or max_price is not None:
            found.price_from_query = True
            low, high = found.price_range
            found.price_range = (min_price if min_price is not None else low,
                                 max_price if max_price is not None else high)

        sink["results"] = results
        sink["label"] = query or "attribute search"
        sink["meta"] = found

        note = ""
        if relaxed:
            terms = ", ".join(f"{k}={v}" for k, v in relaxed.items())
            note = (f"\n\nNOTE: nothing matched {terms} together with the other "
                    f"requirements, so that was dropped. Tell the customer this "
                    f"directly.")
        elif results.empty:
            # Tell the agent which requirement is actually the blocker, so it can
            # offer a real alternative instead of just repeating "nothing found".
            note = _diagnose_empty(fuel, transmission, body_type, drive,
                                   manufacturer, min_price, max_price)

        return f"{len(results)} result(s), now shown on screen:\n{_fmt_rows(results)}{note}"

    def _diagnose_empty(fuel, transmission, body_type, drive, manufacturer,
                        min_price, max_price):
        requested = {"fuel": fuel, "transmission": transmission,
                     "body_type": body_type, "drive": drive,
                     "manufacturer": manufacturer}
        requested = {k: v for k, v in requested.items() if v}
        if not requested:
            return ""

        # Drop one requirement at a time; whichever unblocks the search is the
        # constraint worth telling the customer about.
        alternatives = []
        for drop in requested:
            kept = {k: v for k, v in requested.items() if k != drop}
            df = _filtered(kept.get("fuel"), kept.get("transmission"),
                           kept.get("body_type"), kept.get("drive"),
                           kept.get("manufacturer"), min_price, max_price)
            if not df.empty:
                alternatives.append(
                    f"dropping {drop}={requested[drop]} would give {len(df):,} car(s)"
                )
        if not alternatives:
            return ("\n\nNOTE: no single requirement is the blocker. This "
                    "combination is far outside what the lot carries. Ask the "
                    "customer which requirement matters most.")
        return ("\n\nNOTE: nothing matches all of that. " + "; ".join(alternatives) +
                ". Tell the customer what is unavailable and offer the closest "
                "alternative. Do not pretend it matches.")

    @tool
    def get_car_details(listing_id: int) -> str:
        """Full detail on one listing, including the seller's own description.

        Args:
            listing_id: The numeric id shown in square brackets, e.g. 7314068693.
        """
        match = cars[cars["id"] == int(listing_id)]
        if match.empty:
            return f"No listing with id {listing_id} exists in inventory."
        row = match.iloc[0]
        fields = {c: row[c] for c in CATEGORICAL_COLS}
        return json.dumps({
            "id": int(row["id"]),
            "year": int(row["year"]),
            "model": row["model"],
            "price_usd": int(row["price"]),
            "odometer_miles": int(row["odometer"]),
            "cylinders": None if pd.isna(row.get("cylinders")) else row["cylinders"],
            "paint_color": None if pd.isna(row.get("paint_color")) else row["paint_color"],
            "title_status": None if pd.isna(row.get("title_status")) else row["title_status"],
            "state": None if pd.isna(row.get("state")) else row["state"],
            **fields,
            "seller_description": row["description_clean"][:900],
        }, default=str)

    @tool
    def compare_cars(listing_ids: list[int]) -> str:
        """Side-by-side facts for 2-4 listings the customer is weighing up.

        Args:
            listing_ids: Numeric listing ids, e.g. [7314068693, 7311964042].
        """
        ids = [int(i) for i in listing_ids][:4]
        match = cars[cars["id"].isin(ids)]
        if match.empty:
            return "None of those listing ids exist in inventory."
        missing = set(ids) - set(match["id"])
        out = _fmt_rows(match)
        if missing:
            out += f"\n\nNot in inventory: {sorted(missing)}"
        return out

    @tool
    def inventory_stats(
        group_by: str = None,
        fuel: str = None,
        transmission: str = None,
        body_type: str = None,
        drive: str = None,
        manufacturer: str = None,
        min_price: int = None,
        max_price: int = None,
    ) -> str:
        """Counts and price ranges across the whole lot. Does not change the screen.

        Use for questions about what the dealership stocks overall, e.g. "how many
        electric SUVs do you have?" or "what brands do you carry?".

        Args:
            group_by: Column to break the count down by. One of fuel, transmission,
                type, drive, manufacturer, condition.
            fuel: Restrict to this fuel before counting.
            transmission: Restrict to this transmission before counting.
            body_type: Restrict to this body type before counting.
            drive: Restrict to this drivetrain before counting.
            manufacturer: Restrict to this brand before counting.
            min_price: Minimum asking price in dollars.
            max_price: Maximum asking price in dollars.
        """
        df = _filtered(fuel, transmission, body_type, drive, manufacturer,
                       min_price, max_price)
        if df.empty:
            return "0 cars match that description."

        lines = [f"{len(df):,} car(s) match.",
                 f"Asking price: ${df['price'].min():,} low / "
                 f"${int(df['price'].median()):,} median / ${df['price'].max():,} high.",
                 f"Model years {int(df['year'].min())}-{int(df['year'].max())}."]

        if group_by:
            col = "type" if group_by == "body_type" else group_by
            if col not in cars.columns:
                lines.append(f"(cannot group by {group_by!r})")
            else:
                counts = df[col].value_counts().head(15)
                breakdown = ", ".join(f"{k}: {v:,}" for k, v in counts.items())
                lines.append(f"By {col}: {breakdown}")
        return "\n".join(lines)

    return [search_inventory, get_car_details, compare_cars, inventory_stats]


def build_agent(cars, embeddings, embedder, sink):
    """Wire the tools and prompt into a ReAct agent."""
    tools = build_tools(cars, embeddings, embedder, sink)

    # A callable prompt so the on-screen cars are refreshed on every turn rather
    # than frozen into a static system message at construction time.
    def prompt(state):
        return [SystemMessage(SALES_PROMPT + describe_grid(sink))] + state["messages"]

    return create_react_agent(ChatOpenAI(model=CHAT_MODEL), tools, prompt=prompt)
