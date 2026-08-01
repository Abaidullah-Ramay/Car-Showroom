import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from showroom.sales_agent import build_agent
from showroom.search_engine import VIBE_MAP, get_embedder, load_data

st.set_page_config(page_title="Abaid Automobile Showroom", page_icon="🚗", layout="wide")

FIELD_LABELS = {
    "fuel": "fuel", "transmission": "transmission",
    "type": "body type", "drive": "drivetrain",
}


@st.cache_data
def get_cars_data():
    cars, embeddings = load_data()
    return cars, embeddings


@st.cache_resource
def get_embedder_client():
    return get_embedder()


try:
    cars, embeddings = get_cars_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

embedder = get_embedder_client()

# `sink` is the single source of truth for the results grid: the search box writes to
# it, and so do the agent's search tools. Held per session and mutated in place —
# never reassigned — because the agent's tools closed over this exact dict.
if "sink" not in st.session_state:
    st.session_state.sink = {}
if "agent" not in st.session_state:
    st.session_state.agent = build_agent(
        cars, embeddings, embedder, st.session_state.sink
    )
if "history" not in st.session_state:
    st.session_state.history = []

sink = st.session_state.sink

st.title("Abaid Automobile Showroom")

# No sidebar filters: the query parser already lifts fuel, transmission, body type,
# drivetrain and budget out of what the customer types, and Sam passes the same
# constraints explicitly through his tools. The dropdowns duplicated both.
RESULTS_PER_SEARCH = 12


@st.cache_data
def inventory_summary():
    """Total stock and the count of every body type, for the header.

    "unknown" is the single largest bucket, so a plain count sort leads the showroom
    with it. Real body types come first; the catch-alls go last.
    """
    counts = cars["type"].value_counts()
    vague = [t for t in ("other", "unknown") if t in counts.index]
    named = counts.drop(index=vague)
    return len(cars), pd.concat([named, counts[vague]])


total_cars, type_counts = inventory_summary()

st.markdown(f"**{total_cars:,} vehicles in stock.**")
with st.container(horizontal=True):
    for body_type, count in type_counts.items():
        st.badge(f"{body_type} {count:,}", color="gray")


def describe(spec):
    return ", ".join(f"**{FIELD_LABELS.get(k, k)} = {v}**" for k, v in spec.items())


def md_safe(text):
    """Escape dollar signs before rendering as markdown.

    Streamlit treats `$...$` as LaTeX math. A reply quoting two prices — which is
    most of what a salesperson says — turns everything between the first and second
    dollar sign into mangled math, so "from $3,000 to $22,300" came out as
    "3,000 ∗∗ to ∗∗ 22,300" with the markdown eaten.
    """
    return str(text).replace("$", r"\$")


def car_title(row):
    """Listings are missing any of year / manufacturer / model, so build the heading
    from whatever is actually present rather than assuming all three."""
    make = row["manufacturer"] if row["manufacturer"] != "unknown" else ""
    parts = [
        str(int(row["year"])) if pd.notna(row["year"]) else "",
        str(make).title(),
        str(row["model"]).title(),
    ]
    return " ".join(p for p in parts if p) or "Listing"


def field(row, col, default="not listed"):
    value = row.get(col)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value).strip()
    if not text or text.lower() == "unknown":
        return default
    return text


@st.dialog("Vehicle details", width="large", on_dismiss="rerun")
def show_car_details(car_id):
    """Full detail modal. Streamlit supplies the close (X) button top-right."""
    match = cars[cars["id"] == car_id]
    if match.empty:
        st.error("That listing is no longer in inventory.")
        return
    row = match.iloc[0]

    st.markdown(f"### {car_title(row)}")
    st.caption(
        f"{field(row, 'region')}, {field(row, 'state').upper()}  ·  "
        f"listing `{row['id']}`"
    )

    # No image here — with the photo gone the four headline figures get the full
    # width instead of being squeezed into a 2x2 beside it.
    price_col, miles_col, year_col, cond_col = st.columns(4)
    price_col.metric("Asking price",
                     f"${int(row['price']):,}" if pd.notna(row["price"]) else "n/a",
                     border=True)
    miles_col.metric("Odometer",
                     f"{row['odometer']:,.0f} mi" if pd.notna(row["odometer"]) else "n/a",
                     border=True)
    year_col.metric("Year",
                    str(int(row["year"])) if pd.notna(row["year"]) else "n/a",
                    border=True)
    condition = field(row, "condition", None)
    cond_col.metric("Condition", condition.title() if condition else "Not listed",
                    border=True)

    st.markdown("#### Specification")
    specs = [
        ("Body type", field(row, "type")),
        ("Fuel", field(row, "fuel")),
        ("Transmission", field(row, "transmission")),
        ("Drivetrain", field(row, "drive")),
        ("Cylinders", field(row, "cylinders")),
        ("Size", field(row, "size")),
        ("Paint", field(row, "paint_color")),
        ("Title status", field(row, "title_status")),
        ("VIN", field(row, "VIN")),
        ("Posted", field(row, "posting_date")[:10]),
    ]
    left, right = st.columns(2)
    for i, (label, value) in enumerate(specs):
        (left if i % 2 == 0 else right).markdown(f"**{label}**  \n{value}")

    st.markdown("#### How this car reads")
    vibes = (
        pd.Series({k: row[v] for k, v in VIBE_MAP.items()})
        .sort_values(ascending=True)
        .rename("score")
        .to_frame()
    )
    st.bar_chart(vibes, horizontal=True, height=210)
    st.caption(
        "Model-inferred impressions from the listing text, not manufacturer specs. "
        "They are competitive shares across the six labels, so read them relative "
        "to each other rather than as absolute confidence."
    )

    st.markdown("#### Seller's description")
    with st.container(border=True, height=200):
        st.write(field(row, "description_clean", "No description provided."))
    st.caption(
        "Written by the seller. Craigslist listings contain mistakes, so trust the "
        "specification above over the prose."
    )


def render_card(row):
    with st.container(border=True):
        st.subheader(car_title(row))

        price = f"${int(row['price']):,}" if pd.notna(row["price"]) else "price n/a"
        miles = f"{row['odometer']:,.0f} mi" if pd.notna(row["odometer"]) else "mileage n/a"
        st.markdown(f"**{price}**  ·  {row['type']}  ·  {miles}")
        st.markdown(
            f"`{row['fuel']}` `{row['transmission']}` `{row['drive']}` `{row['condition']}`"
        )
        desc = row["description_clean"]
        st.caption(desc[:180] + ("..." if len(desc) > 180 else ""))

        # Streamlit has no natively clickable container, so a full-width button is
        # the card's hit area. Keyed by listing id — index would collide as soon as
        # a new search reorders the grid.
        if st.button("View full details", key=f"card_{row['id']}",
                     icon=":material/open_in_full:", width="stretch"):
            st.session_state.selected_car = int(row["id"])


def run_agent(user_text):
    """One agent turn. His search tool mutates `sink`, so the results grid updates
    as a side effect of the conversation. That is the whole integration: there is no
    separate search path to keep in step."""
    st.session_state.history.append(HumanMessage(user_text))
    result = st.session_state.agent.invoke({"messages": st.session_state.history})
    st.session_state.history = result["messages"]


results_col, chat_col = st.columns([3, 2], gap="large")

with results_col:
    meta = sink.get("meta")
    if meta is not None:
        enforced = []
        if meta.constraints:
            enforced.append(describe(meta.constraints))
        if meta.price_from_query:
            low, high = meta.price_range
            if low is not None and high is not None:
                enforced.append(f"**price ${low:,}–${high:,}**")
            elif high is not None:
                enforced.append(f"**price under ${high:,}**")
            elif low is not None:
                enforced.append(f"**price over ${low:,}**")
        if enforced:
            st.success(f"Enforcing {', '.join(enforced)} from your search.")
        if meta.excluded:
            st.info(f"Excluding {describe(meta.excluded)}.")
        if meta.relaxed:
            st.warning(
                f"No cars in inventory match {describe(meta.relaxed)} alongside your "
                f"other requirements, so that was relaxed. Results below match "
                f"everything else you asked for."
            )

    results = sink.get("results")
    if results is None:
        st.info(
            "Tell Sam what you are after and the cars will appear here. Try "
            "\"a family SUV under $15,000, automatic\" or \"what is your cheapest "
            "electric car?\"",
            icon=":material/arrow_forward:",
        )
    elif results.empty:
        st.info("No cars matched. Ask Sam to relax a requirement or widen the search.")
    else:
        st.subheader(f"Showing {len(results)} cars")
        st.caption("Click any card to see the full listing.")
        cols = st.columns(2)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 2]:
                render_card(row)

# Popped rather than read, so dismissing the modal does not immediately reopen it
# on the next rerun. Only one dialog may be called per script run.
if "selected_car" in st.session_state:
    show_car_details(st.session_state.pop("selected_car"))

with chat_col:
    heading, clear = st.columns([3, 1], vertical_alignment="bottom")
    heading.subheader("Sam the salesman")
    # Lived in the sidebar before; it belongs next to the conversation it clears.
    if clear.button("Clear", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.history = []
        st.rerun()

    st.caption(
        "Describe what you want and Sam pulls it up on the left. Requirements you "
        "state outright (fuel, transmission, body type, drivetrain, budget) are "
        "enforced exactly; the rest is matched by meaning. He can also look up any "
        "listing, compare cars and answer questions about what is in stock."
    )

    transcript = st.container(height=560, border=True)
    with transcript:
        if not st.session_state.history:
            st.chat_message("assistant").write(
                f"Welcome to Abaid Automobile Showroom. I'm Sam. We have "
                f"{total_cars:,} vehicles on the lot right now. How can I help you "
                f"today?"
            )
        for msg in st.session_state.history:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(md_safe(msg.content))
            elif isinstance(msg, AIMessage):
                # Surface tool calls so it's clear what Sam actually looked up
                # rather than what he might have made up.
                for call in msg.tool_calls or []:
                    args = {k: v for k, v in call["args"].items() if v not in (None, "")}
                    st.caption(f"🔎 `{call['name']}` {args}")
                if msg.content:
                    st.chat_message("assistant").write(md_safe(msg.content))

    user_text = st.chat_input("Describe the car you want, or ask Sam anything")
    if user_text:
        with st.spinner("Sam is looking…"):
            try:
                run_agent(user_text)
            except Exception as exc:
                st.session_state.history.append(
                    AIMessage(f"Sorry, something went wrong on my end: {exc}")
                )
        st.rerun()
