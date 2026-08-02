import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from showroom.config import FREE_TURNS
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
def free_turns_used():
    """Turns spent per visitor, shared across every session in this app process.

    Deliberately not per-session: a session counter resets on reload and in a new
    tab, which makes it no limit at all. This survives both. It does reset when the
    app sleeps or redeploys, and everyone behind one NAT shares an entry, so treat
    it as budget-stretching rather than access control. The real protection is the
    spend cap and rate limits configured on the API project.
    """
    return {}


def visitor_id():
    return getattr(st.context, "ip_address", None) or "local"


try:
    cars, embeddings = get_cars_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# `sink` is the single source of truth for the results grid: the agent's search
# tools write to it. Held per session and mutated in place, never reassigned,
# because the tools closed over this exact dict.
if "sink" not in st.session_state:
    st.session_state.sink = {}
if "history" not in st.session_state:
    st.session_state.history = []
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""

sink = st.session_state.sink

with st.sidebar:
    st.subheader("Your own API key")
    st.caption(
        "The demo runs on a small shared allowance. Add your own OpenAI key to keep "
        "talking to Sam once it runs out."
    )
    st.session_state.user_api_key = st.text_input(
        "OpenAI API key", type="password", placeholder="sk-...",
        value=st.session_state.user_api_key,
        help="Used only for this browser session. Never stored or logged.",
    ).strip()
    if st.session_state.user_api_key:
        st.success("Using your key. The shared allowance is untouched.")

user_key = st.session_state.user_api_key
own_key = bool(user_key)

# Rebuild the clients whenever the key changes, so a visitor's key takes effect
# immediately and clearing it falls back to the shared one.
if st.session_state.get("clients_for_key") != user_key:
    embedder = get_embedder(user_key or None)
    st.session_state.embedder = embedder
    st.session_state.agent = build_agent(
        cars, embeddings, embedder, sink, api_key=user_key or None
    )
    st.session_state.clients_for_key = user_key
embedder = st.session_state.embedder

turns_left = max(0, FREE_TURNS - free_turns_used().get(visitor_id(), 0))
can_talk = own_key or turns_left > 0

st.title("Abaid Automobile Showroom")

# No sidebar filters: the query parser already lifts fuel, transmission, body type,
# drivetrain and budget out of what the customer types, and Sam passes the same
# constraints explicitly through his tools. The dropdowns duplicated both.
RESULTS_PER_SEARCH = 12

# Prints each tool call above Sam's reply, e.g. `search_inventory {'fuel': 'electric'}`.
# Useful when checking whether he actually looked something up or improvised it, but
# it is developer detail a customer should not see. Flip to True to debug.
SHOW_TOOL_CALLS = False


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


OUT_OF_CREDIT = (
    "insufficient_quota", "exceeded your current quota", "billing_hard_limit",
    "rate limit", "rate_limit", "429",
)


def looks_like_quota_error(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in OUT_OF_CREDIT)


def run_agent(user_text):
    """One agent turn. His search tool mutates `sink`, so the results grid updates
    as a side effect of the conversation. That is the whole integration: there is no
    separate search path to keep in step."""
    st.session_state.history.append(HumanMessage(user_text))
    result = st.session_state.agent.invoke({"messages": st.session_state.history})
    st.session_state.history = result["messages"]
    # Only a turn paid for out of the shared allowance counts against it.
    if not own_key:
        used = free_turns_used()
        used[visitor_id()] = used.get(visitor_id(), 0) + 1


ALLOWANCE_SPENT = (
    "That is the shared demo allowance used up. Add your own OpenAI API key in the "
    "sidebar to keep talking to Sam. Everything else on this page still works: you "
    "can open any car for full details."
)


# Sam runs the full page width across the top, with the results underneath.
heading, clear = st.columns([6, 1], vertical_alignment="bottom")
heading.subheader("Sam the salesman")
# Lived in the sidebar before; it belongs next to the conversation it clears.
if clear.button("Clear", icon=":material/delete_sweep:", width="stretch"):
    st.session_state.history = []
    st.rerun()

transcript = st.container(height=360, border=True)
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
            if SHOW_TOOL_CALLS:
                for call in msg.tool_calls or []:
                    args = {k: v for k, v in call["args"].items() if v not in (None, "")}
                    st.caption(f"🔎 `{call['name']}` {args}")
            if msg.content:
                st.chat_message("assistant").write(md_safe(msg.content))

if can_talk:
    user_text = st.chat_input("Describe the car you want, or ask Sam anything")
    if not own_key:
        st.caption(f"{turns_left} of {FREE_TURNS} free messages left on the demo key.")
else:
    st.chat_input("Add your API key in the sidebar to continue", disabled=True)
    user_text = None
    st.warning(ALLOWANCE_SPENT, icon=":material/key:")

# Handled here rather than after the grid so the spinner appears next to the
# conversation and the results below render from the updated sink on this same run,
# instead of flashing the previous search first.
if user_text:
    with st.spinner("Sam is looking…"):
        try:
            run_agent(user_text)
        except Exception as exc:
            # A spent budget is an expected end state, not a crash. Saying "something
            # went wrong" and printing the raw exception at a stranger is the wrong
            # message at the worst moment.
            if looks_like_quota_error(exc):
                message = (
                    "That key has hit its quota or rate limit. Check the key and its "
                    "billing, then try again."
                    if own_key else ALLOWANCE_SPENT
                )
                if not own_key:
                    # Close the gate even if the counter had turns left: the budget
                    # is gone, so further attempts would just fail again.
                    free_turns_used()[visitor_id()] = FREE_TURNS
            else:
                message = "Sorry, something went wrong on my end. Please try again."
                print(f"agent error: {type(exc).__name__}: {exc}")
            st.session_state.history.append(AIMessage(message))
    # The transcript above already rendered, so a rerun is what actually shows the
    # new messages.
    st.rerun()

st.caption(
    "Requirements you state outright (fuel, transmission, body type, drivetrain, "
    "budget) are enforced exactly; the rest is matched by meaning. Sam can also "
    "look up any listing, compare cars and answer questions about what is in stock."
)

st.divider()

# The "enforcing X" and "excluding Y" banners are gone: Sam already says the same
# thing in prose, so they were the same information twice.
#
# The relaxation warning stays. It is the one case where the results do NOT match
# what was asked for, and Sam is a language model that may or may not mention it on
# any given turn. This banner fires deterministically, which is what stops the
# original bug (quietly returning cars that violate a stated requirement) coming
# back through the side door.
meta = sink.get("meta")
if meta is not None and meta.relaxed:
    st.warning(
        f"No cars in inventory match {describe(meta.relaxed)} alongside your "
        f"other requirements, so that was relaxed. Results below match "
        f"everything else you asked for."
    )

results = sink.get("results")
if results is None:
    st.info(
        "Ask Sam above and the cars will appear here. Try \"a family SUV under "
        "$15,000, automatic\" or \"what is your cheapest electric car?\"",
        icon=":material/arrow_upward:",
    )
elif results.empty:
    st.info("No cars matched. Ask Sam to relax a requirement or widen the search.")
else:
    st.subheader(f"Showing {len(results)} cars")
    st.caption("Click any card to see the full listing.")
    # Full width now, so three across instead of two.
    cols = st.columns(3)
    for i, (_, row) in enumerate(results.iterrows()):
        with cols[i % 3]:
            render_card(row)

# Popped rather than read, so dismissing the modal does not immediately reopen it
# on the next rerun. Only one dialog may be called per script run.
if "selected_car" in st.session_state:
    show_car_details(st.session_state.pop("selected_car"))
