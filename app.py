import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from showroom.sales_agent import build_agent
from showroom.search_engine import VIBE_MAP, get_embedder, load_data, search_cars

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

with st.sidebar:
    st.header("Filters")
    st.caption("Sidebar filters always win over anything detected in your search text.")

    type_options = ["All"] + sorted(cars["type"].unique().tolist())
    type_filter = st.selectbox("Body type", type_options)

    manufacturer_options = ["All"] + sorted(cars["manufacturer"].dropna().unique().tolist())
    manufacturer_filter = st.selectbox("Manufacturer", manufacturer_options)

    fuel_options = ["All"] + sorted(cars["fuel"].unique().tolist())
    fuel_filter = st.selectbox("Fuel type", fuel_options)

    transmission_options = ["All"] + sorted(cars["transmission"].unique().tolist())
    transmission_filter = st.selectbox("Transmission", transmission_options)

    drive_options = ["All"] + sorted(cars["drive"].unique().tolist())
    drive_filter = st.selectbox("Drivetrain", drive_options)

    condition_options = ["All"] + sorted(cars["condition"].unique().tolist())
    condition_filter = st.selectbox("Condition", condition_options)

    price_min, price_max = int(cars["price"].min()), int(cars["price"].max())
    price_range = st.slider("Price range ($)", price_min, price_max, (price_min, price_max))

    vibe_options = ["All"] + list(VIBE_MAP.keys())
    vibe_filter = st.selectbox("Vibe", vibe_options)

    top_k = st.slider("Number of results", 4, 24, 12)

    st.divider()
    if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.history = []
        st.rerun()


def describe(spec):
    return ", ".join(f"**{FIELD_LABELS.get(k, k)} = {v}**" for k, v in spec.items())


def render_card(row):
    with st.container(border=True):
        # Listings are missing any of year / manufacturer / model, so build the
        # heading from whatever is actually present instead of assuming all three.
        make = row["manufacturer"] if row["manufacturer"] != "unknown" else ""
        parts = [
            str(int(row["year"])) if pd.notna(row["year"]) else "",
            str(make).title(),
            str(row["model"]).title(),
        ]
        st.subheader(" ".join(p for p in parts if p) or "Listing")

        price = f"${int(row['price']):,}" if pd.notna(row["price"]) else "price n/a"
        miles = f"{row['odometer']:,.0f} mi" if pd.notna(row["odometer"]) else "mileage n/a"
        st.markdown(f"**{price}**  ·  {row['type']}  ·  {miles}")
        st.markdown(
            f"`{row['fuel']}` `{row['transmission']}` `{row['drive']}` `{row['condition']}`"
        )
        desc = row["description_clean"]
        st.caption(desc[:180] + ("..." if len(desc) > 180 else ""))
        st.caption(f"Listing id `{row['id']}` — quote this to Sam to ask about it.")


def run_box_search(query):
    found = search_cars(
        query=query, cars=cars, embeddings=embeddings, embedder=embedder,
        filters={
            "type": type_filter, "manufacturer": manufacturer_filter,
            "fuel": fuel_filter, "transmission": transmission_filter,
            "drive": drive_filter, "condition": condition_filter,
            "price_range": price_range,
        },
        vibe_label=vibe_filter, top_k=top_k,
    )
    sink["results"] = found.results
    sink["label"] = query
    sink["meta"] = found


def run_agent(user_text, from_box=False):
    """One agent turn. Tools mutate `sink`, so the grid updates as a side effect.

    `from_box` marks a turn where the customer used the search box rather than the
    chat: the search has already run, so Sam comments on the results instead of
    searching again. The flag lives on the sink because that is what the agent's
    prompt reads, and it is cleared afterwards so it only affects this one turn.
    """
    sink["from_box"] = from_box
    try:
        st.session_state.history.append(HumanMessage(user_text))
        result = st.session_state.agent.invoke({"messages": st.session_state.history})
        st.session_state.history = result["messages"]
    finally:
        sink["from_box"] = False


search_col, chat_col = st.columns([3, 2], gap="large")

with search_col:
    st.markdown(
        "Describe the car you're looking for in plain language — or just ask Sam "
        "on the right."
    )

    with st.form("search_form"):
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., a spacious car that feels great for long road trips",
        )
        st.caption(
            "Requirements you state outright — fuel, transmission, body type, "
            "drivetrain, and budget (\"under $5,000\") — are enforced exactly. "
            "Everything else (spacious, sporty, reliable) is matched by meaning."
        )
        submit = st.form_submit_button(
            "Find cars", type="primary", icon=":material/search:", width="stretch"
        )

    if submit and query.strip():
        with st.spinner("Searching..."):
            run_box_search(query)
        # Sam reacts to what the search turned up, so the chat stays in step with
        # the grid however the customer chose to drive it.
        with st.spinner("Sam is looking over the results…"):
            try:
                run_agent(query, from_box=True)
            except Exception as exc:
                st.session_state.history.append(
                    AIMessage(f"(Sam couldn't comment on these: {exc})")
                )
    elif submit:
        st.warning("Type a description of what you're looking for first.")

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
        st.info("Search above, or ask Sam on the right, to get started.")
    elif results.empty:
        st.info("No cars matched. Try loosening a filter or broadening your search.")
    else:
        st.subheader(f"Showing {len(results)} cars")
        cols = st.columns(2)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 2]:
                render_card(row)

with chat_col:
    st.subheader("Sam the salesman")
    st.caption(
        "Sam can search the lot, look up any listing, compare cars and answer "
        "questions about what's in stock."
    )

    transcript = st.container(height=520, border=True)
    with transcript:
        if not st.session_state.history:
            st.chat_message("assistant").write(
                "Hi, I'm Sam. Tell me what you're after — budget, body type, "
                "how you'll use it — and I'll pull what we've got on the lot."
            )
        for msg in st.session_state.history:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, AIMessage):
                # Surface tool calls so it's clear what Sam actually looked up
                # rather than what he might have made up.
                for call in msg.tool_calls or []:
                    args = {k: v for k, v in call["args"].items() if v not in (None, "")}
                    st.caption(f"🔎 `{call['name']}` {args}")
                if msg.content:
                    st.chat_message("assistant").write(msg.content)

    user_text = st.chat_input("Ask Sam about the cars…")
    if user_text:
        with st.spinner("Sam is looking…"):
            try:
                run_agent(user_text)
            except Exception as exc:
                st.session_state.history.append(
                    AIMessage(f"Sorry — something went wrong on my end: {exc}")
                )
        st.rerun()
