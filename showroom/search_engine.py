from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from showroom.config import CARS_PARQUET, EMBED_MODEL, EMBEDDINGS_F16
from showroom.query_parser import parse_query

load_dotenv()

CATEGORICAL_COLS = ["type", "manufacturer", "fuel", "transmission", "drive", "condition"]

VIBE_MAP = {
    "Family": "vibe_family",
    "Sporty": "vibe_sporty",
    "Luxury": "vibe_luxury",
    "Eco-friendly": "vibe_ecofriendly",
    "Off-road": "vibe_offroad",
    "Commuter": "vibe_commuter",
}

# Order in which query-derived constraints get dropped when nothing matches all of
# them. Drivetrain is the most incidental ask; fuel is the most deliberate — someone
# who typed "electric" did not mean "or gas is fine", so it goes last.
RELAX_ORDER = ["drive", "type", "transmission", "fuel"]


@dataclass
class SearchResult:
    """Results plus the reasoning behind them, so the UI can explain itself."""
    results: pd.DataFrame
    constraints: dict = field(default_factory=dict)   # enforced, from query text
    relaxed: dict = field(default_factory=dict)       # asked for, but nothing matched
    excluded: dict = field(default_factory=dict)      # negated in the query
    semantic_query: str = ""
    price_range: tuple = (None, None)                 # budget actually applied
    price_from_query: bool = False                    # budget came from the text


def build_embedding_text(row):
    """Structured attributes first, then the ad copy.

    The original index embedded `id + raw craigslist description`, which meant the
    clean `fuel` / `transmission` / `type` columns were never in the vector at all —
    a query saying "automatic" had nothing reliable to match against, and the leading
    numeric id was pure noise. Leading with the structured facts puts the attributes
    people actually search on at the front of the text.
    """
    def val(col, default="unknown"):
        v = row.get(col)
        return default if pd.isna(v) else v

    year = val("year")
    year = str(int(year)) if year != "unknown" else ""
    odometer = row.get("odometer")
    mileage = f"{int(odometer):,} miles" if pd.notna(odometer) else "mileage unknown"

    facts = " ".join(str(x) for x in [
        year, val("manufacturer", ""), val("model", ""),
    ] if x).strip()

    return (
        f"{facts}. "
        f"{val('type')} body, {val('fuel')} fuel, {val('transmission')} transmission, "
        f"{val('drive')} drivetrain, {val('condition')} condition, {mileage}. "
        f"{val('description_clean', '')}"
    ).strip()


def index_exists():
    return CARS_PARQUET.exists() and EMBEDDINGS_F16.exists()


def load_data():
    if not index_exists():
        raise FileNotFoundError(
            f"No search index found. Expected {CARS_PARQUET} and {EMBEDDINGS_F16}.\n"
            f"Build them with:\n"
            f"  uv run python -m pipeline.build_dataset\n"
            f"  uv run python -m pipeline.vibe_tagging\n"
            f"  uv run python -m pipeline.build_embeddings\n"
            f"  uv run python -m pipeline.prepare_deploy"
        )

    cars = pd.read_parquet(CARS_PARQUET)
    # Stored as float16 to halve the file on disk, widened once here so the dot
    # product runs at full speed and memory matches the previous behaviour.
    embeddings = np.load(EMBEDDINGS_F16).astype(np.float32)
    assert len(cars) == embeddings.shape[0], "cars/embeddings row count mismatch"

    for col in CATEGORICAL_COLS:
        cars[col] = cars[col].fillna("unknown")

    # Display-only text columns get an empty string rather than "unknown" — a card
    # reading "2015 Rover" beats "2015 Rover Unknown". Filling them at all matters
    # because a NaN here is a float, and every caller treats these as strings.
    for col in ["model", "description_clean"]:
        cars[col] = cars[col].fillna("").astype(str)

    return cars, embeddings


def get_embedder(api_key=None):
    """Must match the model the index was built with: vectors from different models
    are not comparable, so changing EMBED_MODEL means rebuilding the index.

    `api_key` lets a visitor supply their own credentials once the free allowance is
    used up. None falls back to the environment.
    """
    kwargs = {"model": EMBED_MODEL}
    if api_key:
        kwargs["api_key"] = api_key
    return OpenAIEmbeddings(**kwargs)


def _zscore(series):
    """Put similarity and vibe on the same scale before blending.

    Cosine similarities over this corpus span only ~0.15 (std 0.019) while vibe
    scores span the full 0-1, so a raw `0.7*sim + 0.3*vibe` let the 0.3 term drive
    roughly three times more variance than the 0.7 term — the weights meant the
    opposite of what they read as.
    """
    std = series.std()
    if not np.isfinite(std) or std < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def _apply_exact(df, spec):
    for col, val in spec.items():
        if val and val != "All":
            df = df[df[col] == val]
    return df


def search_cars(query, cars, embeddings, embedder, filters=None,
                vibe_label=None, top_k=12, parse_constraints=True):
    filters = filters or {}

    constraints, semantic_query, excluded = ({}, query, {})
    parsed_price = (None, None)
    if parse_constraints:
        constraints, semantic_query, excluded, parsed_price = parse_query(query)

    # An explicit sidebar choice beats whatever we inferred from the prose.
    sidebar = {c: filters.get(c) for c in CATEGORICAL_COLS
               if filters.get(c) and filters.get(c) != "All"}
    constraints = {k: v for k, v in constraints.items() if k not in sidebar}

    query_vec = np.array(embedder.embed_query(semantic_query))
    sims = (embeddings @ query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    df = cars.copy()
    df["similarity"] = sims

    # Sidebar filters and price are never relaxed — the user set them deliberately.
    df = _apply_exact(df, sidebar)

    # Budget from the slider and budget from the text are both real requirements, so
    # they intersect rather than override — the tighter of the two bounds wins. A
    # stated budget is never relaxed; someone who says "under $5,000" means it.
    low, high = filters.get("price_range") or (None, None)
    parsed_low, parsed_high = parsed_price
    if parsed_low is not None:
        low = parsed_low if low is None else max(low, parsed_low)
    if parsed_high is not None:
        high = parsed_high if high is None else min(high, parsed_high)

    price_range = (low, high)
    if low is not None:
        df = df[df["price"] >= low]
    if high is not None:
        df = df[df["price"] <= high]

    for col, val in excluded.items():
        df = df[df[col] != val]

    # Query-derived constraints are relaxed one at a time, weakest first, rather
    # than silently returning cars that violate what the user typed.
    active = dict(constraints)
    relaxed = {}
    candidates = _apply_exact(df, active)
    for col in RELAX_ORDER:
        if not candidates.empty:
            break
        if col in active:
            relaxed[col] = active.pop(col)
            candidates = _apply_exact(df, active)

    if candidates.empty:
        return SearchResult(candidates, active, relaxed, excluded, semantic_query,
                            price_range, any(v is not None for v in parsed_price))

    candidates = candidates.copy()
    if vibe_label and vibe_label != "All":
        vibe_col = VIBE_MAP[vibe_label]
        candidates["score"] = (
            0.7 * _zscore(candidates["similarity"])
            + 0.3 * _zscore(candidates[vibe_col])
        )
    else:
        candidates["score"] = candidates["similarity"]

    results = candidates.sort_values("score", ascending=False).head(top_k)
    return SearchResult(results, active, relaxed, excluded, semantic_query,
                        price_range, any(v is not None for v in parsed_price))
