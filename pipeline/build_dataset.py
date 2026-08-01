"""Step 1 — build the listing table from the raw Kaggle dump.

Run with:  uv run python -m pipeline.build_dataset

Two things happen here, and the second one matters more than it looks.

A plain sample stratified by body type (1000 per type) gives an even spread of
sedans/SUVs/trucks but leaves fuel wildly imbalanced — the first version of this
dataset ended up 90.6% gas with 37 electric cars out of 12,429, and exactly ONE
electric sedan. No amount of ranking quality fixes "the car you asked for is not
in the table", so the rare fuels are topped up deliberately afterwards.

Output: data/cars_raw.csv  (~14.8k rows, untagged)
"""
import re

import pandas as pd

from showroom.config import DATA_DIR, KAGGLE_DATASET

OUT_PATH = DATA_DIR / "cars_raw.csv"

TYPE_CAP = 1000                 # per body type in the base sample
FUEL_TARGETS = {                # total rows per rare fuel (None = take all available)
    "electric": None,
    "hybrid": None,
    "diesel": 1500,
    "other": 400,
}
SEED = 42


def clean_description(text):
    text = re.sub(r"http\S+|www\.\S+", " ", text)                    # URLs
    text = re.sub(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", " ", text)  # phone numbers
    text = re.sub(r"[☏✅🚘📞]+", " ", text)                            # symbols
    return re.sub(r"\s+", " ", text).strip()


def clean_pool(cars):
    """Drop junk listings. The word-count band cuts both near-empty ads and the
    2000-word dealer boilerplate spam at the other extreme."""
    cars = cars[cars["description"].notna()].copy()
    cars = cars.drop_duplicates(subset="description", keep="first")
    words = cars["description"].str.split().str.len()
    cars = cars[words.between(15, 200)]
    cars = cars[cars["price"].between(500, 100_000)]
    cars = cars[cars["odometer"].between(500, 300_000)]
    cars = cars[cars["year"].between(1990, 2022)]
    return cars


def main():
    import kagglehub

    path = kagglehub.dataset_download(KAGGLE_DATASET)
    pool = clean_pool(pd.read_csv(f"{path}/vehicles.csv", low_memory=False))
    print(f"cleaned pool: {len(pool):,}")

    # Base: even coverage across body types.
    base_idx = pool.groupby("type").apply(
        lambda g: g.sample(n=min(len(g), TYPE_CAP), random_state=SEED).index,
        include_groups=False,
    )
    base = pool.loc[[i for idx in base_idx for i in idx]]
    print(f"base sample: {len(base):,}")

    # Top-up: rescue the rare fuels the type-stratified sample threw away.
    have = base["fuel"].value_counts()
    candidates = pool[~pool["id"].isin(set(base["id"]))]
    picks = [base]
    for fuel, target in FUEL_TARGETS.items():
        avail = candidates[candidates["fuel"] == fuel]
        already = int(have.get(fuel, 0))
        take = len(avail) if target is None else max(0, min(target - already, len(avail)))
        if take:
            picks.append(avail.sample(n=take, random_state=SEED))
        print(f"  {fuel:9s} base {already:5,}  available {len(avail):5,}  adding {take:5,}")

    cars = pd.concat(picks).reset_index(drop=True)
    cars["description_clean"] = cars["description"].apply(clean_description)

    DATA_DIR.mkdir(exist_ok=True)
    cars.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}: {cars.shape}")
    print(cars["fuel"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
