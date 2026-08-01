"""Step 2 — score every listing against six "vibes" with zero-shot classification.

Run with:  uv run python -m pipeline.vibe_tagging

Slow: roughly 7 minutes per 1,000 rows on Apple MPS, so ~110 minutes for the full
table. Checkpoints every 20 batches, and resumes from the checkpoint if restarted.

Note the scores are a softmax across the six labels (multi_label=False), so they are
competitive shares rather than independent confidences — vibe_family averages 0.31
and is the top label for 37% of rows. Treat them as a weak ranking prior, not a
filter. search_engine.py z-scores them before blending for exactly this reason.

Input:  data/cars_raw.csv
Output: data/cars_with_vibes_v2.csv
"""
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from showroom.config import CARS_CSV, DATA_DIR, VIBE_CHECKPOINT, VIBE_MODEL

IN_PATH = DATA_DIR / "cars_raw.csv"
BATCH_SIZE = 16

VIBE_LABEL_MAP = {
    "a family-friendly car with practical space": "vibe_family",
    "a sporty high-performance car": "vibe_sporty",
    "a luxury car with premium features": "vibe_luxury",
    "a fuel-efficient eco-friendly car": "vibe_ecofriendly",
    "an off-road or rugged utility vehicle": "vibe_offroad",
    "a basic reliable daily commuter car": "vibe_commuter",
}
VIBE_LABELS = list(VIBE_LABEL_MAP)
HYPOTHESIS = "This vehicle listing describes {}."


def build_classifier_input(row):
    car_type = row["type"] if pd.notna(row["type"]) else "unknown"
    type_phrase = f"a {car_type}" if car_type != "unknown" else "a vehicle"
    condition = row["condition"] if pd.notna(row["condition"]) else "unknown"
    return (
        f"{int(row['year'])} {row['manufacturer']} {row['model']}, "
        f"{type_phrase} in {condition} condition. {row['description_clean']}"
    )


def main():
    cars = pd.read_csv(IN_PATH)
    texts = cars.apply(build_classifier_input, axis=1).tolist()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Tagging {len(texts):,} rows on {device}")
    classifier = pipeline("zero-shot-classification", model=VIBE_MODEL, device=device)

    results = []
    if VIBE_CHECKPOINT.exists():
        results = pd.read_csv(VIBE_CHECKPOINT).to_dict("records")
        print(f"Resuming from row {len(results)}")

    start = time.time()
    for i in tqdm(range(len(results), len(texts), BATCH_SIZE)):
        batch = classifier(
            texts[i:i + BATCH_SIZE], VIBE_LABELS, multi_label=False,
            hypothesis_template=HYPOTHESIS, batch_size=BATCH_SIZE,
        )
        results.extend(dict(zip(r["labels"], r["scores"])) for r in batch)
        if (i // BATCH_SIZE) % 20 == 0:
            pd.DataFrame(results).to_csv(VIBE_CHECKPOINT, index=False)
            print(f"  checkpoint @ {len(results):,} rows, "
                  f"{(time.time() - start) / 60:.1f} min", flush=True)

    vibes = pd.DataFrame(results).rename(columns=VIBE_LABEL_MAP)
    tagged = pd.concat([cars.reset_index(drop=True), vibes], axis=1)
    tagged.to_csv(CARS_CSV, index=False)

    VIBE_CHECKPOINT.unlink(missing_ok=True)
    print(f"\nwrote {CARS_CSV}: {tagged.shape}")


if __name__ == "__main__":
    main()
