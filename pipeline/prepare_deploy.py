"""Step 4: turn the built index into artifacts small enough to commit.

Run with:  uv run python -m pipeline.prepare_deploy

The working artifacts are 116 MB, which is awkward in git and slow to clone. Two
lossless-enough changes take that to 47 MB:

  cars_with_vibes_v2.csv   28.9 MB  ->  cars.parquet          3.9 MB
  car_embeddings_v2.npy    86.9 MB  ->  embeddings_f16.npy   43.4 MB

The CSV shrinks because parquet is columnar and compressed, and because eight
columns the application never reads (raw description, urls, lat/long, county) are
dropped. The embeddings halve by moving to float16.

float16 is safe here. Cosine similarity over this corpus spans 0.69 to 0.84 with a
standard deviation of 0.019, which is far coarser than float16 resolution, and the
top-k ordering is unchanged. The array is cast back to float32 once at load time,
so runtime maths and memory are identical to before; only the file on disk is
smaller.
"""
import re

import numpy as np
import pandas as pd

from showroom.config import (CARS_CSV, CARS_PARQUET, EMBEDDINGS_F16,
                             EMBEDDINGS_NPY)

# Everything the app or the agent reads. Anything absent here is dropped.
KEEP = [
    "id", "price", "year", "manufacturer", "model", "condition", "cylinders",
    "fuel", "odometer", "title_status", "transmission", "VIN", "drive", "size",
    "type", "paint_color", "region", "state", "posting_date", "description_clean",
    "vibe_offroad", "vibe_family", "vibe_sporty", "vibe_commuter", "vibe_luxury",
    "vibe_ecofriendly",
]

# This artifact is committed to a public repository, so the seller text gets a
# second scrub. The pipeline's original clean already removed most contact details,
# but a handful of formats survived it: twenty rows still carried phone numbers,
# email addresses or urls. These patterns are deliberately broader than the first
# pass, since a false positive here costs nothing and a miss publishes a stranger's
# phone number.
CONTACT_PATTERNS = [
    r"[\w.+-]+@[\w-]+\.[\w.]{2,}",                        # email
    # The pipeline's own phone pattern, kept verbatim. A "looser" rewrite turned out
    # to be narrower on "(541) 480-3265", the commonest US format, and let a real
    # number through. It also matches innocent digit runs such as a mileage next to
    # a model year; that is the right trade here, because the structured columns
    # carry those facts and the description is only prose.
    r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
    r"\+?\d{1,2}[\s.-]\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",   # with country code
    # Trailing whitespace is deliberate: the first cleaning pass leaves "www. site.com"
    # when it strips something between, and "www\.\S+" then fails to match.
    r"https?://\s*\S+|www\.\s*\S+",
]
CONTACT = re.compile("|".join(CONTACT_PATTERNS), re.IGNORECASE)


def scrub(text):
    return re.sub(r"\s+", " ", CONTACT.sub(" ", str(text))).strip()


def main():
    cars = pd.read_csv(CARS_CSV)
    embeddings = np.load(EMBEDDINGS_NPY)
    assert len(cars) == embeddings.shape[0], "cars/embeddings row count mismatch"

    missing = [c for c in KEEP if c not in cars.columns]
    if missing:
        raise SystemExit(f"source is missing expected columns: {missing}")

    slim = cars[KEEP].copy()
    before = slim["description_clean"].astype(str).str.contains(CONTACT, na=False).sum()
    slim["description_clean"] = slim["description_clean"].map(scrub)
    after = slim["description_clean"].str.contains(CONTACT, na=False).sum()
    print(f"contact details scrubbed from seller text: {before} rows -> {after}")
    if after:
        raise SystemExit("contact details survived the scrub, do not publish this")

    slim.to_parquet(CARS_PARQUET, index=False, compression="zstd")

    half = embeddings.astype(np.float16)
    np.save(EMBEDDINGS_F16, half)

    # Ranking must not move. Compare top-20 for a handful of real rows as queries.
    ref = embeddings.astype(np.float32)
    back = half.astype(np.float32)
    rng = np.random.default_rng(0)
    drift = 0
    for i in rng.choice(len(ref), 25, replace=False):
        q = ref[i]
        a = np.argsort((ref @ q) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(q)))[::-1][:20]
        b = np.argsort((back @ q) / (np.linalg.norm(back, axis=1) * np.linalg.norm(q)))[::-1][:20]
        drift += int(not np.array_equal(a, b))

    mb = lambda p: p.stat().st_size / 1048576
    print(f"{CARS_PARQUET.name:22s} {mb(CARS_PARQUET):6.1f} MB  ({slim.shape[1]} of "
          f"{cars.shape[1]} columns kept)")
    print(f"{EMBEDDINGS_F16.name:22s} {mb(EMBEDDINGS_F16):6.1f} MB  {half.shape} float16")
    print(f"total {mb(CARS_PARQUET) + mb(EMBEDDINGS_F16):.1f} MB")
    print(f"\ntop-20 ordering changed on {drift} of 25 sample queries")
    if drift:
        raise SystemExit("float16 altered the ranking, do not ship this")


if __name__ == "__main__":
    main()
