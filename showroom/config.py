"""Project paths and model choices, in one place.

Paths are anchored to the repo root rather than the working directory, so the app,
the pipeline scripts and the tests all resolve the same files no matter where they
are launched from.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Working artifacts, produced by the pipeline. Large, gitignored, not deployed.
CARS_CSV = DATA_DIR / "cars_with_vibes_v2.csv"
EMBEDDINGS_NPY = DATA_DIR / "car_embeddings_v2.npy"

# Deployment artifacts, produced by pipeline/prepare_deploy.py. These are what the
# app loads and what is committed: 47 MB instead of 116 MB. See that script for why
# float16 is safe.
CARS_PARQUET = DATA_DIR / "cars.parquet"
EMBEDDINGS_F16 = DATA_DIR / "embeddings_f16.npy"

# Intermediates, all regenerable and all gitignored.
TOPUP_CSV = DATA_DIR / "cars_topup_raw.csv"
VIBE_CHECKPOINT = DATA_DIR / "vibe_checkpoint_topup.csv"
EMBED_CHECKPOINT = DATA_DIR / "embeddings_checkpoint.npy"

# Kaggle source, downloaded by the exploration notebook via kagglehub.
KAGGLE_DATASET = "austinreese/craigslist-carstrucks-data"

EMBED_MODEL = "text-embedding-3-small"
# Pinned to the dated snapshot rather than the moving `gpt-5.4-mini` alias, so it
# matches the deployment project's allowed-models list exactly. If the alias is
# ever repointed at a newer snapshot, an unpinned app breaks in production on a
# date nobody chose.
CHAT_MODEL = "gpt-5.4-mini-2026-03-17"
VIBE_MODEL = "facebook/bart-large-mnli"

# Free turns per visitor before they are asked for their own API key. The real
# spend protection is the project budget and rate limits set on the OpenAI side;
# this only stops one visitor consuming the whole demo allowance.
FREE_TURNS = 3
