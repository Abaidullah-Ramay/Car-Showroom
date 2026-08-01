"""Project paths and model choices, in one place.

Paths are anchored to the repo root rather than the working directory, so the app,
the pipeline scripts and the tests all resolve the same files no matter where they
are launched from.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Built by pipeline/build_topup.py -> vibe_tagging_topup.py -> build_embeddings.py
CARS_CSV = DATA_DIR / "cars_with_vibes_v2.csv"
EMBEDDINGS_NPY = DATA_DIR / "car_embeddings_v2.npy"

# Intermediates, all regenerable and all gitignored.
TOPUP_CSV = DATA_DIR / "cars_topup_raw.csv"
VIBE_CHECKPOINT = DATA_DIR / "vibe_checkpoint_topup.csv"
EMBED_CHECKPOINT = DATA_DIR / "embeddings_checkpoint.npy"

# Kaggle source, downloaded by the exploration notebook via kagglehub.
KAGGLE_DATASET = "austinreese/craigslist-carstrucks-data"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.4-mini"
VIBE_MODEL = "facebook/bart-large-mnli"
