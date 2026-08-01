"""Step 3 — embed every listing.

Run with:  uv run python -m pipeline.build_embeddings

The previous index embedded `id + raw description`, so the clean fuel / transmission
/ type columns never made it into the vectors. See build_embedding_text().

Writes data/car_embeddings_v2.npy, row-aligned with the CSV by construction
(plain positional list -> np.array, no vector store round-trip that could reorder).
"""
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from showroom.config import CARS_CSV, EMBED_CHECKPOINT, EMBED_MODEL, EMBEDDINGS_NPY
from showroom.search_engine import build_embedding_text

load_dotenv()

BATCH_SIZE = 256
CHECKPOINT = EMBED_CHECKPOINT


def main():
    cars = pd.read_csv(CARS_CSV)
    texts = [build_embedding_text(row) for _, row in cars.iterrows()]
    print(f"{len(texts):,} rows to embed with {EMBED_MODEL}")
    print("\nsample embedding text:\n" + texts[0][:300] + "\n")

    embedder = OpenAIEmbeddings(model=EMBED_MODEL)

    vectors = []
    if os.path.exists(CHECKPOINT):
        vectors = list(np.load(CHECKPOINT))
        print(f"Resuming from row {len(vectors)}")

    for i in tqdm(range(len(vectors), len(texts), BATCH_SIZE)):
        batch = texts[i:i + BATCH_SIZE]
        vectors.extend(embedder.embed_documents(batch))
        if (i // BATCH_SIZE) % 10 == 0:
            np.save(CHECKPOINT, np.array(vectors, dtype=np.float32))

    embeddings = np.array(vectors, dtype=np.float32)
    assert embeddings.shape[0] == len(cars), "row count mismatch"
    np.save(EMBEDDINGS_NPY, embeddings)
    print(f"\nwrote {EMBEDDINGS_NPY}: {embeddings.shape}")

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)


if __name__ == "__main__":
    main()
