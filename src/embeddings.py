"""PlayMatch - semantic embeddings of the games (Week 3, step 1).

Encodes title + publisher + category path + description of each game with BGE
(the text encoder used by GenSAR). Uses item metadata only, so no leakage.

Output: data/processed/items_bge.npy  (one L2-normalised row per item_idx)

Usage
  python -m src.embeddings --data data/processed
  python -m src.embeddings --data data/processed --offline   # tests without internet
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.text_encoder import TextEncoder, item_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    data = Path(args.data)
    items = pd.read_parquet(data / "items.parquet").sort_values("item_idx")
    assert (items.item_idx.to_numpy() == np.arange(len(items))).all(), "item_idx must be 0..n-1"
    texts = [item_text(r) for r in items.to_dict("records")]

    t0 = time.time()
    enc = TextEncoder.build(texts, args.offline, data)
    emb = enc.encode_items(texts)
    np.save(data / "items_bge.npy", emb)
    print(f"Encoder: {enc.mode} | {emb.shape[0]:,} games x {emb.shape[1]} dims | {time.time() - t0:.0f}s")

    # Sanity check: nearest neighbours of a few games should look related.
    titles = items.title.to_numpy()
    for i in np.random.default_rng(0).choice(len(items), size=min(3, len(items)), replace=False):
        sims = emb @ emb[i]
        nn = np.argsort(-sims)[1:4]
        print(f"  {titles[i][:45]!r} -> " + "; ".join(f"{titles[j][:35]!r}" for j in nn))
    print(f"Saved {data / 'items_bge.npy'}")


if __name__ == "__main__":
    main()
