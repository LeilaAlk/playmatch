"""PlayMatch - evaluation harness (Week 2, Q3).

Every model is evaluated the same way: it returns, for each test user, a ranked
list of item ids (games already in the user's training history are excluded).
The harness compares that list with the held-out game (leave-last-out).

Metrics
  NDCG@10    - 1 / log2(rank + 1) if the held-out game is in the top 10, else 0
  Recall@20  - 1 if the held-out game is in the top 20, else 0 (one target per user)
  Coverage@10   - share of the catalog that appears in at least one top-10 list
  AvgPop@10     - mean training popularity of the recommended games
Results are reported for all users, warm users and cold users.

Usage
  python -m src.evaluate --data data/processed --split test
  python -m src.evaluate --selftest          # checks the metric code on toy cases
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

Ranking = dict[int, list[int]]  # user_idx -> ranked item_idx


# ------------------------------------------------------------------ metrics
def ndcg_at_k(ranked: list[int], target: int, k: int) -> float:
    top = ranked[:k]
    if target not in top:
        return 0.0
    return 1.0 / np.log2(top.index(target) + 2)


def recall_at_k(ranked: list[int], target: int, k: int) -> float:
    return float(target in ranked[:k])


def evaluate(rankings: Ranking, truth: dict[int, int], n_items: int,
             popularity: np.ndarray, groups: dict[str, set[int]] | None = None) -> dict:
    groups = groups or {}
    groups = {"all": set(truth), **groups}
    out = {}
    for name, members in groups.items():
        users = [u for u in members if u in truth and u in rankings]
        if not users:
            continue
        ndcg = [ndcg_at_k(rankings[u], truth[u], 10) for u in users]
        rec = [recall_at_k(rankings[u], truth[u], 20) for u in users]
        top10 = [i for u in users for i in rankings[u][:10]]
        out[name] = {
            "users": len(users),
            "NDCG@10": round(float(np.mean(ndcg)), 5),
            "Recall@20": round(float(np.mean(rec)), 5),
            "Coverage@10": round(len(set(top10)) / n_items, 5),
            "AvgPop@10": round(float(popularity[top10].mean()), 2),
        }
    return out


# ---------------------------------------------------------------- baselines
def seen_items(train: pd.DataFrame) -> dict[int, set[int]]:
    return train.groupby("user_idx").item_idx.agg(set).to_dict()


def random_ranking(users, n_items: int, seen, k: int = 20, seed: int = 0) -> Ranking:
    rng = np.random.default_rng(seed)
    out = {}
    for u in users:
        s = seen.get(u, set())
        size = min(n_items, k + len(s))  # enough candidates to fill k after removing seen games
        cand = rng.choice(n_items, size=size, replace=False)
        out[u] = [int(i) for i in cand if i not in s][:k]
    return out


def most_popular_ranking(users, popularity: np.ndarray, seen, k: int = 20) -> Ranking:
    order = np.argsort(-popularity, kind="stable")
    out = {}
    for u in users:
        s = seen.get(u, set())
        out[u] = [int(i) for i in order[: k + len(s)] if i not in s][:k]
    return out


# ----------------------------------------------------------------- selftest
def selftest() -> None:
    assert ndcg_at_k([5, 1, 2], 5, 10) == 1.0
    assert abs(ndcg_at_k([1, 5, 2], 5, 10) - 1 / np.log2(3)) < 1e-12
    assert ndcg_at_k(list(range(20)), 15, 10) == 0.0
    assert recall_at_k(list(range(20)), 15, 20) == 1.0
    assert recall_at_k(list(range(20)), 25, 20) == 0.0
    pop = np.ones(30)
    perfect = evaluate({0: [3, 1], 1: [7, 2]}, {0: 3, 1: 7}, 30, pop)
    assert perfect["all"]["NDCG@10"] == 1.0 and perfect["all"]["Recall@20"] == 1.0
    print("selftest passed: NDCG@10 and Recall@20 give the expected values on toy cases")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out", default="results")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    data = Path(args.data)
    train = pd.read_parquet(data / "train.parquet")
    target = pd.read_parquet(data / f"{args.split}.parquet")
    users = pd.read_parquet(data / "users.parquet")
    n_items = len(pd.read_parquet(data / "items.parquet"))

    popularity = np.bincount(train[train.label == 1].item_idx, minlength=n_items).astype(float)
    truth = dict(zip(target.user_idx, target.item_idx))
    seen = seen_items(train)
    groups = {"warm": set(users[~users.is_cold].user_idx), "cold": set(users[users.is_cold].user_idx)}

    baselines: dict[str, Callable[[], Ranking]] = {
        "random": lambda: random_ranking(truth, n_items, seen),
        "most_popular": lambda: most_popular_ranking(truth, popularity, seen),
    }
    results = {name: evaluate(fn(), truth, n_items, popularity, groups) for name, fn in baselines.items()}

    rows = [{"model": m, "group": g, **v} for m, r in results.items() for g, v in r.items()]
    table = pd.DataFrame(rows)
    print(f"\nSplit: {args.split} | users: {len(truth):,} | catalog: {n_items:,} games\n")
    print(table.to_string(index=False))
    Path(args.out).mkdir(exist_ok=True)
    out_file = Path(args.out) / f"baselines_{args.split}.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
