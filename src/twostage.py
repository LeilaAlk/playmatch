"""PlayMatch - plan B / two-stage hybrid (Week 3).

Stage 1: BGE retrieves the 100 games closest to the request (after the player's hard limits).
Stage 2: re-rank them with   w_text * text similarity
                           + w_collab * NLGCL score (player vector . game vector)
                           + w_profile * share of the game's platform in the player's profile
(text and collaborative scores are standardised inside the 100 candidates).
Cold users have no NLGCL vector: we use the mean vector of the games of their onboarding profile.
The weights are tuned on the VALIDATION requests only, then frozen for the test.

Usage
  python -m src.twostage --data data/processed --split test
  python -m src.twostage --data data/processed --split test --collab lightgcn     # ablation
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.context import Context, load_context
from src.evaluate import evaluate
from src.text_encoder import TextEncoder

N_CANDIDATES = 100


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else np.zeros_like(x)


class TwoStage:
    name = "twostage"

    def __init__(self, ctx: Context, collab: str = "nlgcl", weights: dict | None = None):
        d = ctx.data
        self.ctx = ctx
        self.text = np.load(d / "items_bge.npy")
        self.user_vec = np.load(d / f"{collab}_users.npy")
        self.item_vec = np.load(d / f"{collab}_items.npy")
        self.encoder = None
        self.weights = weights or {"w_text": 1.0, "w_collab": 0.5, "w_profile": 0.5}

    def encode(self, queries: list[str]) -> np.ndarray:
        if self.encoder is None:
            self.encoder = TextEncoder.load(self.ctx.data)
        return self.encoder.encode_queries(queries)

    def player_vector(self, user_idx: int | None, hist: dict) -> np.ndarray | None:
        if user_idx is not None and user_idx not in self.ctx.cold and user_idx < len(self.user_vec):
            return self.user_vec[user_idx]
        liked = hist["item_idx"][hist["label"] == 1]
        return self.item_vec[liked].mean(0) if len(liked) else None

    def platform_affinity(self, hist: dict, override: dict | None) -> dict[str, float]:
        if override and override.get("platforms"):
            return {p: 1.0 / len(override["platforms"]) for p in override["platforms"]}
        liked = hist["item_idx"][(hist["label"] == 1) & ~hist["gift"]]
        if not len(liked):
            return {}
        vals, counts = np.unique(self.ctx.platform[liked], return_counts=True)
        return dict(zip(vals, counts / counts.sum()))

    def recommend(self, query_vec: np.ndarray, hist: dict, user_idx: int | None = None, k: int = 20,
                  platform: str | None = None, max_price: float | None = None, override: dict | None = None,
                  weights: dict | None = None) -> list[int]:
        w = weights or self.weights
        allowed = self.ctx.allowed_mask(platform, max_price)
        allowed[hist["item_idx"]] = False  # never recommend a game already reviewed
        cand = np.where(allowed)[0]
        if not len(cand):
            return []
        sim = self.text[cand] @ query_vec
        top = np.argsort(-sim)[:N_CANDIDATES]
        cand, sim = cand[top], sim[top]
        score = w["w_text"] * zscore(sim)
        pv = self.player_vector(user_idx, hist)
        if pv is not None and w["w_collab"]:
            score += w["w_collab"] * zscore(self.item_vec[cand] @ pv)
        aff = self.platform_affinity(hist, override)
        if aff and w["w_profile"]:
            score += w["w_profile"] * np.array([aff.get(p, 0.0) for p in self.ctx.platform[cand]])
        return [int(i) for i in cand[np.argsort(-score)][:k]]


def run_split(model: TwoStage, rows: pd.DataFrame, qvecs: np.ndarray, onboarding: bool = True,
              weights: dict | None = None, k: int = 20) -> dict[int, list[int]]:
    out = {}
    for r, qv in zip(rows.itertuples(), qvecs):
        hist = model.ctx.history(int(r.user_idx), onboarding=onboarding)
        lp = r.limit_platform if isinstance(r.limit_platform, str) else None
        out[int(r.user_idx)] = model.recommend(qv, hist, int(r.user_idx) if onboarding else None, k, lp,
                                               r.limit_max_price, weights=weights)
    return out


def tune(model: TwoStage, valid_rows: pd.DataFrame, ablation: str | None = None) -> dict:
    qv = model.encode(valid_rows["query"].tolist())
    truth = dict(zip(valid_rows.user_idx, valid_rows.item_idx))
    grid_c = [0.0] if ablation in ("no_collab",) else [0.0, 0.25, 0.5, 1.0, 2.0]
    grid_p = [0.0] if ablation in ("no_profile", "no_collab") else [0.0, 0.25, 0.5, 1.0]
    best, best_w = -1.0, None
    for wc, wp in itertools.product(grid_c, grid_p):
        w = {"w_text": 1.0, "w_collab": wc, "w_profile": wp}
        res = evaluate(run_split(model, valid_rows, qv, weights=w), truth, model.ctx.n_items, model.ctx.popularity)
        if res["all"]["NDCG@10"] > best:
            best, best_w = res["all"]["NDCG@10"], w
    print(f"  tuned on {len(valid_rows):,} validation requests ({ablation or 'full'}): {best_w} -> val NDCG@10 {best:.4f}")
    return best_w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--collab", default="nlgcl", choices=["nlgcl", "lightgcn"])
    ap.add_argument("--ablation", default=None, choices=[None, "no_profile", "no_collab"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    ctx = load_context(args.data)
    from src.queries import load_queries
    q = load_queries(Path(args.data))
    model = TwoStage(ctx, args.collab)
    model.weights = tune(model, q[q.split == "valid"], args.ablation)
    rows = q[q.split == args.split]
    ranking = run_split(model, rows, model.encode(rows["query"].tolist()))
    truth = dict(zip(rows.user_idx, rows.item_idx))
    groups = {"warm": set(rows[~rows.is_cold].user_idx), "cold": set(rows[rows.is_cold].user_idx)}
    res = evaluate(ranking, truth, ctx.n_items, ctx.popularity, groups)
    name = f"twostage_{args.collab}" + (f"_{args.ablation}" if args.ablation else "")
    print(pd.DataFrame(res).T.to_string())
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / f"{name}_{args.split}.json").write_text(json.dumps({"weights": model.weights, "metrics": res}, indent=2))
    (Path(args.out) / f"{name}_weights.json").write_text(json.dumps(model.weights))


if __name__ == "__main__":
    main()
