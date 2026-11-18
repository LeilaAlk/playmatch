"""PlayMatch - final offline evaluation (Week 3: Q2 sanity floor, Q4 innovation proof).

Every model ranks games for the same TEST requests (LLM-generated, see src/queries.py),
with the same rules: the player's hard limits (platform / budget) apply to every model,
and games already reviewed are excluded. Metrics come from src/evaluate.py.

Models (those whose files exist are run):
  random, most_popular         sanity floor
  bm25, bge                    keyword search and dense text retrieval (GenSAR paper baseline)
  twostage                     plan B: BGE + NLGCL + profile
  twostage_no_profile          ablation: no profile term
  twostage_lightgcn            ablation: LightGCN instead of NLGCL
  gensar                       our model (models/gensar_best)
  gensar_noprofile             ablation: same model trained without profile tokens

Innovation metrics
  cold-start NDCG@10: cold users with the request only vs with the onboarding profile
  Platform match@10 : share of the top 10 on the platforms of the player's profile
  Limit respect     : for requests with an explicit limit, share of the top 10 that respects it
                      when GenSAR decodes WITHOUT the hard filter (with it: 100% by construction)

Usage (GPU recommended for gensar)
  python -m src.evaluate_model --data data/processed --split test
  python -m src.evaluate_model --data data/processed --split test --models random most_popular twostage
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.context import load_context
from src.evaluate import evaluate
from src.text_encoder import item_text

ALL = ["random", "most_popular", "bm25", "bge", "twostage", "twostage_no_profile", "twostage_lightgcn",
       "gensar", "gensar_noprofile"]


def limits(r):
    lp = r.limit_platform if isinstance(r.limit_platform, str) else None
    mp = None if pd.isna(r.limit_max_price) else float(r.limit_max_price)
    return lp, mp


# ------------------------------------------------------------- simple baselines
def rank_by_scores(ctx, rows, score_fn, onboarding=True, k=20):
    out = {}
    for r in rows.itertuples():
        hist = ctx.history(int(r.user_idx), onboarding)
        lp, mp = limits(r)
        allowed = ctx.allowed_mask(lp, mp)
        allowed[hist["item_idx"]] = False
        cand = np.where(allowed)[0]
        s = score_fn(r)[cand]
        out[int(r.user_idx)] = [int(i) for i in cand[np.argsort(-s, kind="stable")][:k]]
    return out


class BM25Ranker:
    def __init__(self, ctx):
        from rank_bm25 import BM25Okapi
        docs = [re.findall(r"[a-z0-9]+", item_text(r).lower()) for r in ctx.items.to_dict("records")]
        self.bm25 = BM25Okapi(docs)

    def scores(self, query: str) -> np.ndarray:
        return np.asarray(self.bm25.get_scores(re.findall(r"[a-z0-9]+", query.lower())))


# --------------------------------------------------------------------- models
def run_twostage(ctx, rows, valid_rows, collab, ablation, onboarding_modes):
    from src.twostage import TwoStage, run_split, tune
    model = TwoStage(ctx, collab)
    model.weights = tune(model, valid_rows, ablation)
    qv = model.encode(rows["query"].tolist())
    return {mode: run_split(model, rows, qv, onboarding=mode) for mode in onboarding_modes}, model


def run_gensar(ctx, rows, model_dir, onboarding_modes, batch=16, soft_limits=False):
    from src.predict import GenSARPredictor
    pred = GenSARPredictor(ctx, model_dir)
    results = {}
    for mode in onboarding_modes:
        out = {}
        for key, grp in rows.groupby([rows.limit_platform.fillna(""), rows.limit_max_price.fillna(-1)], sort=False):
            lp = key[0] or None
            mp = None if key[1] == -1 else float(key[1])
            if soft_limits:
                lp, mp = None, None
            grp = list(grp.itertuples())
            for s in range(0, len(grp), batch):
                chunk = grp[s: s + batch]
                hists = [ctx.history(int(r.user_idx), mode) for r in chunk]
                ranked = pred.recommend_batch(hists, [r.query for r in chunk], 20, lp, mp)
                out.update({int(r.user_idx): rk for r, rk in zip(chunk, ranked)})
        results[mode] = out
    return results, pred


# ------------------------------------------------------------ extra metrics
def platform_match(ctx, ranking, rows, k=10):
    vals = []
    for r in rows.itertuples():
        toks = ctx.profile_tokens(ctx.history(int(r.user_idx)))
        plats = {t[6:-1] for t in toks if t.startswith("<plat_")}
        top = ranking.get(int(r.user_idx), [])[:k]
        if plats and top:
            vals.append(np.mean([ctx.platform[i] in plats for i in top]))
    return round(float(np.mean(vals)), 4) if vals else None


def limit_respect(ctx, ranking, rows, k=10):
    vals = []
    for r in rows.itertuples():
        lp, mp = limits(r)
        if lp is None and mp is None:
            continue
        ok = ctx.allowed_mask(lp, mp)
        top = ranking.get(int(r.user_idx), [])[:k]
        if top:
            vals.append(np.mean([ok[i] for i in top]))
    return round(float(np.mean(vals)), 4) if vals else None


def examples_md(ctx, rows, ranking, model_name, n=3, seed=0) -> str:
    """3 random test players with different main platforms (not cherry-picked on hits) + 1 cold player."""
    rng = np.random.default_rng(seed)
    warm = rows[~rows.is_cold].sample(frac=1, random_state=seed)
    picked, plats = [], set()
    for r in warm.itertuples():
        toks = ctx.profile_tokens(ctx.history(int(r.user_idx)))
        main = next((t for t in toks if t.startswith("<plat_")), None)
        if main and main not in plats:
            picked.append(r)
            plats.add(main)
        if len(picked) == n:
            break
    cold = rows[rows.is_cold]
    if len(cold):
        picked.append(next(cold.sample(1, random_state=int(rng.integers(1e6))).itertuples()))
    lines = [f"# Sample users ({model_name})\n"]
    for r in picked:
        hist = ctx.history(int(r.user_idx))
        toks = ctx.profile_tokens(hist)
        lines.append(f"## Player {r.user_idx} ({'cold, onboarding profile' if r.is_cold else 'warm'})")
        lines.append(f"- Profile tokens: {' '.join(toks)}")
        last = list(zip(hist['item_idx'], hist['label']))[-5:]
        lab = {1: "liked", 0: "played", -1: "disliked"}
        lines.append("- Last games: " + "; ".join(f"{ctx.title[i][:40]} [{ctx.platform[i]}, {lab[int(l)]}]" for i, l in last))
        lim = limits(r)
        lines.append(f"- Request: \"{r.query}\"" + (f" (limit: {lim})" if any(x is not None for x in lim) else ""))
        lines.append(f"- Held-out game: {ctx.title[r.item_idx][:50]} [{ctx.platform[r.item_idx]}]")
        top = ranking.get(int(r.user_idx), [])[:10]
        for rank, i in enumerate(top, 1):
            price = "" if np.isnan(ctx.price[i]) else f", ${ctx.price[i]:.0f}"
            hit = "  <-- held-out game" if i == r.item_idx else ""
            lines.append(f"  {rank}. {ctx.title[i][:55]} [{ctx.platform[i]}{price}, popularity {int(ctx.popularity[i])}]{hit}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--models", nargs="+", default=ALL)
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    ctx = load_context(data)
    q = pd.read_parquet(data / "queries.parquet")
    rows, valid_rows = q[q.split == args.split].copy(), q[q.split == "valid"]
    truth = dict(zip(rows.user_idx, rows.item_idx))
    groups = {"warm": set(rows[~rows.is_cold].user_idx), "cold": set(rows[rows.is_cold].user_idx)}
    cold_rows = rows[rows.is_cold]
    print(f"{len(rows):,} {args.split} requests ({len(groups['warm']):,} warm, {len(groups['cold']):,} cold players); "
          f"sources: {rows.source.value_counts().to_dict()}")

    rankings, cold_request_only, latency, extra = {}, {}, {}, {}
    rng = np.random.default_rng(0)
    for m in args.models:
        t0 = time.time()
        if m == "random":
            rankings[m] = rank_by_scores(ctx, rows, lambda r: rng.random(ctx.n_items))
        elif m == "most_popular":
            rankings[m] = rank_by_scores(ctx, rows, lambda r: ctx.popularity)
        elif m == "bm25":
            bm = BM25Ranker(ctx)
            rankings[m] = rank_by_scores(ctx, rows, lambda r: bm.scores(r.query))
        elif m in ("bge", "twostage", "twostage_no_profile", "twostage_lightgcn"):
            collab = "lightgcn" if m == "twostage_lightgcn" else "nlgcl"
            if not (data / f"{collab}_items.npy").exists():
                print(f"skip {m}: {collab}_items.npy missing")
                continue
            ablation = {"bge": "no_collab", "twostage_no_profile": "no_profile"}.get(m)
            res, _ = run_twostage(ctx, rows, valid_rows, collab, ablation, [True, False])
            rankings[m] = res[True]
            cold_request_only[m] = {u: res[False][u] for u in groups["cold"]}
        elif m in ("gensar", "gensar_noprofile"):
            mdir = Path(args.models_dir) / f"{m}_best"
            if not (mdir / "playmatch.json").exists():
                print(f"skip {m}: {mdir} missing")
                continue
            res, _ = run_gensar(ctx, rows, mdir, [True])
            rankings[m] = res[True]
            res_cold, _ = run_gensar(ctx, cold_rows, mdir, [False])
            cold_request_only[m] = res_cold[False]
            if m == "gensar":
                lim_rows = rows[rows.limit_platform.notna() | rows.limit_max_price.notna()]
                soft, _ = run_gensar(ctx, lim_rows, mdir, [True], soft_limits=True)
                extra["gensar_limit_respect_without_filter"] = limit_respect(ctx, soft[True], lim_rows)
                extra["gensar_ndcg_limit_rows_without_filter"] = evaluate(
                    soft[True], dict(zip(lim_rows.user_idx, lim_rows.item_idx)), ctx.n_items, ctx.popularity)["all"]["NDCG@10"]
        else:
            continue
        latency[m] = round(1000 * (time.time() - t0) / max(1, len(rows)), 1)
        print(f"  {m}: done ({latency[m]} ms per request)")

    table, results = [], {}
    for m, rk in rankings.items():
        res = evaluate(rk, truth, ctx.n_items, ctx.popularity, groups)
        if m in cold_request_only:
            res["cold_request_only"] = evaluate(cold_request_only[m], truth, ctx.n_items, ctx.popularity)["all"]
        res["platform_match@10"] = platform_match(ctx, rk, rows)
        res["limit_respect@10"] = limit_respect(ctx, rk, rows)
        res["ms_per_request"] = latency.get(m)
        results[m] = res
        table.append({"model": m, "NDCG@10": res["all"]["NDCG@10"], "Recall@20": res["all"]["Recall@20"],
                      "warm NDCG@10": res.get("warm", {}).get("NDCG@10"),
                      "cold NDCG@10 (request only)": res.get("cold_request_only", {}).get("NDCG@10"),
                      "cold NDCG@10 (+ onboarding)": res.get("cold", {}).get("NDCG@10"),
                      "Coverage@10": res["all"]["Coverage@10"], "AvgPop@10": res["all"]["AvgPop@10"],
                      "Platform match@10": res["platform_match@10"]})
    results["extra"] = extra
    df = pd.DataFrame(table)
    print("\n" + df.to_string(index=False))
    if extra:
        print("\n" + json.dumps(extra, indent=2))
    (out / f"model_comparison_{args.split}.json").write_text(json.dumps(results, indent=2, default=float))
    (out / f"model_comparison_{args.split}.md").write_text(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
    best = "gensar" if "gensar" in rankings else ("twostage" if "twostage" in rankings else None)
    if best:
        (out / "sample_users.md").write_text(examples_md(ctx, rows, rankings[best], best))
        print(f"\nSample users written to {out / 'sample_users.md'}")
    print(f"Saved {out / f'model_comparison_{args.split}.json'}")


if __name__ == "__main__":
    main()
