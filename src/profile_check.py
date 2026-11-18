"""PlayMatch - how reliable is the inferred player profile?

Amazon has no "this player owns a PS5" field. We infer the profile from the games
the player liked during the TRAINING period, using the platform and price of each
game in the metadata, with weights that favor recent games (profile.py).

This script checks the inference on data the profile never saw: for each warm
user, is the held-out TEST game on one of the platforms of his profile, and in
his price range? We compare with a naive guess that ignores the player
(the most common platform / price bucket of the catalog he plays in).

Usage
  python -m src.profile_check --data data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.profile import build_profiles, price_bucket


def history_bucket(n: int) -> str:
    return "1-4 liked" if n <= 4 else ("5-10 liked" if n <= 10 else "11+ liked")


def check(prof: pd.DataFrame, test: pd.DataFrame, items: pd.DataFrame, train: pd.DataFrame) -> dict:
    it = items.set_index("item_idx")
    df = test.merge(prof, on="user_idx", how="inner")
    df["test_platform"] = df.item_idx.map(it.platform)
    df["test_price"] = df.item_idx.map(it.price).map(price_bucket)

    # Naive guess = most common platform / price bucket among liked training games (same for everyone).
    liked = train[train.label == 1].item_idx
    naive_platform = it.platform.reindex(liked).value_counts().idxmax()
    naive_price = it.price.reindex(liked).map(price_bucket).loc[lambda s: s != "unknown"].value_counts().idxmax()

    plat = df[(df.main_platform != "unknown") & (df.test_platform != "other")].copy()
    plat["main_ok"] = plat.main_platform == plat.test_platform
    plat["in_profile"] = [isinstance(s, dict) and s.get(p, 0) >= 0.3
                          for s, p in zip(plat.platform_shares, plat.test_platform)]
    plat["naive_ok"] = plat.test_platform == naive_platform
    price = df[(df.price_bucket != "unknown") & (df.test_price != "unknown")].copy()
    price["ok"] = price.price_bucket == price.test_price
    price["naive_ok"] = price.test_price == naive_price

    by_hist = (plat.assign(h=plat.n_liked.map(history_bucket))
               .groupby("h")[["main_ok", "in_profile"]].mean().round(3).to_dict(orient="index"))
    return {
        "users_checked_platform": len(plat),
        "platform_main_matches_test_game": round(plat.main_ok.mean(), 3),
        "test_game_platform_in_profile_tokens": round(plat.in_profile.mean(), 3),
        "naive_guess": naive_platform,
        "naive_platform_accuracy": round(plat.naive_ok.mean(), 3),
        "by_history_length": by_hist,
        "users_checked_price": len(price),
        "price_bucket_matches_test_game": round(price.ok.mean(), 3),
        "naive_price_guess": naive_price,
        "naive_price_accuracy": round(price.naive_ok.mean(), 3),
        "median_liked_games_per_profile": float(np.median(df.n_liked)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    data = Path(args.data)
    train = pd.read_parquet(data / "train.parquet")
    test = pd.read_parquet(data / "test.parquet")
    items = pd.read_parquet(data / "items.parquet")

    results = {
        "with_recency_weights": check(build_profiles(train, items), test, items, train),
        "without_recency_weights": check(build_profiles(train, items, half_life_days=1e9), test, items, train),
    }
    r = results["with_recency_weights"]
    print(f"Users checked: {r['users_checked_platform']:,} (median {r['median_liked_games_per_profile']:.0f} liked games per profile)")
    print(f"Platform: main profile platform = platform of the held-out test game for {r['platform_main_matches_test_game']:.1%} "
          f"of users (naive guess '{r['naive_guess']}' for everyone: {r['naive_platform_accuracy']:.1%})")
    print(f"          test game on a platform of the profile tokens: {r['test_game_platform_in_profile_tokens']:.1%}")
    print(f"          without recency weights: {results['without_recency_weights']['platform_main_matches_test_game']:.1%}")
    print(f"          by history length: {r['by_history_length']}")
    print(f"Price:    profile bucket = test game bucket for {r['price_bucket_matches_test_game']:.1%} of {r['users_checked_price']:,} users "
          f"(naive guess '{r['naive_price_guess']}': {r['naive_price_accuracy']:.1%})")
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "profile_check.json").write_text(json.dumps(results, indent=2))
    print(f"Saved to {Path(args.out) / 'profile_check.json'}")


if __name__ == "__main__":
    main()
