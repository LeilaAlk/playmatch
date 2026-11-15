"""PlayMatch - player profile builder (first version).

Built from TRAINING interactions only (no leakage). Done in this version:
  - likely platforms (time-weighted share of liked games per platform)
  - usual price range (median price of liked games, 4 buckets)
  - counts of liked / played / disliked games
  - recency: weight halves every HALF_LIFE_DAYS before the user's last training interaction
  - gifts: reviews flagged as gifts do not shape the profile
Not yet: genres (BGE zero-shot), franchises, LLM gift flags.

Usage
  python -m src.profile --data data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HALF_LIFE_DAYS = 365
MS_PER_DAY = 86_400_000
PRICE_BUCKETS = [(0, 20, "under_20"), (20, 40, "20_40"), (40, 60, "40_60"), (60, float("inf"), "over_60")]


def price_bucket(price: float) -> str:
    if price is None or np.isnan(price):
        return "unknown"
    for low, high, name in PRICE_BUCKETS:
        if low <= price < high:
            return name
    return "unknown"


def build_profiles(train: pd.DataFrame, items: pd.DataFrame, half_life_days: float = HALF_LIFE_DAYS) -> pd.DataFrame:
    df = train.merge(items[["item_idx", "platform", "price"]], on="item_idx", how="left")
    counts = df.groupby("user_idx").label.agg(
        n_liked=lambda s: int((s == 1).sum()),
        n_played=lambda s: int((s == 0).sum()),
        n_disliked=lambda s: int((s == -1).sum()),
    )
    df = df[~df.gift_flag]  # gifts stay in the graph, not in the profile
    last = df.groupby("user_idx").timestamp.transform("max")
    df = df.assign(weight=0.5 ** ((last - df.timestamp) / MS_PER_DAY / half_life_days))
    liked = df[df.label == 1]

    plat = liked.groupby(["user_idx", "platform"]).weight.sum().unstack(fill_value=0.0)
    shares = plat.div(plat.sum(axis=1), axis=0).round(3)
    platforms = shares.apply(lambda r: {k: float(v) for k, v in r.items() if v > 0}, axis=1).rename("platform_shares")
    main_platform = shares.idxmax(axis=1).rename("main_platform")

    priced = liked.dropna(subset=["price"])
    known = priced.groupby("user_idx").price.agg(["median", "count"])
    bucket = known.apply(lambda r: price_bucket(r["median"]) if r["count"] >= 2 else "unknown", axis=1).rename("price_bucket")

    prof = counts.join([platforms, main_platform, bucket], how="left").reset_index()
    prof["price_bucket"] = prof.price_bucket.fillna("unknown")
    prof["main_platform"] = prof.main_platform.fillna("unknown")
    return prof


def profile_tokens(row) -> list[str]:
    """Profile as tokens for the generative model: platforms with >= 30% share + price bucket."""
    shares = row.get("platform_shares")
    shares = shares if isinstance(shares, dict) else {}  # NaN when the user has no liked, non-gift game
    tokens = [f"<plat_{p}>" for p, s in sorted(shares.items(), key=lambda x: -x[1]) if s >= 0.3]
    return tokens + [f"<price_{row.get('price_bucket', 'unknown')}>"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args()
    data = Path(args.data)
    train = pd.read_parquet(data / "train.parquet")
    items = pd.read_parquet(data / "items.parquet")
    prof = build_profiles(train, items)
    prof.to_parquet(data / "profiles.parquet", index=False)
    print(f"Profiles built for {len(prof):,} users")
    print(f"  main platform known: {(prof.main_platform != 'unknown').mean():.1%}")
    print(f"  price bucket known:  {(prof.price_bucket != 'unknown').mean():.1%}")
    print("\nThree example profiles:")
    for _, r in prof.sample(3, random_state=1).iterrows():
        print(f"  user {r.user_idx}: liked={r.n_liked} played={r.n_played} disliked={r.n_disliked} "
              f"platforms={r.platform_shares} price={r.price_bucket} -> tokens {profile_tokens(r)}")


if __name__ == "__main__":
    main()
