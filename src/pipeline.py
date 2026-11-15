"""PlayMatch - V1 data pipeline (Week 2, Q1).

Reads the raw Amazon Reviews 2023 Video_Games files (review + meta, .jsonl.gz),
and writes clean Parquet files ready for the models and the evaluation harness.

Steps
  1. Items: keep games only (drop accessories and hardware), extract platform,
     price and release year from the metadata.
  2. Reviews: keep the columns we need, label each rating
     (4-5 = liked, 3 = played, 1-2 = disliked), flag likely gifts with a
     simple keyword rule (the LLM version comes later).
  3. k-core filter on liked interactions (users and items with >= k likes).
  4. Temporal leave-last-out split: per user, last liked game = test,
     the one before = validation, everything strictly before = train.
  5. Cold users: a random 10% of users are removed from training entirely.

Usage
  python -m src.pipeline --raw data/raw --out data/processed --k 5
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Accepted names: compressed (.gz, from datarepo.eng.ucsd.edu) or not (.jsonl, from Hugging Face).
REVIEW_FILES = ["Video_Games.jsonl.gz", "Video_Games.jsonl"]
META_FILES = ["meta_Video_Games.jsonl.gz", "meta_Video_Games.jsonl"]

# Words that mark non-game products in the category path or the title.
NON_GAME_WORDS = re.compile(
    r"\b(accessor\w*|controllers?|headsets?|cables?|chargers?|adapters?|batter(y|ies)|"
    r"memory cards?|skins?|faceplates?|mounts?|stands?|cases?|bags?|cooling|keyboards?|mouse|mice|"
    r"gamepads?|joysticks?|steering wheels?|consoles?|hardware|stylus|screen protectors?)\b",
    re.IGNORECASE,
)
PLATFORM_PATTERNS = [
    ("PlayStation", re.compile(r"playstation|\bps ?[1-5]\b|\bps vita\b|\bpsp\b", re.I)),
    ("Xbox", re.compile(r"xbox", re.I)),
    ("Nintendo", re.compile(r"nintendo|switch|\bwii\b|wii u|3ds|\bds\b|game ?boy|gamecube|\bn64\b", re.I)),
    ("PC", re.compile(r"\bpc\b|windows|\bmac\b|macintosh|linux|steam", re.I)),
]
GIFT_PATTERN = re.compile(
    r"\b(gift|present|christmas|birthday|for my (son|daughter|kid|kids|grandson|granddaughter|"
    r"nephew|niece|husband|wife|boyfriend|girlfriend|brother|sister|dad|mom|father|mother))\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(19[7-9]\d|20[0-2]\d)\b")


def counts(series: pd.Series) -> dict:
    """value_counts() as a JSON-safe dict."""
    return {str(k): int(v) for k, v in series.value_counts().items()}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def find_file(raw: Path, names: list[str]) -> Path:
    for name in names:
        path = raw / name
        if path.exists():
            with open(path, "rb") as f:
                head = f.read(64)
            if head.lstrip().startswith(b"<"):
                raise SystemExit(f"{path} is a web page ({path.stat().st_size} bytes), not data: "
                                 "the download failed. Delete it and download it again.")
            return path
    raise SystemExit(f"None of {names} found in {raw.resolve()}")


def read_jsonl_gz(path: Path):
    with open(path, "rb") as f:
        is_gzip = f.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# --------------------------------------------------------------------- items
def as_dict(details) -> dict:
    if isinstance(details, dict):
        return details
    if isinstance(details, str) and details.strip().startswith("{"):
        try:
            return json.loads(details)
        except json.JSONDecodeError:
            return {}
    return {}


def parse_price(value) -> float | None:
    if value is None:
        return None
    try:
        price = float(str(value).replace("$", "").replace(",", ""))
    except ValueError:
        return None
    return price if price > 0 else None


def detect_platform(categories: list[str], title: str, details: dict) -> str:
    text = " ".join(categories) + " " + str(details.get("Platform", "")) + " " + title
    for name, pattern in PLATFORM_PATTERNS:
        if pattern.search(text):
            return name
    return "other"


def is_game(categories: list[str], title: str) -> tuple[bool, str]:
    """Return (keep, reason). Reason is used for the cleaning statistics."""
    if categories:
        path = " > ".join(categories)
        if NON_GAME_WORDS.search(path):
            return False, "non_game_category"
        if any("game" in c.lower() for c in categories[1:]):
            return True, "game_category"
        return False, "no_game_category"
    # No category path: fall back on the title.
    if NON_GAME_WORDS.search(title):
        return False, "non_game_title"
    return True, "kept_without_category"


def load_items(meta_path: Path) -> tuple[pd.DataFrame, dict]:
    rows, reasons = [], {}
    for rec in read_jsonl_gz(meta_path):
        categories = rec.get("categories") or []
        title = rec.get("title") or ""
        keep, reason = is_game(categories, title)
        reasons[reason] = reasons.get(reason, 0) + 1
        if not keep:
            continue
        details = as_dict(rec.get("details"))
        release = str(details.get("Release date") or details.get("Date First Available") or "")
        year = YEAR_PATTERN.search(release)
        rows.append(
            {
                "parent_asin": rec["parent_asin"],
                "title": title,
                "store": rec.get("store") or "",
                "platform": detect_platform(categories, title, details),
                "price": parse_price(rec.get("price")),
                "release_year": int(year.group(0)) if year else None,
                "average_rating": rec.get("average_rating"),
                "rating_number": rec.get("rating_number") or 0,
                "categories": " > ".join(categories),
                "description": " ".join(rec.get("description") or [])[:2000],
            }
        )
    items = pd.DataFrame(rows).drop_duplicates("parent_asin")
    return items, reasons


# ------------------------------------------------------------------- reviews
def load_reviews(review_path: Path) -> pd.DataFrame:
    cols = {"user_id": [], "parent_asin": [], "rating": [], "timestamp": [],
            "verified_purchase": [], "helpful_vote": [], "gift_flag": []}
    for n, rec in enumerate(read_jsonl_gz(review_path), 1):
        cols["user_id"].append(rec["user_id"])
        cols["parent_asin"].append(rec["parent_asin"])
        cols["rating"].append(float(rec["rating"]))
        cols["timestamp"].append(int(rec["timestamp"]))
        cols["verified_purchase"].append(bool(rec.get("verified_purchase", False)))
        cols["helpful_vote"].append(int(rec.get("helpful_vote") or 0))
        text = f"{rec.get('title', '')} {rec.get('text', '')}"
        cols["gift_flag"].append(bool(GIFT_PATTERN.search(text)))
        if n % 500_000 == 0:
            log(f"  read {n:,} reviews")
    df = pd.DataFrame(cols)
    df["label"] = np.select([df.rating >= 4, df.rating <= 2], [1, -1], default=0).astype("int8")
    return df


# -------------------------------------------------------------------- k-core
def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Keep users and items with at least k liked interactions (iterated)."""
    pos = df[df.label == 1]
    while True:
        u = pos.user_id.value_counts()
        i = pos.parent_asin.value_counts()
        keep = pos.user_id.isin(u[u >= k].index) & pos.parent_asin.isin(i[i >= k].index)
        if keep.all():
            break
        pos = pos[keep]
    users, items = set(pos.user_id), set(pos.parent_asin)
    return df[df.user_id.isin(users) & df.parent_asin.isin(items)]


# --------------------------------------------------------------------- split
def temporal_split(df: pd.DataFrame, cold_share: float, seed: int):
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    pos = df[df.label == 1]
    last_two = pos.groupby("user_id").tail(2)
    test = last_two.groupby("user_id").tail(1)
    valid = last_two.drop(test.index)
    # Anything at or after the validation time is future information: drop it from train.
    cut = valid.set_index("user_id").timestamp
    train = df.drop(index=test.index.union(valid.index))
    train = train[train.timestamp < train.user_id.map(cut)]

    rng = np.random.default_rng(seed)
    users = df.user_id.unique()
    cold = set(rng.choice(users, size=int(len(users) * cold_share), replace=False))
    cold_history = train[train.user_id.isin(cold)]  # hidden from every model, used for onboarding
    train = train[~train.user_id.isin(cold)]
    return train, valid, test, cold_history, cold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cold-share", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stats: dict = {"k": args.k, "cold_share": args.cold_share, "seed": args.seed}

    log("Loading item metadata...")
    items, reasons = load_items(find_file(raw, META_FILES))
    stats["meta_filter_reasons"] = reasons
    stats["items_games"] = len(items)
    stats["platform_counts"] = counts(items.platform)
    stats["share_items_with_price"] = round(items.price.notna().mean(), 4)
    stats["share_items_with_release_year"] = round(items.release_year.notna().mean(), 4)
    log(f"  {len(items):,} games kept; filter reasons: {reasons}")

    log("Loading reviews (this is the long step)...")
    reviews = load_reviews(find_file(raw, REVIEW_FILES))
    stats["raw"] = {"reviews": len(reviews), "users": reviews.user_id.nunique(),
                    "items": reviews.parent_asin.nunique()}
    reviews = reviews[reviews.parent_asin.isin(items.parent_asin)]
    reviews = reviews.sort_values("timestamp").drop_duplicates(["user_id", "parent_asin"], keep="last")
    stats["games_only"] = {"reviews": len(reviews), "users": reviews.user_id.nunique(),
                           "items": reviews.parent_asin.nunique(),
                           "label_counts": counts(reviews.label),
                           "share_gift_flag": round(reviews.gift_flag.mean(), 4)}
    log(f"  games only: {stats['games_only']}")

    log(f"k-core filter (k={args.k}) on liked interactions...")
    core = k_core(reviews, args.k)
    stats["k_core"] = {"interactions": len(core), "users": core.user_id.nunique(),
                       "items": core.parent_asin.nunique(),
                       "label_counts": counts(core.label),
                       "density": round(len(core) / (core.user_id.nunique() * core.parent_asin.nunique()), 6)}
    log(f"  after k-core: {stats['k_core']}")

    # Integer ids for the models.
    user_ids = {u: i for i, u in enumerate(sorted(core.user_id.unique()))}
    item_ids = {a: i for i, a in enumerate(sorted(core.parent_asin.unique()))}
    core = core.assign(user_idx=core.user_id.map(user_ids), item_idx=core.parent_asin.map(item_ids))
    items = items[items.parent_asin.isin(item_ids)].assign(item_idx=lambda d: d.parent_asin.map(item_ids))

    log("Temporal leave-last-out split + cold users...")
    train, valid, test, cold_hist, cold = temporal_split(core, args.cold_share, args.seed)
    users = pd.DataFrame({"user_id": list(user_ids), "user_idx": list(user_ids.values())})
    users["is_cold"] = users.user_id.isin(cold)

    keep_cols = ["user_idx", "item_idx", "rating", "label", "timestamp",
                 "verified_purchase", "helpful_vote", "gift_flag"]
    for name, frame in [("train", train), ("valid", valid), ("test", test), ("cold_history", cold_hist)]:
        frame[keep_cols].to_parquet(out / f"{name}.parquet", index=False)
    items.sort_values("item_idx").to_parquet(out / "items.parquet", index=False)
    users.to_parquet(out / "users.parquet", index=False)

    # Leakage checks written into the stats file (evidence for the report).
    tr_last = train.groupby("user_idx").timestamp.max()
    te_time = test.set_index("user_idx").timestamp
    common = tr_last.index.intersection(te_time.index)
    stats["split"] = {
        "train_interactions": len(train), "valid": len(valid), "test": len(test),
        "cold_users": len(cold), "warm_users": users.shape[0] - len(cold),
        "cold_users_in_train": int(train.user_idx.isin(users[users.is_cold].user_idx).sum()),
        "test_pairs_in_train": int(len(train.merge(test[["user_idx", "item_idx"]], on=["user_idx", "item_idx"]))),
        "train_after_test_time": int((tr_last.loc[common] >= te_time.loc[common]).sum()),
    }
    stats["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=int))
    log(f"Done in {stats['runtime_seconds']} s. Split: {stats['split']}")
    log(f"Files written to {out.resolve()}")


if __name__ == "__main__":
    main()
