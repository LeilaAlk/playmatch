"""Add the display metadata of each game to data/processed/items.parquet (for the interface).

Adds: image_url (cover), features (bullet points), release_date (true "Release date" field),
date_first_available (date the product appeared on Amazon), manufacturer / brand, platform_raw
(the "Platform" field of the details), and prints how many games have each field.
Splits, ids and models are not changed.

Usage:  python scripts/add_metadata.py --raw data/raw --data data/processed
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import META_FILES, as_dict, find_file, read_jsonl_gz  # noqa: E402


def first_image(images) -> str | None:
    """Raw files: list of {thumb, large, hi_res, variant}; Hugging Face 'datasets': dict of lists."""
    if isinstance(images, list):
        for im in images:
            if isinstance(im, dict):
                url = im.get("large") or im.get("hi_res") or im.get("thumb")
                if url:
                    return url
    if isinstance(images, dict):
        for key in ("large", "hi_res", "thumb"):
            urls = [u for u in (images.get(key) or []) if u]
            if urls:
                return urls[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--data", default="data/processed")
    args = ap.parse_args()
    path = Path(args.data) / "items.parquet"
    items = pd.read_parquet(path)
    wanted = set(items.parent_asin)
    rows = {}
    for rec in read_jsonl_gz(find_file(Path(args.raw), META_FILES)):
        if rec["parent_asin"] not in wanted:
            continue
        d = as_dict(rec.get("details"))
        rows[rec["parent_asin"]] = {
            "image_url": first_image(rec.get("images")),
            "features": " | ".join(rec.get("features") or [])[:1000],
            "release_date": d.get("Release date"),
            "date_first_available": d.get("Date First Available"),
            "manufacturer": d.get("Manufacturer") or d.get("Brand"),
            "platform_raw": d.get("Platform"),
            "details_json": json.dumps(d)[:2000],
        }
    extra = pd.DataFrame.from_dict(rows, orient="index").rename_axis("parent_asin").reset_index()
    items = items.drop(columns=[c for c in extra.columns if c != "parent_asin" and c in items.columns])
    items = items.merge(extra, on="parent_asin", how="left")
    items.to_parquet(path, index=False)
    print(f"{len(items):,} games updated in {path}")
    for col in ["image_url", "features", "release_date", "date_first_available", "manufacturer", "platform_raw"]:
        share = items[col].notna() & (items[col].astype(str).str.len() > 0)
        print(f"  {col:22s} {share.mean():6.1%}")
    keys = pd.Series([k for s in items.details_json.dropna() for k in json.loads(s)]).value_counts().head(15)
    print("Most common keys of 'details':", keys.to_dict())


if __name__ == "__main__":
    main()
