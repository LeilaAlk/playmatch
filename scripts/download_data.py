"""Download the Video_Games review and meta files from the official Hugging Face copy
of Amazon Reviews 2023, and save them in data/raw/ with the names the pipeline expects.

Usage:  python scripts/download_data.py
"""
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO = "McAuley-Lab/Amazon-Reviews-2023"
RAW = Path("data/raw")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith("raw/") and "Video_Games" in f and f.endswith((".jsonl", ".jsonl.gz"))]
    print("Files found on Hugging Face:", files)
    review = [f for f in files if f.startswith("raw/review_categories/")]
    meta = [f for f in files if f.startswith("raw/meta_categories/")]
    if not review or not meta:
        sys.exit("Could not find both files; send this output to the team.")

    for remote, local_name in [(meta[0], "meta_Video_Games"), (review[0], "Video_Games")]:
        suffix = ".jsonl.gz" if remote.endswith(".gz") else ".jsonl"
        target = RAW / f"{local_name}{suffix}"
        if target.exists() and target.stat().st_size > 10_000_000:
            print(f"Already there: {target} ({target.stat().st_size / 1e9:.2f} GB)")
            continue
        print(f"Downloading {remote} ... (several GB for the reviews, be patient)")
        path = hf_hub_download(REPO, remote, repo_type="dataset", local_dir="data/hf")
        os.replace(path, target)  # move, no copy (saves disk space)
        print(f"Saved {target} ({target.stat().st_size / 1e9:.2f} GB)")
    print("Done. Next: python -m src.pipeline --raw data/raw --out data/processed --k 5")


if __name__ == "__main__":
    main()
