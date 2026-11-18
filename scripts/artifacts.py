"""Share the trained model and the processed data through the Hugging Face Hub,
so that a clean machine can run the system without redoing the whole pipeline (Q6).

Upload (once, by the person who trained; needs a free Hugging Face account):
  huggingface-cli login
  python scripts/artifacts.py upload --repo <hf-user>/playmatch-artifacts

Download (clean machine, no account needed if the repo is public):
  python scripts/artifacts.py download --repo <hf-user>/playmatch-artifacts
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

PROCESSED = ["items.parquet", "users.parquet", "train.parquet", "valid.parquet", "test.parquet",
             "cold_history.parquet", "stats.json", "items_bge.npy", "text_encoder.json",
             "nlgcl_users.npy", "nlgcl_items.npy", "semantic_ids.parquet", "queries.parquet"]


def upload(repo: str) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path="data/processed", path_in_repo="processed",
                      allow_patterns=PROCESSED)
    for name in ["gensar_best", "gensar_noprofile_best"]:
        if (Path("models") / name).exists():
            api.upload_folder(repo_id=repo, folder_path=f"models/{name}", path_in_repo=f"models/{name}")
    for f in ["results/twostage_nlgcl_weights.json"]:
        if Path(f).exists():
            api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=repo)
    print(f"Uploaded to https://huggingface.co/{repo}")


def download(repo: str) -> None:
    local = Path(snapshot_download(repo_id=repo, local_dir="hf_artifacts"))
    Path("data").mkdir(exist_ok=True)
    for src, dst in [(local / "processed", Path("data/processed")), (local / "models", Path("models")),
                     (local / "results", Path("results"))]:
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.rglob("*"):
                if f.is_file():
                    target = dst / f.relative_to(src)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(f.read_bytes())
    print("Artifacts ready in data/processed, models/ and results/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["upload", "download"])
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    upload(a.repo) if a.step == "upload" else download(a.repo)
