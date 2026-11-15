"""Write a small fake dataset with the same schema as Amazon Reviews 2023 Video_Games.

Only for testing the code without the real files:
  python tests/make_synthetic_data.py --out data/raw_synthetic
  python -m src.pipeline --raw data/raw_synthetic --out data/processed_synthetic --k 3
"""
import argparse
import gzip
import json
import random
from pathlib import Path

PLATFORMS = ["PlayStation 4", "Xbox One", "Nintendo Switch", "PC"]
WORDS = ["Quest", "Legends", "Farm", "Racer", "Shadow", "Puzzle", "Kingdom", "Galaxy", "Island", "Hero"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_synthetic")
    ap.add_argument("--items", type=int, default=400)
    ap.add_argument("--users", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    items = []
    with gzip.open(out / "meta_Video_Games.jsonl.gz", "wt") as f:
        for i in range(args.items):
            plat = rng.choice(PLATFORMS)
            accessory = rng.random() < 0.15
            leaf = ["Accessories", "Controllers"] if accessory else ["Games"]
            title = f"{rng.choice(WORDS)} {rng.choice(WORDS)} {i}" + (" Controller" if accessory else "")
            rec = {
                "main_category": "Video Games", "title": title,
                "average_rating": round(rng.uniform(2.5, 5), 1), "rating_number": rng.randint(1, 5000),
                "features": [], "description": [f"A {plat} game about {rng.choice(WORDS).lower()}."],
                "price": rng.choice([None, round(rng.uniform(5, 70), 2)]),
                "images": [], "videos": [], "store": f"Studio {i % 30}",
                "categories": ["Video Games", plat] + leaf,
                "details": {"Release date": f"March {1 + i % 28}, {2008 + i % 15}"},
                "parent_asin": f"B{i:09d}", "bought_together": None,
            }
            f.write(json.dumps(rec) + "\n")
            if not accessory:
                items.append((rec["parent_asin"], plat))

    pop = [1 / (r + 1) ** 0.8 for r in range(len(items))]  # long-tail popularity
    with gzip.open(out / "Video_Games.jsonl.gz", "wt") as f:
        for u in range(args.users):
            fav = rng.choice(PLATFORMS)
            n = rng.randint(2, 15)
            ts = 1_400_000_000_000 + rng.randint(0, 10**11)
            pool = [it for it in items if it[1] == fav or rng.random() < 0.2]
            weights = [pop[items.index(it)] for it in pool]
            for asin, _ in rng.choices(pool, weights=weights, k=n):
                ts += rng.randint(10**8, 10**10)
                text = rng.choice(["Great game.", "Bought it as a gift for my son.", "Boring.", "Loved it!"])
                rec = {"rating": float(rng.choices([1, 2, 3, 4, 5], [5, 5, 10, 30, 50])[0]),
                       "title": "Review", "text": text, "images": [], "asin": asin, "parent_asin": asin,
                       "user_id": f"U{u:06d}", "timestamp": ts, "helpful_vote": rng.randint(0, 5),
                       "verified_purchase": rng.random() < 0.8}
                f.write(json.dumps(rec) + "\n")
    print(f"Synthetic data written to {out}")


if __name__ == "__main__":
    main()
