"""PlayMatch - GenSAR joint item identifiers with RQ-VAE (Week 3, step 3).

Each game gets two identifiers of 3 codes that share their first code, as in GenSAR:
  semantic      <s_x> <a_y> <a_z>
  collaborative <s_x> <b_y> <b_z>
  - shared code s : 1-level RQ-VAE on [BGE text vector ; NLGCL vector]
  - a_y, a_z      : 2-level RQ-VAE on the BGE text vector      (what the game is about)
  - b_y, b_z      : 2-level RQ-VAE on the NLGCL vector          (who plays it)
Two games can not share the same identifier: collisions are resolved by moving the
later game to the nearest free code of the last level.

Inputs : data/processed/items_bge.npy, nlgcl_items.npy (from src.embeddings, src.nlgcl)
Outputs: data/processed/semantic_ids.parquet, results/rqvae_stats.json

Usage
  python -m src.rqvae --data data/processed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CODEBOOK = 256  # must match src/model_skeleton.py


class RQVAE(torch.nn.Module):
    def __init__(self, in_dim: int, levels: int, codebook: int = CODEBOOK, latent: int = 32, hidden: int = 256):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, latent))
        self.dec = torch.nn.Sequential(torch.nn.Linear(latent, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, in_dim))
        self.books = torch.nn.ModuleList([torch.nn.Embedding(codebook, latent) for _ in range(levels)])
        self.initialised = False

    def quantize(self, z: torch.Tensor):
        residual, zq, codes, residuals, qloss = z, torch.zeros_like(z), [], [], 0.0
        for book in self.books:
            residuals.append(residual)
            d = torch.cdist(residual, book.weight)
            idx = d.argmin(1)
            q = book(idx)
            qloss = qloss + F.mse_loss(q, residual.detach()) + 0.25 * F.mse_loss(residual, q.detach())
            zq = zq + q
            residual = residual - q.detach()
            codes.append(idx)
        return zq, torch.stack(codes, 1), residuals, qloss

    @torch.no_grad()
    def init_codebooks(self, z: torch.Tensor, rng: np.random.Generator) -> None:
        residual = z
        for book in self.books:
            pick = rng.choice(len(residual), size=book.num_embeddings, replace=len(residual) < book.num_embeddings)
            book.weight.copy_(residual[pick] + 0.01 * torch.randn_like(residual[pick]))
            idx = torch.cdist(residual, book.weight).argmin(1)
            residual = residual - book(idx)
        self.initialised = True

    @torch.no_grad()
    def reset_dead_codes(self, z: torch.Tensor, rng: np.random.Generator) -> int:
        _, codes, residuals, _ = self.quantize(z)
        dead_total = 0
        for level, book in enumerate(self.books):
            used = torch.bincount(codes[:, level], minlength=book.num_embeddings)
            dead = torch.where(used == 0)[0]
            if len(dead):
                pick = rng.choice(len(z), size=len(dead))
                book.weight[dead] = residuals[level][pick] + 0.01 * torch.randn_like(residuals[level][pick])
                dead_total += len(dead)
        return dead_total

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        zq, codes, residuals, qloss = self.quantize(z)
        x_hat = self.dec(z + (zq - z).detach())  # straight-through estimator
        return x_hat, codes, residuals, F.mse_loss(x_hat, x) + qloss


def train_rqvae(x: np.ndarray, levels: int, epochs: int, seed: int, label: str):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt = torch.tensor(x, dtype=torch.float32)
    model = RQVAE(x.shape[1], levels)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        model.init_codebooks(model.enc(xt), rng)
    for epoch in range(1, epochs + 1):
        _, _, _, loss = model(xt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 and epoch < epochs * 0.8:
            model.reset_dead_codes(model.enc(xt), rng)
        if epoch % 100 == 0 or epoch == epochs:
            print(f"  [{label}] epoch {epoch}: loss {loss.item():.5f}")
    model.eval()
    with torch.no_grad():
        _, codes, residuals, loss = model(xt)
        last = residuals[-1]
        dist_last = torch.cdist(last, model.books[-1].weight).numpy()
    return codes.numpy(), dist_last, float(loss)


def resolve_collisions(prefix: np.ndarray, last: np.ndarray, dist_last: np.ndarray) -> tuple[np.ndarray, int]:
    """Make (prefix..., last) unique: a colliding game takes the nearest free code of the last level."""
    used: dict[tuple, set] = {}
    out, moved = last.copy(), 0
    for i in range(len(last)):
        key = tuple(prefix[i])
        taken = used.setdefault(key, set())
        if out[i] in taken:
            for c in np.argsort(dist_last[i]):
                if c not in taken:
                    out[i], moved = c, moved + 1
                    break
            else:
                raise SystemExit("More than 256 games share one prefix: increase CODEBOOK")
        taken.add(int(out[i]))
    return out, moved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--collab", default="nlgcl", choices=["nlgcl", "lightgcn"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    text = np.load(data / "items_bge.npy")
    collab = np.load(data / f"{args.collab}_items.npy")
    collab = collab / np.clip(np.linalg.norm(collab, axis=1, keepdims=True), 1e-9, None)
    n = len(text)

    print("Shared code (text + collaborative)...")
    shared, _, loss_s = train_rqvae(np.hstack([text, collab]), 1, args.epochs, args.seed, "shared")
    print("Semantic codes (text)...")
    sem, dist_a, loss_a = train_rqvae(text, 2, args.epochs, args.seed + 1, "semantic")
    print("Collaborative codes (NLGCL)...")
    col, dist_b, loss_b = train_rqvae(collab, 2, args.epochs, args.seed + 2, "collaborative")

    s = shared[:, 0]
    sem_prefix = np.column_stack([s, sem[:, 0]])
    col_prefix = np.column_stack([s, col[:, 0]])
    a2, moved_a = resolve_collisions(sem_prefix, sem[:, 1], dist_a)
    b2, moved_b = resolve_collisions(col_prefix, col[:, 1], dist_b)
    ids = pd.DataFrame({"item_idx": np.arange(n), "s": s, "a1": sem[:, 0], "a2": a2, "b1": col[:, 0], "b2": b2})
    assert not ids.duplicated(["s", "a1", "a2"]).any() and not ids.duplicated(["s", "b1", "b2"]).any()
    ids.to_parquet(data / "semantic_ids.parquet", index=False)

    # Sanity check: do similar games (BGE nearest neighbour) share their first semantic codes?
    sims = text @ text.T
    np.fill_diagonal(sims, -1)
    nn = sims.argmax(1)
    same_prefix = float(np.mean((s == s[nn]) & (sem[:, 0] == sem[nn, 0])))
    rand = np.random.default_rng(0).permutation(n)
    rand_prefix = float(np.mean((s == s[rand]) & (sem[:, 0] == sem[rand, 0])))
    stats = {
        "games": n, "collab_source": args.collab,
        "codes_used": {"s": int(len(np.unique(s))), "a1": int(len(np.unique(sem[:, 0]))),
                       "b1": int(len(np.unique(col[:, 0])))},
        "collisions_resolved": {"semantic": moved_a, "collaborative": moved_b},
        "collision_rate": {"semantic": round(moved_a / n, 4), "collaborative": round(moved_b / n, 4)},
        "final_losses": {"shared": loss_s, "semantic": loss_a, "collaborative": loss_b},
        "nearest_text_neighbour_shares_first_two_codes": round(same_prefix, 4),
        "random_pair_shares_first_two_codes": round(rand_prefix, 4),
    }
    (out / "rqvae_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"Saved {data / 'semantic_ids.parquet'}")


if __name__ == "__main__":
    main()
