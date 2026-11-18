"""PlayMatch - collaborative embeddings with NLGCL (Week 3, step 2).

LightGCN graph model trained with BPR, plus the NLGCL contrastive loss
(Xu et al., RecSys 2025): for a user-game edge (u, i), the vector of u at layer l
is pulled toward the vector of its neighbour i at layer l+1 (and i at layer l
toward u at layer l+1); the other nodes of the batch are negatives. The views are
the layers that already exist, so no graph augmentation is needed.
This is our own implementation of that idea, in plain PyTorch.

The graph is built from TRAINING liked interactions only (no validation, no test).
Each epoch logs the training loss, the validation loss (BPR on validation pairs)
and validation NDCG@10 / Recall@20; the best epoch on validation NDCG is kept.

Outputs
  data/processed/nlgcl_users.npy, nlgcl_items.npy   (or lightgcn_*.npy with --no-contrastive)
  results/nlgcl_curve.png, results/nlgcl_history.json

Usage
  python -m src.nlgcl --data data/processed --epochs 50
  python -m src.nlgcl --data data/processed --epochs 50 --no-contrastive   # LightGCN ablation
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def build_adjacency(users: np.ndarray, items: np.ndarray, n_users: int, n_items: int) -> torch.Tensor:
    """Symmetric normalised adjacency D^-1/2 A D^-1/2 of the user-item bipartite graph."""
    rows = np.concatenate([users, items + n_users])
    cols = np.concatenate([items + n_users, users])
    n = n_users + n_items
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    inv = np.zeros_like(deg)
    inv[deg > 0] = deg[deg > 0] ** -0.5
    vals = inv[rows] * inv[cols]
    idx = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(vals, dtype=torch.float32), (n, n)).coalesce()


class LightGCN(torch.nn.Module):
    def __init__(self, n_users: int, n_items: int, dim: int, layers: int):
        super().__init__()
        self.n_users, self.n_items, self.layers = n_users, n_items, layers
        self.emb = torch.nn.Embedding(n_users + n_items, dim)
        torch.nn.init.normal_(self.emb.weight, std=0.1)

    def propagate(self, adj: torch.Tensor) -> list[torch.Tensor]:
        out = [self.emb.weight]
        for _ in range(self.layers):
            out.append(torch.sparse.mm(adj, out[-1]))
        return out

    def final(self, layer_embs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.stack(layer_embs).mean(0)
        return e[: self.n_users], e[self.n_users:]


def nlgcl_loss(layer_embs, u, i, n_users: int, tau: float) -> torch.Tensor:
    """Neighbour-layer InfoNCE: u at layer l vs its neighbour i at layer l+1, and the reverse."""
    loss = 0.0
    labels = torch.arange(len(u), device=u.device)
    # Another edge of the batch that shares the user or the game is also a true neighbour pair:
    # it must not be used as a negative (masked out of the softmax).
    eye = torch.eye(len(u), dtype=torch.bool, device=u.device)
    false_neg = ((u[:, None] == u[None, :]) | (i[:, None] == i[None, :])) & ~eye
    for l in range(len(layer_embs) - 1):
        cur, nxt = layer_embs[l], layer_embs[l + 1]
        zu, zi_next = F.normalize(cur[u], dim=1), F.normalize(nxt[i + n_users], dim=1)
        zi, zu_next = F.normalize(cur[i + n_users], dim=1), F.normalize(nxt[u], dim=1)
        logits_u = (zu @ zi_next.T / tau).masked_fill(false_neg, -1e9)
        logits_i = (zi @ zu_next.T / tau).masked_fill(false_neg, -1e9)
        loss = loss + F.cross_entropy(logits_u, labels) + F.cross_entropy(logits_i, labels)
    return loss / (len(layer_embs) - 1)


def bpr_loss(user_e, item_e, layer0, u, i, j, n_users: int, reg: float) -> torch.Tensor:
    pos = (user_e[u] * item_e[i]).sum(1)
    neg = (user_e[u] * item_e[j]).sum(1)
    l2 = (layer0[u].pow(2).sum() + layer0[i + n_users].pow(2).sum() + layer0[j + n_users].pow(2).sum()) / len(u)
    return -F.logsigmoid(pos - neg).mean() + reg * l2


@torch.no_grad()
def rank_metrics(user_e, item_e, users, targets, seen: dict, k_ndcg=10, k_rec=20) -> tuple[float, float]:
    scores = user_e[users] @ item_e.T
    for row, u in enumerate(users.tolist()):
        s = seen.get(u)
        if s:
            scores[row, list(s)] = -1e9
    top = torch.topk(scores, k_rec, dim=1).indices.cpu().numpy()
    ndcg, rec = [], []
    for row, t in enumerate(targets.tolist()):
        hit = np.where(top[row] == t)[0]
        rec.append(float(len(hit) > 0))
        ndcg.append(1 / np.log2(hit[0] + 2) if len(hit) and hit[0] < k_ndcg else 0.0)
    return float(np.mean(ndcg)), float(np.mean(rec))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--cl-weight", type=float, default=0.001, help="tune on validation: 0.0005, 0.001, 0.005")
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--no-contrastive", action="store_true", help="plain LightGCN (ablation)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    name = "lightgcn" if args.no_contrastive else "nlgcl"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    train = pd.read_parquet(data / "train.parquet")
    valid = pd.read_parquet(data / "valid.parquet")
    n_users = len(pd.read_parquet(data / "users.parquet"))
    n_items = len(pd.read_parquet(data / "items.parquet"))
    pos = train[train.label == 1]
    eu, ei = pos.user_idx.to_numpy(), pos.item_idx.to_numpy()
    adj = build_adjacency(eu, ei, n_users, n_items).to(device)
    seen = train.groupby("user_idx").item_idx.agg(set).to_dict()

    warm = valid[valid.user_idx.isin(np.unique(eu))]
    if len(warm) > 5000:
        warm = warm.sample(5000, random_state=args.seed)
    vu = torch.tensor(warm.user_idx.to_numpy(), device=device)
    vi = torch.tensor(warm.item_idx.to_numpy(), device=device)
    vj = torch.tensor(rng.integers(0, n_items, len(warm)), device=device)  # fixed negatives for the validation loss

    model = LightGCN(n_users, n_items, args.dim, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{name}: {n_users:,} users, {n_items:,} games, {len(eu):,} training edges, device {device}")

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = rng.permutation(len(eu))
        total, steps = 0.0, 0
        for start in range(0, len(perm), args.batch):
            b = perm[start: start + args.batch]
            u = torch.tensor(eu[b], device=device)
            i = torch.tensor(ei[b], device=device)
            j = torch.tensor(rng.integers(0, n_items, len(b)), device=device)
            layers = model.propagate(adj)
            ue, ie = model.final(layers)
            loss = bpr_loss(ue, ie, layers[0], u, i, j, n_users, args.reg)
            if not args.no_contrastive:
                loss = loss + args.cl_weight * nlgcl_loss(layers, u, i, n_users, args.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        model.eval()
        with torch.no_grad():
            layers = model.propagate(adj)
            ue, ie = model.final(layers)
            val_loss = bpr_loss(ue, ie, layers[0], vu, vi, vj, n_users, 0.0).item()
            ndcg, rec = rank_metrics(ue, ie, vu, vi, seen)
        train_loss = total / steps
        if not np.isfinite(train_loss):
            raise SystemExit(f"Training loss is {train_loss} at epoch {epoch}: stop and lower --lr")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_ndcg10": ndcg, "val_recall20": rec})
        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"val NDCG@10 {ndcg:.4f} | val Recall@20 {rec:.4f} | {time.time() - t0:.1f}s", flush=True)
        if ndcg > best:
            best, bad = ndcg, 0
            best_state = (ue.cpu().numpy().copy(), ie.cpu().numpy().copy(), epoch)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop: no validation gain for {args.patience} epochs")
                break

    users_e, items_e, best_epoch = best_state
    np.save(data / f"{name}_users.npy", users_e.astype(np.float32))
    np.save(data / f"{name}_items.npy", items_e.astype(np.float32))
    (out / f"{name}_history.json").write_text(json.dumps({"best_epoch": best_epoch, "args": vars(args),
                                                           "history": history}, indent=2))
    plot_curves(history, best_epoch, name.upper(), out / f"{name}_curve.png")
    print(f"Best epoch {best_epoch}: val NDCG@10 {best:.4f}. Saved {name}_users.npy / {name}_items.npy "
          f"and {out / (name + '_curve.png')}")


def plot_curves(history: list[dict], best_epoch: int, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    a.plot(ep, [h["train_loss"] for h in history], label="train loss")
    a.plot(ep, [h["val_loss"] for h in history], label="validation loss (BPR)")
    a.set_xlabel("epoch"), a.set_title(f"{title}: losses"), a.legend()
    b.plot(ep, [h["val_ndcg10"] for h in history], label="val NDCG@10")
    b.plot(ep, [h["val_recall20"] for h in history], label="val Recall@20")
    for ax in (a, b):
        ax.axvline(best_epoch, color="grey", ls=":", lw=1)
        ax.grid(alpha=0.3)
    b.set_xlabel("epoch"), b.set_title(f"{title}: validation ranking"), b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
