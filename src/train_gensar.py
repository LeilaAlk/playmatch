"""PlayMatch - training of the GenSAR model with the player profile (Week 3, step 4).

Training examples (training period only):
  for each warm player, his 5 most recent liked games are targets; the input is the
  profile and the history BEFORE that game (so the profile never contains the target);
  - recommendation example: target = collaborative id
  - search example (when an LLM request exists for that pair): target = semantic id
Validation (each epoch): loss on validation pairs, and NDCG@10 / Recall@20 of the
constrained beam search on validation requests. The best epoch (lowest validation
loss) is saved. Test data is never used here.

Outputs: models/gensar_best/ (or models/gensar_noprofile_best/ with --no-profile),
         results/gensar_curve.png, results/gensar_history.json

Usage (Colab, GPU)
  python -m src.train_gensar --data data/processed --sample 5000 --epochs 3   # quick check
  python -m src.train_gensar --data data/processed --epochs 10
  python -m src.train_gensar --data data/processed --epochs 10 --no-profile   # ablation
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.context import load_context
from src.gensar import Trie, build_input, encode_batch, generate_ranked, load_ids, target_tokens
from src.model_skeleton import load_model, load_tokenizer


def slice_hist(h: dict, end: int) -> dict:
    return {k: v[:end] for k, v in h.items()}


def build_examples(ctx, sem, col, queries, args, rng):
    train_q = {(r.user_idx, r.item_idx): r.query for r in queries[queries.split == "train"].itertuples()}
    users = sorted(ctx.warm_hist)
    if args.sample:
        users = sorted(rng.choice(users, size=min(args.sample, len(users)), replace=False).tolist())
    train = []
    for u in users:
        h = ctx.warm_hist[u]
        pos = [p for p in np.where(h["label"] == 1)[0] if p >= 1][-args.max_targets:]
        for p in pos:
            before, item = slice_hist(h, p), int(h["item_idx"][p])
            prof = ctx.profile_tokens(before)
            train.append((build_input(prof, before, col, None, not args.no_profile), target_tokens(item, sem, col, False)))
            q = train_q.get((u, item))
            if q:
                train.append((build_input(prof, before, col, q, not args.no_profile), target_tokens(item, sem, col, True)))

    valid = pd.read_parquet(ctx.data / "valid.parquet")
    valid = valid[valid.user_idx.isin(ctx.warm_hist)]
    if args.sample:
        valid = valid[valid.user_idx.isin(users)]
    valid = valid.sample(min(3000, len(valid)), random_state=args.seed)
    val = []
    for r in valid.itertuples():
        h = ctx.warm_hist[r.user_idx]
        val.append((build_input(ctx.profile_tokens(h), h, col, None, not args.no_profile),
                    target_tokens(r.item_idx, sem, col, False)))
    vq = queries[(queries.split == "valid") & queries.user_idx.isin(ctx.warm_hist)]
    if args.sample:
        vq = vq[vq.user_idx.isin(users)]
    val_search = []
    for r in vq.itertuples():
        h = ctx.warm_hist[r.user_idx]
        text = build_input(ctx.profile_tokens(h), h, col, r.query, not args.no_profile)
        val.append((text, target_tokens(r.item_idx, sem, col, True)))
        val_search.append((text, int(r.item_idx), set(h["item_idx"].tolist())))
    return train, val, val_search[: args.val_rank]


def batches(examples, size, shuffle, rng):
    idx = list(range(len(examples)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), size):
        yield [examples[i] for i in idx[s: s + size]]


@torch.no_grad()
def validate(model, tok, val, val_search, trie, device, batch: int) -> tuple[float, float, float]:
    model.eval()
    total, n = 0.0, 0
    for b in batches(val, batch, False, None):
        ids, mask, labels = encode_batch(tok, [x for x, _ in b], [y for _, y in b], device)
        total += model(input_ids=ids, attention_mask=mask, labels=labels).loss.item() * len(b)
        n += len(b)
    ndcg, rec = [], []
    for s in range(0, len(val_search), 16):
        chunk = val_search[s: s + 16]
        ranked = generate_ranked(model, tok, [c[0] for c in chunk], trie, 30, device, [c[2] for c in chunk])
        for (_, target, _), r in zip(chunk, ranked):
            rec.append(float(target in r[:20]))
            ndcg.append(1 / np.log2(r.index(target) + 2) if target in r[:10] else 0.0)
    return total / max(n, 1), float(np.mean(ndcg)) if ndcg else float("nan"), float(np.mean(rec)) if rec else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-targets", type=int, default=5)
    ap.add_argument("--sample", type=int, default=0, help="train on this many players only (quick check)")
    ap.add_argument("--val-rank", type=int, default=500, help="validation requests ranked each epoch")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--no-profile", action="store_true", help="ablation: no profile tokens in the input")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    py_rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = "gensar_noprofile" if args.no_profile else "gensar"

    data = Path(args.data)
    ctx = load_context(data)
    sem, col = load_ids(data)
    from src.queries import load_queries
    queries = load_queries(data)
    train, val, val_search = build_examples(ctx, sem, col, queries, args, rng)
    print(f"{name}: {len(train):,} training examples, {len(val):,} validation examples, "
          f"{len(val_search)} validation requests ranked each epoch, device {device}")

    corpus = [x for x, _ in train + val] + queries["query"].tolist()
    tok, tok_name = load_tokenizer(args.offline, corpus)
    model, model_name = load_model(args.offline, len(tok), tok.pad_token_id)
    model.to(device)
    print(f"Model: {model_name} | tokenizer: {tok_name} | vocab {len(tok):,}")
    trie = Trie(tok, sem, "sem", np.ones(len(sem), dtype=bool))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps))) * max(0.0, 1 - s / steps))
    save_dir = Path(args.models) / f"{name}_best"
    history, best, bad = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for b in batches(train, args.batch, True, py_rng):
            ids, mask, labels = encode_batch(tok, [x for x, _ in b], [y for _, y in b], device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"Loss is {loss.item()} at epoch {epoch}: stop, lower --lr")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item() * len(b), n + len(b)
        val_loss, ndcg, rec = validate(model, tok, val, val_search, trie, device, args.batch)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                        "val_ndcg10": ndcg, "val_recall20": rec})
        print(f"epoch {epoch:2d} | train {total / n:.4f} | val loss {val_loss:.4f} | val NDCG@10 {ndcg:.4f} | "
              f"val Recall@20 {rec:.4f} | {time.time() - t0:.0f}s", flush=True)
        if val_loss < best:
            best, bad = val_loss, 0
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tok.save_pretrained(save_dir)
            (save_dir / "playmatch.json").write_text(json.dumps({"use_profile": not args.no_profile,
                                                                  "epoch": epoch, "val_loss": val_loss}))
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop: validation loss has not improved for {args.patience} epochs")
                break

    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / f"{name}_history.json").write_text(json.dumps({"args": vars(args), "history": history}, indent=2))
    best_epoch = min(history, key=lambda h: h["val_loss"])["epoch"]
    from src.nlgcl import plot_curves
    plot_curves([{**h, "val_ndcg10": h["val_ndcg10"], "val_recall20": h["val_recall20"]} for h in history],
                best_epoch, name.upper(), Path(args.out) / f"{name}_curve.png")
    print(f"Best epoch {best_epoch} saved to {save_dir}; curves in {Path(args.out) / (name + '_curve.png')}")


if __name__ == "__main__":
    main()
