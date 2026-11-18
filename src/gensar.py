"""GenSAR building blocks shared by training (train_gensar.py), prediction (predict.py) and evaluation.

Input text  : <task_search|task_rec> [profile tokens] (behavior token + collaborative id) x history [<query> request]
Output      : the 3 codes of one game (semantic id for search, collaborative id for recommendation)
Decoding    : constrained beam search over a prefix tree of real game ids (optionally cut by hard limits)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model_skeleton import BEHAVIOR, col_tokens, sem_tokens

MAX_HISTORY = 20


def load_ids(data: Path) -> tuple[np.ndarray, np.ndarray]:
    ids = pd.read_parquet(data / "semantic_ids.parquet").sort_values("item_idx")
    return ids[["s", "a1", "a2"]].to_numpy(), ids[["s", "b1", "b2"]].to_numpy()


def build_input(profile: list[str], hist: dict, col: np.ndarray, query: str | None = None,
                use_profile: bool = True, max_history: int = MAX_HISTORY) -> str:
    parts = ["<task_search>" if query else "<task_rec>"]
    if use_profile:
        parts += profile
    items, labels = hist["item_idx"][-max_history:], hist["label"][-max_history:]
    for i, lab in zip(items, labels):
        parts += [BEHAVIOR[int(lab)], *col_tokens(col[int(i)])]
    if query:
        parts += ["<query>", " ".join(str(query).lower().split())]
    return " ".join(parts)


def target_tokens(item: int, sem: np.ndarray, col: np.ndarray, search: bool) -> list[str]:
    return sem_tokens(sem[item]) if search else col_tokens(col[item])


def encode_batch(tok, inputs: list[str], targets: list[list[str]] | None, device, max_len: int = 256):
    eos = tok.eos_token_id
    seqs = [tok(t, add_special_tokens=False).input_ids[: max_len - 1] + [eos] for t in inputs]
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), tok.pad_token_id)
    mask = torch.zeros_like(ids)
    for k, s in enumerate(seqs):
        ids[k, : len(s)] = torch.tensor(s)
        mask[k, : len(s)] = 1
    labels = None
    if targets is not None:
        labels = torch.tensor([tok.convert_tokens_to_ids(t) + [eos] for t in targets])
        labels = labels.to(device)
    return ids.to(device), mask.to(device), labels


class Trie:
    """Prefix tree of the allowed game ids, in token ids."""

    def __init__(self, tok, codes: np.ndarray, kind: str, allowed: np.ndarray):
        to_tokens = sem_tokens if kind == "sem" else col_tokens
        self.next: dict[tuple, list[int]] = {}
        self.item: dict[tuple, int] = {}
        nxt: dict[tuple, set] = {}
        for i in np.where(allowed)[0]:
            seq = tuple(tok.convert_tokens_to_ids(to_tokens(codes[i])))
            self.item[seq] = int(i)
            for d in range(len(seq)):
                nxt.setdefault(seq[:d], set()).add(seq[d])
        self.next = {k: sorted(v) for k, v in nxt.items()}
        self.eos = tok.eos_token_id
        self.pad = tok.pad_token_id

    def allowed_fn(self):
        def fn(batch_id, prefix):
            p = tuple(prefix.tolist()[1:])  # drop the decoder start token
            if p in self.item:
                return [self.eos]
            return self.next.get(p, [self.eos])
        return fn

    def decode(self, seq: list[int]) -> int | None:
        return self.item.get(tuple(t for t in seq[1:] if t not in (self.eos, self.pad)))


@torch.no_grad()
def generate_ranked(model, tok, inputs: list[str], trie: Trie, beams: int, device,
                    exclude: list[set] | None = None, k: int = 20) -> list[list[int]]:
    ids, mask, _ = encode_batch(tok, inputs, None, device)
    out = model.generate(input_ids=ids, attention_mask=mask, num_beams=beams, num_return_sequences=beams,
                         max_new_tokens=4, prefix_allowed_tokens_fn=trie.allowed_fn(), early_stopping=True)
    seqs = out.tolist()
    ranked = []
    for row in range(len(inputs)):
        seen = exclude[row] if exclude else set()
        items = []
        for seq in seqs[row * beams: (row + 1) * beams]:
            it = trie.decode(seq)
            if it is not None and it not in seen and it not in items:
                items.append(it)
        ranked.append(items[:k])
    return ranked
