"""PlayMatch - skeleton of the innovative model (Week 2, Q2).

GenSAR-style generative retrieval with a player profile:
  input  = task token + profile tokens + history (behavior token + item identifier) + request
  output = identifier of one game, generated code by code by T5-small

What is real in this skeleton
  - T5-small encoder-decoder (pretrained weights from Hugging Face, or the same
    architecture randomly initialised with --offline)
  - the full input format: <task_*>, profile tokens, behavior tokens (<liked>,
    <played>, <disliked>), item identifiers, plain-language request
  - one training step (forward + backward) on real training examples
  - constrained beam search: only identifiers of real games can be generated,
    and a hard platform filter cuts the tree of allowed identifiers

Placeholders, replaced in Week 3
  - item identifiers are random codes; Week 3 uses RQ-VAE codes learned from
    BGE text embeddings (semantic) and NLGCL embeddings (collaborative)
  - requests are template sentences; Week 3 uses LLM-generated requests

Each game has two identifiers that share their first code, as in GenSAR:
  semantic      <s_x> <a_y> <a_z>   (target of search examples)
  collaborative <s_x> <b_y> <b_z>   (target of recommendation examples, and used in the history)

Usage
  python -m src.model_skeleton --data data/processed --users 64
  python -m src.model_skeleton --data data/processed --offline   # no download needed
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

from src.profile import build_profiles, profile_tokens

CODEBOOK = 256
MAX_HISTORY = 20
BEHAVIOR = {1: "<liked>", 0: "<played>", -1: "<disliked>"}
PLATFORMS = ["PlayStation", "Xbox", "Nintendo", "PC", "other", "unknown"]
PRICES = ["under_20", "20_40", "40_60", "over_60", "unknown"]


# ---------------------------------------------------------- item identifiers
def placeholder_identifiers(n_items: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random unique codes (shared, semantic x2) and (shared, collaborative x2) per item."""
    rng = np.random.default_rng(seed)
    shared = rng.integers(0, CODEBOOK, n_items)

    def unique_suffix() -> np.ndarray:
        suffix = rng.integers(0, CODEBOOK, (n_items, 2))
        seen = set()
        for i in range(n_items):
            while (shared[i], *suffix[i]) in seen:
                suffix[i] = rng.integers(0, CODEBOOK, 2)
            seen.add((shared[i], *suffix[i]))
        return suffix

    sem = np.column_stack([shared, unique_suffix()])
    col = np.column_stack([shared, unique_suffix()])
    return sem, col


def sem_tokens(code) -> list[str]:
    return [f"<s_{code[0]}>", f"<a_{code[1]}>", f"<a_{code[2]}>"]


def col_tokens(code) -> list[str]:
    return [f"<s_{code[0]}>", f"<b_{code[1]}>", f"<b_{code[2]}>"]


def special_tokens() -> list[str]:
    toks = ["<task_search>", "<task_rec>", *BEHAVIOR.values(), "<query>"]
    toks += [f"<plat_{p}>" for p in PLATFORMS] + [f"<price_{p}>" for p in PRICES]
    toks += [f"<{c}_{i}>" for c in ("s", "a", "b") for i in range(CODEBOOK)]
    return toks


# ------------------------------------------------------------------ tokenizer
def load_tokenizer(offline: bool, corpus: list[str]):
    if not offline:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("t5-small")
            tok.add_tokens(special_tokens())
            return tok, "t5-small (pretrained tokenizer)"
        except Exception as exc:  # no internet, proxy, ...
            print(f"Could not download t5-small ({exc.__class__.__name__}); falling back to --offline mode")
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = sorted({w for text in corpus for w in text.lower().split()})
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
    tok.add_tokens(special_tokens())
    return tok, "word-level tokenizer (offline)"


def load_model(offline: bool, vocab_size: int, pad_id: int):
    if not offline:
        try:
            model = T5ForConditionalGeneration.from_pretrained("t5-small")
            model.resize_token_embeddings(vocab_size)
            return model, "t5-small (pretrained weights)"
        except Exception as exc:
            print(f"Could not download t5-small weights ({exc.__class__.__name__}); using random init")
    cfg = T5Config(vocab_size=vocab_size, d_model=512, d_ff=2048, num_layers=6, num_heads=8,
                   pad_token_id=pad_id, eos_token_id=1, decoder_start_token_id=pad_id)
    return T5ForConditionalGeneration(cfg), "T5-small architecture (random init, offline)"


# ------------------------------------------------------------------- examples
def template_request(item: pd.Series) -> str:
    """Placeholder request built from the target's metadata, never from its title."""
    plat = item.platform if item.platform != "other" else "any console"
    price = "" if pd.isna(item.price) else f" under {int(item.price) + 10} dollars"
    return f"i am looking for a {plat.lower()} game{price}"


def build_examples(train, items, profiles, sem, col, n_users: int, seed: int):
    rng = np.random.default_rng(seed)
    liked_users = train[train.label == 1].groupby("user_idx").size()
    candidates = liked_users[liked_users >= 2].index.to_numpy()
    chosen = rng.choice(candidates, size=min(n_users, len(candidates)), replace=False)
    prof = profiles.set_index("user_idx")
    items_by_idx = items.set_index("item_idx")
    examples = []
    for n, u in enumerate(chosen):
        hist = train[train.user_idx == u].sort_values("timestamp")
        target_row = hist[hist.label == 1].iloc[-1]                  # last liked game = target
        hist = hist[hist.timestamp < target_row.timestamp].tail(MAX_HISTORY)
        target = int(target_row.item_idx)
        search = n % 2 == 0                                          # half search, half recommendation
        parts = ["<task_search>" if search else "<task_rec>"]
        parts += profile_tokens(prof.loc[u]) if u in prof.index else ["<price_unknown>"]
        for _, h in hist.iterrows():
            parts += [BEHAVIOR[int(h.label)], *col_tokens(col[int(h.item_idx)])]
        if search:
            parts += ["<query>", template_request(items_by_idx.loc[target])]
        target_tokens = sem_tokens(sem[target]) if search else col_tokens(col[target])
        examples.append({"user": int(u), "input": " ".join(parts), "target": target_tokens,
                         "target_item": target, "task": "search" if search else "rec"})
    return examples


def encode(tok, examples, device):
    eos = tok.eos_token_id
    inputs = [tok(e["input"], add_special_tokens=False).input_ids + [eos] for e in examples]
    labels = [tok.convert_tokens_to_ids(e["target"]) + [eos] for e in examples]
    width = max(map(len, inputs))
    ids = torch.full((len(inputs), width), tok.pad_token_id)
    mask = torch.zeros_like(ids)
    for i, seq in enumerate(inputs):
        ids[i, : len(seq)] = torch.tensor(seq)
        mask[i, : len(seq)] = 1
    return ids.to(device), mask.to(device), torch.tensor(labels).to(device)


# -------------------------------------------------- constrained beam search
def build_trie(tok, codes: np.ndarray, kind: str, allowed_items: np.ndarray) -> dict:
    to_tokens = sem_tokens if kind == "sem" else col_tokens
    trie: dict[tuple, set] = {}
    lookup: dict[tuple, int] = {}
    for i in allowed_items:
        seq = tuple(tok.convert_tokens_to_ids(to_tokens(codes[i])))
        lookup[seq] = int(i)
        for d in range(len(seq)):
            trie.setdefault(seq[:d], set()).add(seq[d])
    return trie, lookup


def constrained_generate(model, tok, ids, mask, trie, lookup, n: int = 20):
    eos = tok.eos_token_id

    def allowed(batch_id, prefix):
        prefix = tuple(prefix.tolist()[1:])  # drop decoder start token
        if prefix in lookup:
            return [eos]
        return list(trie.get(prefix, {eos}))

    out = model.generate(input_ids=ids, attention_mask=mask, num_beams=n, num_return_sequences=n,
                         max_new_tokens=4, prefix_allowed_tokens_fn=allowed, early_stopping=True)
    ranked = []
    for seq in out.tolist():
        key = tuple(t for t in seq[1:] if t not in (eos, tok.pad_token_id))
        if key in lookup:
            ranked.append(lookup[key])
    return ranked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--users", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    data = Path(args.data)
    train = pd.read_parquet(data / "train.parquet")
    items = pd.read_parquet(data / "items.parquet")
    profiles = build_profiles(train, items)
    sem, col = placeholder_identifiers(len(items), args.seed)
    examples = build_examples(train, items, profiles, sem, col, args.users, args.seed)

    corpus = [e["input"] for e in examples]
    tok, tok_name = load_tokenizer(args.offline, corpus)
    model, model_name = load_model(args.offline, len(tok), tok.pad_token_id)
    model.to(device)
    log(f"Model: {model_name} | tokenizer: {tok_name} | vocab {len(tok):,} | "
        f"parameters {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M | device {device}")
    log(f"Catalog: {len(items):,} games, {len(examples)} training examples "
        f"({sum(e['task'] == 'search' for e in examples)} search, {sum(e['task'] == 'rec' for e in examples)} rec)")
    log(f"\nExample input  : {examples[0]['input'][:300]}")
    log(f"Example target : {' '.join(examples[0]['target'])}")

    # ---- forward pass + one training step
    ids, mask, labels = encode(tok, examples[: args.batch], device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    t0 = time.time()
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    log(f"\nForward pass OK: input {tuple(ids.shape)}, labels {tuple(labels.shape)}, "
        f"logits {tuple(out.logits.shape)}, loss {out.loss.item():.4f} ({time.time() - t0:.2f}s)")
    assert torch.isfinite(out.loss), "loss is NaN or inf"
    out.loss.backward()
    opt.step()
    opt.zero_grad()
    for _ in range(4):  # a few more steps on the same batch: the loss should go down
        step = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        step.backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        loss_after = model(input_ids=ids, attention_mask=mask, labels=labels).loss.item()
    log(f"Backward pass + 5 AdamW steps OK: loss on the same batch {out.loss.item():.4f} -> {loss_after:.4f}")

    # ---- constrained beam search (predict function stub) with and without the platform filter
    model.eval()
    ex = next(e for e in examples if e["task"] == "search")
    one_ids, one_mask, _ = encode(tok, [ex], device)
    all_items = np.arange(len(items))
    trie, lookup = build_trie(tok, sem, "sem", all_items)
    ranked = constrained_generate(model, tok, one_ids, one_mask, trie, lookup)
    log(f"\nConstrained beam search: {len(ranked)} results, all valid game ids: {all(r in all_items for r in ranked)}")

    plat = items.set_index("item_idx").platform
    want = "Nintendo" if (plat == "Nintendo").any() else plat.iloc[0]
    allowed_items = items[items.platform == want].item_idx.to_numpy()
    trie_f, lookup_f = build_trie(tok, sem, "sem", allowed_items)
    ranked_f = constrained_generate(model, tok, one_ids, one_mask, trie_f, lookup_f)
    ok = all(plat.loc[r] == want for r in ranked_f)
    log(f"With hard platform filter '{want}': {len(ranked_f)} results, all on {want}: {ok}")
    titles = items.set_index("item_idx").title
    log("Top 5 (untrained model, so the order is meaningless for now): "
        + "; ".join(f"{titles.loc[r][:40]} [{plat.loc[r]}]" for r in ranked_f[:5]))

    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / "forward_pass_log.txt").write_text("\n".join(log_lines) + "\n")
    print(f"\nLog saved to {Path(args.out) / 'forward_pass_log.txt'}")


if __name__ == "__main__":
    main()
