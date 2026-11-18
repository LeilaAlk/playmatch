"""PlayMatch - prediction with a trained model (used by the API and by evaluate_model.py).

Both models expose the same method:
  recommend(hist, query=None, user_idx=None, k=10, platform=None, max_price=None, override=None) -> [item_idx]
hist = the player's history (dict of arrays, see src/context.py); override = profile typed in the app.

  GenSARPredictor : trained T5 model (models/gensar_best), constrained beam search
  TwoStagePredictor: plan B (BGE retrieval + NLGCL/profile re-ranking)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.context import Context
from src.gensar import Trie, build_input, generate_ranked, load_ids


class GenSARPredictor:
    name = "gensar"

    def __init__(self, ctx: Context, model_dir: str | Path = "models/gensar_best", device: str | None = None):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.ctx = ctx
        self.model_dir = Path(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_dir).to(self.device).eval()
        cfg = json.loads((self.model_dir / "playmatch.json").read_text())
        self.use_profile = cfg.get("use_profile", True)
        self.name = "gensar" if self.use_profile else "gensar_noprofile"
        self.sem, self.col = load_ids(ctx.data)

    @lru_cache(maxsize=64)
    def trie(self, kind: str, platform: str | None, max_price: float | None) -> Trie:
        codes = self.sem if kind == "sem" else self.col
        return Trie(self.tok, codes, kind, self.ctx.allowed_mask(platform, max_price))

    def inputs(self, hist: dict, query: str | None, override: dict | None) -> str:
        return build_input(self.ctx.profile_tokens(hist, override), hist, self.col, query, self.use_profile)

    def recommend(self, hist: dict, query: str | None = None, user_idx: int | None = None, k: int = 10,
                  platform: str | None = None, max_price: float | None = None, override: dict | None = None,
                  beams: int | None = None) -> list[int]:
        return self.recommend_batch([hist], [query], k, platform, max_price, [override], beams)[0]

    def recommend_batch(self, hists: list[dict], queries: list[str | None], k: int = 20,
                        platform: str | None = None, max_price: float | None = None,
                        overrides: list | None = None, beams: int | None = None) -> list[list[int]]:
        """All rows of a batch share the same task (search if queries[0]) and the same limits."""
        mp = None if max_price is None or np.isnan(max_price) else float(max_price)
        trie = self.trie("sem" if queries[0] else "col", platform, mp)
        overrides = overrides or [None] * len(hists)
        texts = [self.inputs(h, q, o) for h, q, o in zip(hists, queries, overrides)]
        exclude = [set(h["item_idx"].tolist()) for h in hists]
        beams = beams or max(k + 10, 20)
        return generate_ranked(self.model, self.tok, texts, trie, beams, self.device, exclude, k)


class TwoStagePredictor:
    name = "twostage"

    def __init__(self, ctx: Context, collab: str = "nlgcl", weights_file: str | Path = "results/twostage_nlgcl_weights.json"):
        from src.twostage import TwoStage
        weights = json.loads(Path(weights_file).read_text()) if Path(weights_file).exists() else None
        self.model = TwoStage(ctx, collab, weights)
        self.ctx = ctx

    def recommend(self, hist: dict, query: str | None = None, user_idx: int | None = None, k: int = 10,
                  platform: str | None = None, max_price: float | None = None, override: dict | None = None,
                  **_) -> list[int]:
        if query:
            query_vec = self.model.encode([query])[0]
        else:  # no request ("For you" page): start from the average text of the player's own games
            items = hist["item_idx"]
            query_vec = self.model.text[items].mean(0) if len(items) else self.model.text.mean(0)
        return self.model.recommend(query_vec, hist, user_idx, k, platform, max_price, override)


def load_predictor(ctx: Context, kind: str = "auto"):
    """auto = GenSAR if a trained model exists, else plan B if its embeddings exist, else None (mock)."""
    if kind in ("auto", "gensar") and (Path("models/gensar_best") / "playmatch.json").exists():
        return GenSARPredictor(ctx)
    if kind in ("auto", "twostage") and (ctx.data / "nlgcl_items.npy").exists():
        return TwoStagePredictor(ctx)
    return None
