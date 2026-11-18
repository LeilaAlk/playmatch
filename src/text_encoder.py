"""Text encoder shared by every script, so that games and requests live in the same space.

Default: BGE (BAAI/bge-small-en-v1.5, 384 dimensions) via sentence-transformers, as in GenSAR.
Offline fallback (no internet): TF-IDF + SVD fitted on the game texts. Only for tests.
The choice made by src/embeddings.py is written in data/processed/text_encoder.json,
and every other script reloads the same encoder from there.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def item_text(row) -> str:
    """Text of a game: title, publisher, category path, description (never review text)."""
    parts = [str(row.get("title") or ""), str(row.get("store") or ""),
             str(row.get("categories") or ""), str(row.get("description") or "")[:1000]]
    return ". ".join(p for p in parts if p)


class TextEncoder:
    def __init__(self, mode: str, model=None):
        self.mode, self.model = mode, model

    # ------------------------------------------------------------- building
    @classmethod
    def build(cls, texts: list[str], offline: bool, data_dir: Path) -> "TextEncoder":
        if not offline:
            try:
                from sentence_transformers import SentenceTransformer
                enc = cls("bge", SentenceTransformer(BGE_MODEL))
                (data_dir / "text_encoder.json").write_text(json.dumps({"mode": "bge", "model": BGE_MODEL}))
                return enc
            except Exception as exc:
                print(f"Could not load {BGE_MODEL} ({exc.__class__.__name__}); using the offline TF-IDF encoder")
        import joblib
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline
        dim = max(2, min(256, len(texts) - 1))
        pipe = make_pipeline(TfidfVectorizer(min_df=1, max_features=50_000, sublinear_tf=True),
                             TruncatedSVD(n_components=dim, random_state=0))
        pipe.fit(texts)
        joblib.dump(pipe, data_dir / "text_encoder_offline.joblib")
        (data_dir / "text_encoder.json").write_text(json.dumps({"mode": "offline", "model": "tfidf+svd"}))
        return cls("offline", pipe)

    @classmethod
    def load(cls, data_dir: Path) -> "TextEncoder":
        info = json.loads((data_dir / "text_encoder.json").read_text())
        if info["mode"] == "bge":
            from sentence_transformers import SentenceTransformer
            return cls("bge", SentenceTransformer(info["model"]))
        import joblib
        return cls("offline", joblib.load(data_dir / "text_encoder_offline.joblib"))

    # ------------------------------------------------------------- encoding
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)

    def encode_items(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.mode == "bge":
            return self._normalize(self.model.encode(texts, batch_size=batch_size, show_progress_bar=True))
        return self._normalize(self.model.transform(texts))

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.mode == "bge":
            texts = [BGE_QUERY_PREFIX + t for t in texts]
            return self._normalize(self.model.encode(texts, batch_size=batch_size, show_progress_bar=False))
        return self._normalize(self.model.transform(texts))
