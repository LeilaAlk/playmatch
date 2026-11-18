"""Shared loading of the processed data for the models, the evaluation and the API."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.profile import tokens_from_history

EMPTY = {"item_idx": np.zeros(0, int), "label": np.zeros(0, int), "timestamp": np.zeros(0, float),
         "gift": np.zeros(0, bool)}


def _histories(df: pd.DataFrame) -> dict[int, dict]:
    df = df.sort_values(["user_idx", "timestamp"])
    out = {}
    for u, g in df.groupby("user_idx", sort=False):
        out[int(u)] = {"item_idx": g.item_idx.to_numpy(), "label": g.label.to_numpy(),
                       "timestamp": g.timestamp.to_numpy(dtype=float), "gift": g.gift_flag.to_numpy(dtype=bool)}
    return out


@dataclass
class Context:
    data: Path
    items: pd.DataFrame
    users: pd.DataFrame
    platform: np.ndarray
    price: np.ndarray
    title: np.ndarray
    popularity: np.ndarray
    warm_hist: dict = field(repr=False)
    cold_hist: dict = field(repr=False)
    cold: set = field(repr=False)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def history(self, user_idx: int, onboarding: bool = True) -> dict:
        """Training history for warm users; the hidden earlier reviews (onboarding) for cold users."""
        if user_idx in self.cold:
            return self.cold_hist.get(user_idx, EMPTY) if onboarding else EMPTY
        return self.warm_hist.get(user_idx, EMPTY)

    def profile_tokens(self, hist: dict, override: dict | None = None) -> list[str]:
        return tokens_from_history(hist["item_idx"], hist["label"], hist["timestamp"], hist["gift"],
                                   self.platform, self.price, override=override)

    def allowed_mask(self, platform: str | None = None, max_price: float | None = None) -> np.ndarray:
        """Hard limits asked by the player: platform and/or budget (games without a price are excluded)."""
        mask = np.ones(self.n_items, dtype=bool)
        if platform:
            mask &= self.platform == platform
        if max_price is not None and not np.isnan(max_price):
            mask &= ~np.isnan(self.price) & (self.price <= max_price)
        return mask


def load_context(data_dir: str | Path) -> Context:
    data = Path(data_dir)
    items = pd.read_parquet(data / "items.parquet").sort_values("item_idx").reset_index(drop=True)
    users = pd.read_parquet(data / "users.parquet")
    train = pd.read_parquet(data / "train.parquet")
    cold_df = pd.read_parquet(data / "cold_history.parquet")
    pop = np.bincount(train[train.label == 1].item_idx, minlength=len(items)).astype(float)
    return Context(
        data=data, items=items, users=users,
        platform=items.platform.to_numpy(dtype=object), price=items.price.to_numpy(dtype=float),
        title=items.title.to_numpy(dtype=object), popularity=pop,
        warm_hist=_histories(train), cold_hist=_histories(cold_df), cold=set(users[users.is_cold].user_idx),
    )
