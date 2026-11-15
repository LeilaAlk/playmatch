"""PlayMatch - backend API (Week 2, Q4).

/recommend returns MOCK recommendations for now: the most popular games of the
training data, filtered by the platform and budget the player asks for. In Week 3
the same endpoint calls the trained model (src/model_skeleton.py -> predict).

/log_interaction and /profile already write to a real SQLite database.

Run:   uvicorn api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DATA_DIR = Path(os.environ.get("PLAYMATCH_DATA", "data/processed"))
DB_PATH = Path(os.environ.get("PLAYMATCH_DB", "data/app.db"))

app = FastAPI(title="PlayMatch API", version="0.1-week2")


# ------------------------------------------------------------------ storage
def db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS interactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, query TEXT,
        item_id TEXT NOT NULL, action TEXT NOT NULL, ts REAL NOT NULL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS profiles (
        user_id TEXT PRIMARY KEY, profile_json TEXT NOT NULL, updated_ts REAL NOT NULL)""")
    return con


def load_catalog() -> pd.DataFrame:
    """Games ranked by popularity in the training data; a tiny dummy catalog if the data is missing."""
    try:
        items = pd.read_parquet(DATA_DIR / "items.parquet")
        train = pd.read_parquet(DATA_DIR / "train.parquet")
        pop = train[train.label == 1].item_idx.value_counts()
        items["popularity"] = items.item_idx.map(pop).fillna(0).astype(int)
        return items.sort_values("popularity", ascending=False).reset_index(drop=True)
    except FileNotFoundError:
        return pd.DataFrame({
            "parent_asin": [f"DUMMY{i}" for i in range(5)],
            "title": ["Dummy Farm", "Dummy Racer", "Dummy Quest", "Dummy Puzzle", "Dummy Party"],
            "platform": ["Nintendo", "PlayStation", "PC", "Nintendo", "Xbox"],
            "price": [29.99, 59.99, 19.99, 9.99, 39.99], "popularity": [5, 4, 3, 2, 1],
        })


CATALOG = load_catalog()


# ------------------------------------------------------------------- schemas
class RecommendRequest(BaseModel):
    user_id: str = Field(examples=["demo_user"])
    query: str = Field("", examples=["a relaxing game to play on the couch with my boyfriend, no shooting"])
    platform: Optional[Literal["PlayStation", "Xbox", "Nintendo", "PC", "other"]] = None
    max_price: Optional[float] = Field(None, gt=0)
    k: int = Field(10, ge=1, le=50)


class Interaction(BaseModel):
    user_id: str
    item_id: str
    action: Literal["click", "like", "dislike", "not_interested"]
    query: str = ""


class Profile(BaseModel):
    platforms: list[str] = []
    favorite_genres: list[str] = []
    games_played: list[str] = []
    price_range: Literal["under_20", "20_40", "40_60", "over_60", "unknown"] = "unknown"
    gender: Optional[Literal["girl", "boy", "other"]] = None  # optional, stored only, not used by the model


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "catalog_size": len(CATALOG), "data_dir": str(DATA_DIR)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    games = CATALOG
    if req.platform:
        games = games[games.platform == req.platform]
    if req.max_price:
        games = games[games.price.notna() & (games.price <= req.max_price)]
    top = games.head(req.k)
    return {
        "mock": True,
        "note": "Week 2 mock: most popular games after the platform/budget filter. The trained model replaces this in Week 3.",
        "user_id": req.user_id, "query": req.query,
        "items": [{"rank": i + 1, "item_id": r.parent_asin, "title": r.title, "platform": r.platform,
                   "price": None if pd.isna(r.price) else float(r.price)}
                  for i, r in enumerate(top.itertuples())],
    }


@app.post("/log_interaction")
def log_interaction(event: Interaction):
    with db() as con:
        cur = con.execute("INSERT INTO interactions (user_id, query, item_id, action, ts) VALUES (?,?,?,?,?)",
                          (event.user_id, event.query, event.item_id, event.action, time.time()))
    return {"stored": True, "id": cur.lastrowid}


@app.get("/interactions/{user_id}")
def list_interactions(user_id: str):
    with db() as con:
        rows = con.execute("SELECT id, query, item_id, action, ts FROM interactions WHERE user_id=? ORDER BY id",
                           (user_id,)).fetchall()
    return [dict(zip(["id", "query", "item_id", "action", "ts"], r)) for r in rows]


@app.put("/profile/{user_id}")
def put_profile(user_id: str, profile: Profile):
    with db() as con:
        con.execute("INSERT OR REPLACE INTO profiles (user_id, profile_json, updated_ts) VALUES (?,?,?)",
                    (user_id, profile.model_dump_json(), time.time()))
    return {"stored": True, "user_id": user_id}


@app.get("/profile/{user_id}")
def get_profile(user_id: str):
    with db() as con:
        row = con.execute("SELECT profile_json FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "no profile for this user")
    return {"user_id": user_id, **json.loads(row[0])}
