"""PlayMatch - backend API (Week 3: real predictions).

/recommend calls the trained model: GenSAR (models/gensar_best) if it exists, else
plan B (BGE + NLGCL + profile), else the Week 2 mock. Choose with PLAYMATCH_MODEL=
gensar | twostage | mock | auto (default).

Who is the player?
  - "player_<n>"   : player n of the dataset (his training history, or onboarding if cold) - for the demo
  - any other id   : an app user. His profile (/profile) and his likes/dislikes (/log_interaction)
                     become his history, so the recommendations change as he uses the app.

Discovery loop ("rollback"): /next proposes ONE game; the player answers like / dislike
(/log_interaction). A dislike is stored as a negative game in his history, the game is never
proposed again, and the next proposal is re-scored away from it (and toward the games he liked).
/undo removes the player's last answer: the history goes back to the state before it.

Run:   uvicorn api.main:app --port 8000          Docs:  http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

DATA_DIR = Path(os.environ.get("PLAYMATCH_DATA", "data/processed"))
DB_PATH = Path(os.environ.get("PLAYMATCH_DB", "data/app.db"))
MODEL_KIND = os.environ.get("PLAYMATCH_MODEL", "auto")
Platform = Literal["PlayStation", "Xbox", "Nintendo", "PC", "other"]

app = FastAPI(title="PlayMatch API", version="0.2-week3")


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


# --------------------------------------------------------- model and data
class State:
    ctx = None
    predictor = None
    catalog = None      # DataFrame for the mock and /games
    asin_to_idx: dict = {}
    text_emb = None     # BGE vectors of the games (for the discovery loop and the explanations)


def startup() -> None:
    try:
        from src.context import load_context
        State.ctx = load_context(DATA_DIR)
        items = State.ctx.items.assign(popularity=State.ctx.popularity)
        State.catalog = items.sort_values("popularity", ascending=False).reset_index(drop=True)
        State.asin_to_idx = dict(zip(items.parent_asin, items.item_idx))
        if (DATA_DIR / "items_bge.npy").exists():
            State.text_emb = np.load(DATA_DIR / "items_bge.npy")
        if MODEL_KIND != "mock":
            from src.predict import load_predictor
            State.predictor = load_predictor(State.ctx, MODEL_KIND)
    except FileNotFoundError:
        State.catalog = pd.DataFrame({
            "item_idx": range(5), "parent_asin": [f"DUMMY{i}" for i in range(5)],
            "title": ["Dummy Farm", "Dummy Racer", "Dummy Quest", "Dummy Puzzle", "Dummy Party"],
            "platform": ["Nintendo", "PlayStation", "PC", "Nintendo", "Xbox"],
            "price": [29.99, 59.99, 19.99, 9.99, 39.99], "popularity": [5, 4, 3, 2, 1]})
        State.asin_to_idx = dict(zip(State.catalog.parent_asin, State.catalog.item_idx))


startup()


def model_name() -> str:
    return State.predictor.name if State.predictor else "mock"


# ------------------------------------------------------------------- schemas
class RecommendRequest(BaseModel):
    user_id: str = Field(examples=["demo_user", "player_42"])
    query: str = Field("", examples=["a relaxing game to play on the couch with my boyfriend, no shooting"])
    platform: Optional[Platform] = None
    max_price: Optional[float] = Field(None, gt=0)
    k: int = Field(10, ge=1, le=50)


class Interaction(BaseModel):
    user_id: str
    item_id: str
    action: Literal["click", "like", "dislike", "not_interested"]
    query: str = ""


class Profile(BaseModel):
    platforms: list[Platform] = []
    favorite_genres: list[str] = []
    games_played: list[str] = Field([], description="parent_asin ids of games the player liked (see /games)")
    price_range: Literal["under_20", "20_40", "40_60", "over_60", "unknown"] = "unknown"
    gender: Optional[Literal["girl", "boy", "other"]] = None  # optional, stored only, not used by the model


# ---------------------------------------------------------------- history
EMPTY_HIST = {"item_idx": np.zeros(0, int), "label": np.zeros(0, int), "timestamp": np.zeros(0, float),
              "gift": np.zeros(0, bool)}


def app_events(user_id: str) -> tuple[dict, list]:
    with db() as con:
        row = con.execute("SELECT profile_json FROM profiles WHERE user_id=?", (user_id,)).fetchone()
        events = con.execute("SELECT item_id, action, ts FROM interactions WHERE user_id=? ORDER BY id",
                             (user_id,)).fetchall()
    return (json.loads(row[0]) if row else {}), events


def player_state(user_id: str) -> dict:
    """History used by the model = dataset history (player_<n>) or declared games (app user)
    + the likes / dislikes given in the app. Also returns what the player already answered."""
    prof, events = app_events(user_id)
    u, base = None, EMPTY_HIST
    if user_id.startswith("player_") and user_id[7:].isdigit():
        u = int(user_id[7:])
        if State.ctx is None or u >= len(State.ctx.users):
            raise HTTPException(404, f"no dataset player {u}")
        base = State.ctx.history(u)
    now = time.time() * 1000
    items, labels, ts = list(base["item_idx"]), list(base["label"]), list(base["timestamp"])
    for asin in prof.get("games_played", []):
        if asin in State.asin_to_idx:
            items.append(State.asin_to_idx[asin]), labels.append(1), ts.append(now - 1e6)
    acted, likes, dislikes = set(), [], []
    for asin, action, t in events:
        idx = State.asin_to_idx.get(asin)
        if idx is None:
            continue
        acted.add(idx)
        if action in ("like", "dislike"):
            items.append(idx), labels.append(1 if action == "like" else -1), ts.append(t * 1000)
            (likes if action == "like" else dislikes).append(idx)
    hist = {"item_idx": np.array(items, dtype=int), "label": np.array(labels, dtype=int),
            "timestamp": np.array(ts, dtype=float), "gift": np.zeros(len(items), dtype=bool)}
    if len(base["gift"]):
        hist["gift"][: len(base["gift"])] = base["gift"]
    override = None
    if prof.get("platforms"):
        override = {"platforms": prof["platforms"], "price_bucket": prof.get("price_range", "unknown")}
    return {"hist": hist, "override": override, "user_idx": u, "acted": acted, "likes": likes, "dislikes": dislikes}


def candidates(st: dict, query: str, k: int, platform, max_price) -> list[int]:
    if State.predictor is None:
        games = State.catalog
        if platform:
            games = games[games.platform == platform]
        if max_price:
            games = games[games.price.notna() & (games.price <= max_price)]
        idx = games.item_idx.tolist()
    else:
        idx = State.predictor.recommend(st["hist"], query or None, user_idx=st["user_idx"], k=k + len(st["acted"]),
                                        platform=platform, max_price=max_price, override=st["override"])
    seen = st["acted"] | set(st["hist"]["item_idx"].tolist())
    return [i for i in idx if i not in seen][:k]


def explain(idx: int, st: dict) -> str:
    reasons = []
    liked = st["hist"]["item_idx"][st["hist"]["label"] == 1]
    if State.text_emb is not None and len(liked):
        sims = State.text_emb[liked] @ State.text_emb[idx]
        j = int(liked[int(np.argmax(sims))])
        reasons.append(f"close to {State.ctx.title[j][:45]}, which you liked")
    if State.ctx is not None:
        toks = State.ctx.profile_tokens(st["hist"], st["override"])
        plats = {t[6:-1] for t in toks if t.startswith("<plat_")}
        if State.ctx.platform[idx] in plats:
            reasons.append(f"on {State.ctx.platform[idx]}, a platform you play on")
    return "; ".join(reasons) or "popular with players like you"


def card(idx: int, rank: int, why: str | None = None) -> dict:
    r = State.catalog.set_index("item_idx").loc[idx]
    get = lambda c: (None if c not in r.index or pd.isna(r[c]) else r[c])  # noqa: E731
    out = {"rank": rank, "item_id": r.parent_asin, "title": r.title, "platform": r.platform,
           "price": None if pd.isna(r.price) else float(r.price),
           "release": get("release_date") or (None if get("release_year") is None else str(int(get("release_year")))),
           "publisher": get("store") or get("manufacturer"), "image_url": get("image_url"),
           "age_rating": get("age_rating"),
           "average_rating": None if get("average_rating") is None else float(get("average_rating"))}
    if why:
        out["why"] = why
    return out


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model": model_name(), "catalog_size": len(State.catalog), "data_dir": str(DATA_DIR)}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    t0 = time.time()
    st = player_state(req.user_id)
    idx = candidates(st, req.query, req.k, req.platform, req.max_price)
    return {"mock": State.predictor is None, "model": model_name(), "user_id": req.user_id, "query": req.query,
            "latency_ms": round(1000 * (time.time() - t0), 1),
            "items": [card(i, r + 1) for r, i in enumerate(idx)]}


class NextRequest(BaseModel):
    user_id: str = Field(examples=["demo_user", "player_42"])
    query: str = ""
    platform: Optional[Platform] = None
    max_price: Optional[float] = Field(None, gt=0)


@app.post("/next")
def next_game(req: NextRequest):
    """Discovery loop: ONE game at a time, adjusted by the answers given so far."""
    t0 = time.time()
    st = player_state(req.user_id)
    cands = candidates(st, req.query, 30, req.platform, req.max_price)
    if not cands:
        raise HTTPException(404, "no more games match these limits")
    adjusted = bool(State.text_emb is not None and (st["likes"] or st["dislikes"]))
    if adjusted:
        e = State.text_emb
        base = 1 - np.arange(len(cands)) / len(cands)                 # model order
        pen = (e[cands] @ e[st["dislikes"]].T).max(1) if st["dislikes"] else np.zeros(len(cands))
        bonus = (e[cands] @ e[st["likes"]].T).max(1) if st["likes"] else np.zeros(len(cands))
        best = cands[int(np.argmax(base - 0.6 * pen + 0.3 * bonus))]  # away from dislikes, toward likes
    else:
        best = cands[0]
    return {"mock": State.predictor is None, "model": model_name(), "adjusted": adjusted,
            "answers_so_far": {"liked": len(st["likes"]), "disliked": len(st["dislikes"])},
            "latency_ms": round(1000 * (time.time() - t0), 1),
            "game": card(best, 1, explain(best, st) if State.ctx is not None else None)}


class UndoRequest(BaseModel):
    user_id: str


@app.post("/undo")
def undo(req: UndoRequest):
    """Rollback: delete the player's last answer; the next proposal is computed without it."""
    with db() as con:
        row = con.execute("SELECT id, item_id, action FROM interactions WHERE user_id=? ORDER BY id DESC LIMIT 1",
                          (req.user_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "nothing to undo")
        con.execute("DELETE FROM interactions WHERE id=?", (row[0],))
    return {"undone": {"id": row[0], "item_id": row[1], "action": row[2]}}


@app.get("/games")
def search_games(q: str = "", limit: int = 20):
    """Title search, most popular first (used by the interface to declare games played)."""
    games = State.catalog
    if q:
        games = games[games.title.str.contains(q, case=False, regex=False, na=False)]
    return [card(int(r.item_idx), k + 1) for k, r in enumerate(games.head(limit).itertuples())]


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
