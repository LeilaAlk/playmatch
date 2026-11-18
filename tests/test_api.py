"""Smoke test of the API (run: python tests/test_api.py, or python -m pytest tests/test_api.py -q).

Works with the mock (no data) and with a trained model (data/processed + models/).
"""
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PLAYMATCH_DB", str(Path(tempfile.mkdtemp()) / "test.db"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_all() -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    print("Model served:", health["model"])

    games = client.get("/games", params={"limit": 3}).json()
    assert games, "empty catalog"
    t0 = time.time()
    r = client.post("/recommend", json={"user_id": "u1", "query": "a relaxing game to play with a friend",
                                        "platform": "Nintendo", "k": 5})
    body = r.json()
    assert r.status_code == 200, body
    assert all(it["platform"] == "Nintendo" for it in body["items"])
    print(f"/recommend ({body['model']}, mock={body['mock']}): {len(body['items'])} games in "
          f"{1000 * (time.time() - t0):.0f} ms (server: {body['latency_ms']} ms)")

    item = (body["items"] or games)[0]["item_id"]
    assert client.post("/log_interaction", json={"user_id": "u1", "item_id": item, "action": "like",
                                                 "query": "a relaxing game"}).json()["stored"]
    assert client.get("/interactions/u1").json()[-1]["action"] == "like"
    assert client.put("/profile/u1", json={"platforms": ["Nintendo"], "price_range": "20_40",
                                           "games_played": [games[0]["item_id"]]}).json()["stored"]
    assert client.get("/profile/u1").json()["platforms"] == ["Nintendo"]
    r2 = client.post("/recommend", json={"user_id": "u1", "query": "", "k": 5}).json()
    assert item not in [it["item_id"] for it in r2["items"]] or r2["mock"], "a liked game was recommended again"

    # Discovery loop: propose, dislike, the next proposal changes, undo restores the state
    first = client.post("/next", json={"user_id": "u2", "query": "a fun racing game"}).json()["game"]
    client.post("/log_interaction", json={"user_id": "u2", "item_id": first["item_id"], "action": "dislike"})
    second = client.post("/next", json={"user_id": "u2", "query": "a fun racing game"}).json()
    assert second["game"]["item_id"] != first["item_id"], "a disliked game was proposed again"
    undone = client.post("/undo", json={"user_id": "u2"}).json()["undone"]
    assert undone["item_id"] == first["item_id"] and undone["action"] == "dislike"
    again = client.post("/next", json={"user_id": "u2", "query": "a fun racing game"}).json()["game"]
    assert again["item_id"] == first["item_id"], "undo did not restore the previous state"
    print(f"Discovery loop OK: proposed {first['title']!r}, disliked -> {second['game']['title']!r} "
          f"(adjusted={second['adjusted']}), undo -> {again['title']!r} again. Why: {second['game'].get('why')}")
    print("API smoke test passed:", [it["title"] for it in r2["items"][:3]])


if __name__ == "__main__":
    test_all()
