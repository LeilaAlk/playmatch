"""Smoke test of the API (run: python -m pytest tests/test_api.py -q, or python tests/test_api.py)."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("PLAYMATCH_DB", str(Path(tempfile.mkdtemp()) / "test.db"))

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

client = TestClient(app)


def test_all() -> None:
    assert client.get("/health").json()["status"] == "ok"
    r = client.post("/recommend", json={"user_id": "u1", "query": "a relaxing game", "platform": "Nintendo", "k": 5})
    body = r.json()
    assert r.status_code == 200 and body["mock"] is True and len(body["items"]) > 0
    assert all(it["platform"] == "Nintendo" for it in body["items"])
    item = body["items"][0]["item_id"]
    assert client.post("/log_interaction", json={"user_id": "u1", "item_id": item, "action": "like",
                                                 "query": "a relaxing game"}).json()["stored"]
    assert client.get("/interactions/u1").json()[0]["action"] == "like"
    assert client.put("/profile/u1", json={"platforms": ["Nintendo"], "price_range": "20_40"}).json()["stored"]
    assert client.get("/profile/u1").json()["platforms"] == ["Nintendo"]
    print("API smoke test passed:", body["items"][:2])


if __name__ == "__main__":
    test_all()
