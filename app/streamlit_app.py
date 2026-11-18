"""PlayMatch - web interface (Week 3, Q5). The interface only talks to the API.

Run (API first):
  uvicorn api.main:app --port 8000
  streamlit run app/streamlit_app.py
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

API = os.environ.get("PLAYMATCH_API", "http://localhost:8000")
PLATFORMS = ["PlayStation", "Xbox", "Nintendo", "PC"]
PRICES = {"Not sure": "unknown", "Under $20": "under_20", "$20 to 40": "20_40", "$40 to 60": "40_60", "Over $60": "over_60"}

st.set_page_config(page_title="PlayMatch", page_icon="🎮", layout="wide")
st.title("🎮 PlayMatch")
st.caption("Tell us what you feel like playing. We match it with who you are as a player.")


def call(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=60, **kw)
        return r.json() if r.ok else {"error": r.json().get("detail", r.text)}
    except requests.RequestException as exc:
        return {"error": str(exc)}


def meta_line(g: dict) -> str:
    bits = [g["platform"]]
    if g.get("price") is not None:
        bits.append(f"${g['price']:.2f}")
    if g.get("release"):
        bits.append(str(g["release"]))
    if g.get("publisher"):
        bits.append(str(g["publisher"])[:30])
    if g.get("average_rating"):
        bits.append(f"★ {g['average_rating']:.1f}")
    return " · ".join(bits)


def log(user_id: str, item_id: str, action: str, query: str = "") -> None:
    call("POST", "/log_interaction", json={"user_id": user_id, "item_id": item_id, "action": action, "query": query})


with st.sidebar:
    user_id = st.text_input("Your user id", value=st.session_state.get("user_id", "demo_user"))
    if user_id != st.session_state.get("user_id"):
        st.session_state.user_id = user_id
        st.session_state.pop("current", None)
    st.caption("Tip: `player_42` loads player 42 of the dataset (his real history).")
    health = call("GET", "/health")
    if "error" in health:
        st.error(f"API not reachable at {API}. Start it with: uvicorn api.main:app --port 8000")
        st.stop()
    st.success(f"API online - model: {health['model']}")
    st.subheader("Limits (optional)")
    platform = st.selectbox("Only on", ["Any platform"] + PLATFORMS)
    max_price = st.number_input("Max price ($, 0 = no limit)", min_value=0, value=0, step=5)
    limits = {"platform": None if platform == "Any platform" else platform, "max_price": max_price or None}

tab_discover, tab_search, tab_profile, tab_history = st.tabs(
    ["Discover (one by one)", "Search", "My player profile", "My activity"])

# ---------------------------------------------------------------- discover
with tab_discover:
    st.write("We propose one game. Tell us if you like it: every answer adjusts the next proposal. "
             "**Undo** cancels your last answer.")
    dq = st.text_input("What do you feel like? (optional)", key="discover_query",
                       placeholder="something relaxing after work, no online play")

    def fetch_next():
        res = call("POST", "/next", json={"user_id": user_id, "query": dq, **limits})
        st.session_state.current = res

    if "current" not in st.session_state or st.button("Start / refresh"):
        fetch_next()
    res = st.session_state.current
    if "error" in res:
        st.warning(res["error"])
    else:
        g = res["game"]
        left, right = st.columns([1, 2])
        if g.get("image_url"):
            left.image(g["image_url"], width=220)
        right.subheader(g["title"])
        right.write(meta_line(g))
        if g.get("why"):
            right.info(f"Why this game: {g['why']}")
        right.caption(f"Model: {res['model']}{' (MOCK)' if res['mock'] else ''} · "
                      f"{'adjusted to your answers' if res['adjusted'] else 'first proposal'} · "
                      f"liked {res['answers_so_far']['liked']}, disliked {res['answers_so_far']['disliked']} · "
                      f"{res['latency_ms']} ms")
        b1, b2, b3, b4 = right.columns(4)
        if b1.button("❤️ I like it", type="primary"):
            log(user_id, g["item_id"], "like", dq)
            st.toast(f"Liked: {g['title'][:40]}")
            fetch_next()
            st.rerun()
        if b2.button("👎 Not for me"):
            log(user_id, g["item_id"], "dislike", dq)
            st.toast("Got it, adjusting...")
            fetch_next()
            st.rerun()
        if b3.button("⏭️ Skip"):
            log(user_id, g["item_id"], "not_interested", dq)
            fetch_next()
            st.rerun()
        if b4.button("↩️ Undo"):
            u = call("POST", "/undo", json={"user_id": user_id})
            st.toast("Last answer cancelled" if "undone" in u else u.get("error", "nothing to undo"))
            fetch_next()
            st.rerun()

# ------------------------------------------------------------------ search
with tab_search:
    query = st.text_area("What are you looking for?",
                         placeholder="a relaxing game to play on the couch with my boyfriend, no shooting")
    k = st.slider("Number of games", 5, 20, 10)
    if st.button("Recommend", type="primary"):
        t0 = time.time()
        st.session_state.results = call("POST", "/recommend", json={"user_id": user_id, "query": query, "k": k, **limits})
        st.session_state.roundtrip = 1000 * (time.time() - t0)
    res = st.session_state.get("results")
    if res:
        if "error" in res:
            st.error(res["error"])
        else:
            st.caption(f"Model: {res['model']}{' (MOCK)' if res['mock'] else ''} - server {res['latency_ms']} ms, "
                       f"round trip {st.session_state.roundtrip:.0f} ms")
            for it in res["items"]:
                img, a, b, c = st.columns([1, 6, 1, 1])
                if it.get("image_url"):
                    img.image(it["image_url"], width=60)
                a.markdown(f"**{it['rank']}. {it['title']}**  \n{meta_line(it)}")
                if b.button("👍", key=f"like_{it['item_id']}_{it['rank']}"):
                    log(user_id, it["item_id"], "like", res["query"])
                    st.toast(f"Saved: like - {it['title'][:40]}")
                if c.button("👎", key=f"dislike_{it['item_id']}_{it['rank']}"):
                    log(user_id, it["item_id"], "dislike", res["query"])
                    st.toast(f"Saved: dislike - {it['title'][:40]}")

# ----------------------------------------------------------------- profile
with tab_profile:
    current = call("GET", f"/profile/{user_id}")
    current = {} if "error" in current else current
    st.write("We guess your profile from the games you liked. You can correct it here.")
    plats = st.multiselect("I play on", PLATFORMS, default=current.get("platforms", []))
    inv = {v: k for k, v in PRICES.items()}
    price = st.selectbox("I usually pay", list(PRICES),
                         index=list(PRICES).index(inv.get(current.get("price_range"), "Not sure")))
    gender = st.selectbox("Gender (optional, not used by the model)", ["Prefer not to say", "girl", "boy", "other"])
    st.write("Games I played and liked")
    search = st.text_input("Search a game title")
    found = call("GET", "/games", params={"q": search, "limit": 15}) if search else []
    played = st.session_state.setdefault("played", {g: g for g in current.get("games_played", [])})
    for g in found if isinstance(found, list) else []:
        if st.checkbox(f"{g['title']} ({g['platform']})", key=f"pick_{g['item_id']}", value=g["item_id"] in played):
            played[g["item_id"]] = g["title"]
        else:
            played.pop(g["item_id"], None)
    if played:
        st.caption("Selected: " + "; ".join(str(t)[:40] for t in played.values()))
    if st.button("Save my profile", type="primary"):
        body = {"platforms": plats, "price_range": PRICES[price], "games_played": list(played),
                "gender": None if gender == "Prefer not to say" else gender}
        r = call("PUT", f"/profile/{user_id}", json=body)
        st.success("Profile saved") if "stored" in r else st.error(r)

# ----------------------------------------------------------------- history
with tab_history:
    rows = call("GET", f"/interactions/{user_id}")
    if isinstance(rows, list) and rows:
        df = pd.DataFrame(rows)
        df["time"] = pd.to_datetime(df.ts, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(df[["id", "time", "action", "item_id", "query"]], width="stretch")
        st.caption("These rows come straight from the SQLite database behind /log_interaction.")
    else:
        st.write("No activity yet: answer a proposal in the first tab.")
