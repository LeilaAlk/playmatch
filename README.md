# PlayMatch

Recommending video games from a player profile and a plain-language request.
RecSys Startup Sprint. Team: Julia (Project Lead), Andrea (MLE-Core), Talia (MLE-Ops), Leila (Data + Systems).

Status: **Week 2 (first build)**. The data pipeline, evaluation harness, model skeleton and API exist.
The model is not trained yet and the API returns mock recommendations.

## Run it locally

```bash
# 1. clone and create an environment (Python 3.10+)
git clone <repo-url> && cd playmatch
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. download the two raw files into data/raw/ (keep them as .gz)
#    https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz
#    https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/meta_categories/meta_Video_Games.jsonl.gz

# 3. data pipeline (Q1): ~10-20 min on a laptop, writes data/processed/ and data/processed/stats.json
python -m src.pipeline --raw data/raw --out data/processed --k 5

# 4. player profiles (first version)
python -m src.profile --data data/processed

# 5. evaluation harness (Q3): metric self-test, then random and most-popular baselines
python -m src.evaluate --selftest
python -m src.evaluate --data data/processed --split test

# 6. model skeleton (Q2): forward pass, training steps, constrained beam search
python -m src.model_skeleton --data data/processed --users 64

# 7. API (Q4) - then open http://localhost:8000/docs
uvicorn api.main:app --reload --port 8000
python tests/test_api.py      # automatic check of every endpoint
```

No real data yet? Everything also runs on a small fake dataset with the same schema:

```bash
python tests/make_synthetic_data.py --out data/raw_synthetic
python -m src.pipeline --raw data/raw_synthetic --out data/processed_synthetic --k 3
python -m src.evaluate --data data/processed_synthetic
python -m src.model_skeleton --data data/processed_synthetic --offline
```

## Project layout

| Path | What it does | Owner |
| --- | --- | --- |
| `src/pipeline.py` | Games-only filter, rating labels (liked / played / disliked), gift keyword flag, k-core, temporal leave-last-out split, 10% cold users, leakage checks in `stats.json` | Leila |
| `src/profile.py` | Player profile: time-weighted platforms, price range, counts; gifts excluded | Leila |
| `src/evaluate.py` | NDCG@10, Recall@20, coverage, popularity; random and most-popular baselines; warm vs cold users | Julia |
| `src/model_skeleton.py` | GenSAR-style T5 input with profile tokens, forward/backward pass, constrained beam search with platform filter | Andrea, Talia |
| `api/main.py` | FastAPI: `/recommend` (mock), `/log_interaction` and `/profile` (SQLite) | Leila |

## Next (Week 3)

RQ-VAE identifiers from BGE and NLGCL embeddings, LLM-generated requests, full GenSAR training,
genres and franchises in the profile, real `/recommend`, Streamlit interface.
