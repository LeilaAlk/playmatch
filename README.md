# PlayMatch

Recommending video games from a player profile and a plain-language request.
RecSys Startup Sprint. Team: Julia (Project Lead), Andrea (MLE-Core), Talia (MLE-Ops), Leila (Data + Systems).

Model served: **two-stage hybrid** (BGE text retrieval of 100 candidates, re-ranked with NLGCL collaborative
embeddings (Xu et al., RecSys 2025) and the player profile). Also implemented and evaluated: a GenSAR-style
generative model (Shi et al., RecSys 2025), which did not beat the baselines (see `results/`).

## Quick start on a clean machine (target: under 10 minutes)

You need Python 3.10+ and Git. The processed data and embeddings are **not** on GitHub (Amazon data):
ask the team for access to the shared Drive folder `playmatch` and download `processed.zip`.

**Windows (Command Prompt)**

```bat
git clone https://github.com/LeilaAlk/playmatch.git
cd playmatch
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put `processed.zip` in the `playmatch` folder, then:

```bat
powershell Expand-Archive processed.zip data
dir data\processed\items_bge.npy
python -m uvicorn api.main:app --port 8000
```

Wait for `Model ready (twostage)` and `Uvicorn running on http://127.0.0.1:8000`. In a **second** Command Prompt:

```bat
cd playmatch
.venv\Scripts\activate
python -m streamlit run app/streamlit_app.py
```

The interface opens at http://localhost:8501 (press Enter if Streamlit asks for an e-mail).
Check: the sidebar shows `model: twostage`; in *Discover*, type a request and click *Start / refresh*.

**Mac / Linux**: same steps with `source .venv/bin/activate` and `unzip processed.zip -d data`.

Notes
- Always start the servers with `python -m uvicorn` / `python -m streamlit`: some antivirus software
  (Avast CyberCapture) blocks the `uvicorn.exe` / `streamlit.exe` launchers.
- The first start downloads the BGE text encoder (about 130 MB) once.
- `python tests/test_api.py` checks every endpoint and prints the model served and the response time.

## Full pipeline (reproduce everything)

| Step | Command | Where | Output |
| --- | --- | --- | --- |
| 1. Download raw data | `python scripts/download_data.py` | laptop | `data/raw/` (3 GB) |
| 2. Clean + split | `python -m src.pipeline --raw data/raw --out data/processed --k 5` | laptop | parquet files, `stats.json` |
| 2b. Display metadata (covers, release date, publisher) | `python scripts/add_metadata.py --raw data/raw --data data/processed` | laptop | new columns in `items.parquet` |
| 3. Profiles + reliability | `python -m src.profile_check --data data/processed` | laptop | `results/profile_check.json` |
| 4. Text embeddings (BGE) | `python -m src.embeddings --data data/processed` | laptop | `items_bge.npy` |
| 5. NLGCL (+ LightGCN ablation) | `python -m src.nlgcl --data data/processed --epochs 50` then add `--no-contrastive` | laptop | `nlgcl_*.npy`, `results/nlgcl_curve.png` |
| 6. GenSAR ids (RQ-VAE) | `python -m src.rqvae --data data/processed` | laptop | `semantic_ids.parquet` |
| 7a. Request sources | `python -m src.queries extract --raw data/raw --data data/processed` | laptop | `query_sources.parquet` |
| 7b. LLM requests | `python -m src.queries generate --data data/processed` | Colab GPU | `queries.parquet` |
| 8. Plan B | `python -m src.twostage --data data/processed --split test` | laptop | `results/twostage_*` |
| 9. GenSAR training | `python -m src.train_gensar --data data/processed --epochs 10` (+ `--no-profile`) | Colab GPU | `models/gensar_best/`, `results/gensar_curve.png` |
| 10. Final evaluation | `python -m src.evaluate_model --data data/processed --split test` | Colab GPU | `results/model_comparison_test.*`, `results/sample_users.md` |
| 11. Share | `powershell Compress-Archive -Force data\processed processed.zip`, put it in the team Drive folder | laptop | `processed.zip` (optional: `scripts/artifacts.py` publishes to a Hugging Face repo) |

After step 7a, zip `data/processed` and put it on the Drive for Colab: `colab_playmatch.ipynb` runs steps 7b, 9 and 10.

NLGCL has one setting to tune on validation: `--cl-weight` (try 0.0005, 0.001, 0.005, keep the best val NDCG@10).

## Discovery loop (one game at a time, with undo)

`POST /next` proposes ONE game for the player (and an explanation: "close to X, which you liked; on Nintendo, a
platform you play on"). The interface asks: I like it / Not for me / Skip / Undo.

- **Like / Not for me** are stored by `/log_interaction` and become part of the player's history (as `<liked>` /
  `<disliked>` for GenSAR). The answered game is never proposed again.
- The next proposal takes the model's top 30 and re-scores them: away from the games the player disliked, toward the
  games he liked (text similarity of BGE vectors).
- **Undo** (`POST /undo`) deletes the last answer from the database: the history goes back to its previous state
  and the previous proposal comes back (rollback).

## No leakage

- `src/pipeline.py`: temporal leave-last-out split; three counters in `stats.json` must be 0.
- The graph (NLGCL), the profiles, the GenSAR training examples and the RQ-VAE ids use training interactions only;
  BGE uses game metadata only.
- Weights of plan B and the best GenSAR epoch are chosen on the validation split; the test split is used once, in step 10.

## Project layout

| Path | What it does |
| --- | --- |
| `src/pipeline.py` | Games-only filter, rating labels, gift keyword flag, k-core, temporal split, cold users |
| `src/profile.py`, `src/profile_check.py` | Player profile from liked games (platforms, price, recency, no gifts) and its reliability check |
| `src/embeddings.py`, `src/text_encoder.py` | BGE embeddings of games and requests |
| `src/nlgcl.py` | LightGCN + NLGCL contrastive loss, training curves |
| `src/rqvae.py` | GenSAR joint identifiers (shared + semantic / collaborative codes) |
| `src/queries.py` | LLM-generated requests (with a title-leak check) |
| `src/gensar.py`, `src/train_gensar.py` | GenSAR input format, training loop with validation curves |
| `src/predict.py`, `src/twostage.py` | Predictors used by the API and the evaluation |
| `src/evaluate.py`, `src/evaluate_model.py` | Metrics, baselines, ablations, innovation metrics, sample users |
| `api/main.py` | FastAPI: `/recommend`, `/next`, `/undo`, `/games`, `/log_interaction`, `/profile` (SQLite) |
| `app/streamlit_app.py` | Interface: discover one by one (like / dislike / undo), search, profile, activity |
| `scripts/` | Data download, display metadata, Hugging Face artifacts |
| `tests/` | Synthetic data generator, API smoke test |

## Test everything without the real data

```bash
python tests/make_synthetic_data.py --out data/raw_synthetic
python -m src.pipeline --raw data/raw_synthetic --out data/processed_synthetic --k 3
python -m src.embeddings --data data/processed_synthetic --offline
python -m src.nlgcl --data data/processed_synthetic --epochs 20
python -m src.rqvae --data data/processed_synthetic
python -m src.queries extract --raw data/raw_synthetic --data data/processed_synthetic --test-sample 300 --valid-sample 200 --train-sample 1000
python -m src.queries generate --data data/processed_synthetic --template
python -m src.train_gensar --data data/processed_synthetic --sample 200 --epochs 2 --offline
python -m src.evaluate_model --data data/processed_synthetic --split test
```
