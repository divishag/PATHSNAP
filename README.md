# PathSnap: Map-Matching and Route Reconstruction

PES University, UE24CS352A Machine Learning hackathon.

Noisy GPS observations are matched to road segments by a candidate classifier,
then decoded into a connected route with an HMM / Viterbi decoder. A FastAPI
server (Laptop 1) runs the full pipeline; a Streamlit UI (Laptop 2) selects a
trip and displays the returned route and metrics.

## Repository layout

| Folder | Contents |
|---|---|
| `src/pathsnap/data` | trip loading, validation, EDA, reference-route alignment (labels only) |
| `src/pathsnap/features` | candidate generation, v2 feature engineering, labels, quality flags |
| `src/pathsnap/models` | model factory, grouped CV, evaluation metrics |
| `src/pathsnap/optimization` | Gaussian-process Bayesian optimisation |
| `src/pathsnap/decoding` | HMM emissions, network transitions, Viterbi |
| `src/pathsnap/api` | FastAPI backend (`main.py`) |
| `src/pathsnap/visualization` | Folium maps and plots |
| `frontend/` | Streamlit evaluator UI (Laptop 2) |
| `scripts/` | runnable pipeline steps (below) |
| `configs/` | model, decoder and API settings |
| `results/` | EDA, experiments, tuning, comparison, end-to-end metrics, maps |
| `artifacts/` | trained models (`final/model.joblib` is served by the API) |
| `reports/` | report material, incl. `feature_engineering_v2.md` |

## Setup (both laptops)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Place the raw data in `Student folder/train/` and `Student folder/student_test/`
(not committed).

## Reproducing the results (Laptop 1, repository root)

```bash
export PYTHONPATH=src

# 1. EDA
python scripts/explore_dataset.py
python scripts/write_eda_report.py

# 2. Candidates, features and labels  -> data/processed/
python scripts/build_features.py --partition train
python scripts/build_features.py --partition student_test

# 3. Five models, 5-fold GroupKFold by trip  -> results/experiments/laptop1/, artifacts/laptop1/
python scripts/train_logistic_regression.py
python scripts/train_weighted_knn.py
python scripts/train_random_forest.py
python scripts/train_hist_gradient_boosting.py
python scripts/train_mlp.py

# 4. HMM/Viterbi end-to-end on the 10 student-test trips  -> results/end_to_end/
python scripts/run_end_to_end.py --model hist_gradient_boosting --limit 10
python scripts/run_end_to_end.py --model random_forest --limit 10

# 5. Bayesian hyperparameter optimisation of the selected model  -> results/tuning/
python scripts/run_bayesian_tuning.py --model hist_gradient_boosting --trials 10 --folds 3

# 6. Comparison table, confusion matrices, classifier-vs-HMM, Folium maps
python scripts/make_comparison.py   # -> results/comparison/, results/maps/

# Tests
python -m pytest -q tests
```

Seeds are fixed at 42. The evaluator-testing partition is never used in any of
these steps.

## Results (student-test = 10 trips; CV = 80 train trips)

| Model | CV Top-1 | Student-test Top-1 | Test Macro F1 | Log loss |
|---|---|---|---|---|
| Logistic Regression | 0.911 ± 0.027 | 0.898 | 0.896 | 0.136 |
| Weighted KNN | 0.901 ± 0.027 | 0.889 | 0.929 | 0.200 |
| Random Forest | 0.925 ± 0.024 | 0.917 | 0.944 | 0.075 |
| **HistGradientBoosting (selected)** | 0.924 ± 0.024 | **0.919** | 0.932 | 0.093 |
| MLP | 0.920 ± 0.024 | 0.906 | 0.922 | 0.106 |

With HMM/Viterbi: HGB 0.912 arc accuracy, route continuity 0.797 → 0.888.
Full tables: `results/comparison/`. Tuning: `results/tuning/`.

## Deployment (final evaluation)

**Laptop 1, API.** Both laptops on the same network; allow port 8000 through
the firewall.

```bash
source .venv/bin/activate
export PYTHONPATH=src
export PATHSNAP_TRIP_ROOT="/path/to/evaluator_test"     # the 10 reserved trips
uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
```

Find Laptop 1's IP (`ipconfig getifaddr en0` on macOS, `ipconfig` on Windows).
The terminal shows one log line per `POST /predict` (trip, accuracy, continuity,
time). Show this log to the evaluator.

**Laptop 2, UI.**

```bash
source .venv/bin/activate
python -m pip install -r frontend/requirements.txt
cd frontend
python -m streamlit run app.py
```

Open http://localhost:8501, enter `http://<laptop-1-ip>:8000`, click
**Connect / refresh trips**, select a trip, click **Reconstruct route**.

Endpoints: `GET /trips`, `POST /predict {"trip_id": "..."}`.

## Leakage and integrity

- Prediction path (`/predict`): raw `.track`, `.nodes`, `.arcs` only →
  `generate_candidates` → `add_features` → model → HMM. Reference routes are
  not loaded.
- Accuracy is computed afterwards, server-side, from the trip's `.route`
  file, and only the number is returned. Labels never reach the UI.
- No hardcoded predictions; all preprocessing is fitted on training trips only.
