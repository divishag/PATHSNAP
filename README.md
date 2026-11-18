# PathSnap: Distributed Map-Matching and Route Reconstruction

PES University — Machine Learning Hackathon

## Objective

Reconstruct realistic vehicle routes from noisy GPS observations
using machine learning and HMM/Viterbi decoding.

## Dataset

The dataset contains:
- 80 training trips
- 10 student-testing trips
- 10 evaluator-testing trips reserved for final evaluation

Each trip may contain:
- .track: GPS longitude, latitude and timestamp
- .nodes: road-network node coordinates
- .arcs: road-network connections
- .route: reference road-segment sequence

The raw dataset is stored locally in:
Student folder/train/
Student folder/student_test/

## Project Components

- Dataset exploration and preprocessing
- Candidate road-segment generation
- Feature engineering
- Five-model experimentation and comparison
- Bayesian hyperparameter optimisation
- HMM/Viterbi route reconstruction
- Folium geographical visualisation
- FastAPI backend
- Streamlit frontend
- Technical report and experiment records

## Setup

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

## Development

Source code is located in src/pathsnap/.

Executable scripts are located in scripts/.

## Evaluation

### Laptop 2 frontend

The evaluator UI is implemented in `frontend/app.py`. Install its dependencies
with `python -m pip install -r frontend/requirements.txt`, then run
`python -m streamlit run app.py` from the `frontend` directory.
See `frontend/README.md` for the proposed API contract, network setup, and
integration checklist. The UI requires real backend endpoints; backend
integration is pending.

The backend runs on Laptop 1.
The frontend runs on Laptop 2.

Evaluator-testing ground truth must remain server-side and
must never be used to generate predictions.

## Status

Laptop 2's independent implementation is complete: real-data EDA, evaluator UI
with dark/light modes, reusable maps, five-family model workflow, optional grouped
calibration, Bayesian tuning, artifact export code and regression checks.

The current models are Logistic Regression, distance-weighted kNN, Random Forest,
HistGradientBoosting and a shallow MLP. Selection and tuning use grouped-CV
observation Top-1 accuracy; candidate Macro F1 and Log Loss are also recorded.
Actual model fitting waits for Laptop 1's candidate features and aligned labels.
No real model scores or trained weights have been produced.

Read [Laptop 2 runbook](docs/laptop2_runbook.md),
[shared integration contract](docs/laptop2_integration.md), and
[task ownership and handoff status](docs/task_ownership.md).
Completed report sections are in [EDA/UI progress](reports/EDA_and_UI_progress.md).

Use `requirements-laptop2.txt` for this part of the project. It does not require
Optuna; Bayesian search uses a Gaussian-process surrogate and expected improvement.
Run from the repository root:

```powershell
.\.venv\Scripts\python.exe tests/test_laptop2.py
.\.venv\Scripts\python.exe tests/test_frontend_laptop2.py
.\.venv\Scripts\python.exe scripts/write_eda_report.py
.\.venv\Scripts\python.exe scripts/package_laptop2.py
```

The handoff archive is an overlay for review and merge into the shared project,
not a replacement for Laptop 1's implementation. It excludes raw data, virtual
environments, Git metadata and synthetic experiment artifacts.
