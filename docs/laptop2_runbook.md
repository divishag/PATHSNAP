# Laptop 2 runbook

Commands run from the repository root unless stated otherwise.

## Environment and EDA

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-laptop2.txt
.\.venv\Scripts\python.exe scripts/explore_dataset.py --data-root "Student folder"
```

EDA outputs are under `results/eda/`: inventory.csv, partition_statistics.csv,
summary.json, methodology.md, representative Folium HTML maps and PNG plots.
Raw files remain unchanged. Maps show the road network within each trip bounding
box plus a 0.005-degree display margin. Network statistics use the full graph.
Distances use haversine metres; angular distances are not treated as metres.
Reference-route layers appear only in development maps, not the evaluator UI.

Generate a specific development map:

```powershell
.\.venv\Scripts\python.exe scripts/generate_maps.py --track "Student folder/train/00000001/00000001.track" --output results/maps/training_example.html
```

## Model workflow after Laptop 1 handoff

Laptop 1 must supply aligned reference labels, generated candidate features and
observation inventories according to laptop2_integration.md. Configure the four
input CSV paths in configs/models.yaml. Set `features` to the exact numeric
predictors and fill `feature_units`. The empty feature list deliberately stops
training before accidental label or ID leakage. Train/student-test trip IDs
must be globally unique; preserve a mapping to raw IDs if prefixes are needed.
The provided configuration expects 80 training and 10 student-test trips.

```powershell
.\.venv\Scripts\python.exe scripts/train_models.py --config configs/models.yaml
.\.venv\Scripts\python.exe scripts/compare_models.py
.\.venv\Scripts\python.exe scripts/tune_model.py --config configs/models.yaml --trials 30
```

Five models: Logistic Regression, distance-weighted kNN, Random Forest,
HistGradientBoosting and a shallow MLP. Imputation/scaling are fitted
inside whole-trip CV folds. Boosting's internal early stopping is disabled to
avoid hidden row-level validation. Each experiment records configurations,
train/test metrics, fold membership, timing and seeds.

Selection maximises mean grouped-CV observation Top-1 accuracy, breaking ties by
lower Top-1 fold standard deviation and then higher candidate Macro F1.
Bayesian tuning uses scikit-learn GaussianProcessRegressor
with expected improvement over 1,024 random candidate search points per trial.
It starts with five random trials; use at least six trials for Bayesian acquisition.
Default: 30. Bounds are in optimization/bayesian.py; log bounds are base-10
exponents and integer parameters are rounded. Trials, search ranges, fold
metrics, timings and tuned/untuned comparisons are saved under experiments/tuning.
Student-test data is never used in the search objective. An untuned baseline
is exported if tuning lowers CV Top-1. Final refit uses training data only.
This implementation does not require Optuna.

The initial linear/forest/boosting models use class weights. CandidateMLP computes
balanced sample weights using only its current fitting fold (scikit-learn >=1.7).
kNN distance weighting is not class balancing; its ranking, minority performance
and latency must be measured. The search can compare weighted and unweighted
settings for the other models. kNN uses algorithm=auto; tune neighbours, metric
power and leaf size. No candidate count or runtime is assumed from raw GPS counts.

Set calibration_folds >=2 to compare inner trip-grouped sigmoid calibration.
The default 0 keeps raw scores. Each calibration estimator includes its own
preprocessing. All inner folds must contain both classes, and enough training
trips are needed inside every outer fold. Use separate experiment directories
for raw and calibrated configurations. Compare Top-1, Log Loss and Brier score;
weighted objectives do not guarantee naturally calibrated probabilities.

MLP internal row-based validation is disabled. The epoch budget is tuned through
grouped CV, and its actual training loss_curve_ is saved and plotted. No validation
loss curve is invented; a dedicated grouped epoch-wise validation loop would be
additional future work. Convergence warnings indicate that the budget needs review.

Exports: artifacts/model.joblib and artifacts/model_manifest.json, recording
features, class order, seed, versions, configuration and input/model SHA-256.
The backend passes ordered manifest features and locates positive class 1 via
classes_; scores remain unnormalised across candidates. Only load trusted team
joblib files. HMM score normalisation/calibration must be agreed with Laptop 1.

## Metrics and graphs

- Candidate binary Macro F1/precision/recall average labels [0,1], assigning zero
  to undefined class scores. Predictions use a fixed 0.5 threshold. Candidate
  Log Loss uses clipped binary probabilities. Brier score is mean squared
  probability error. Invalid reference labels are excluded.
- Observation Top-1 selects the candidate with the greatest probability, ties
  following input row order. The denominator includes every label-valid inventory
  observation, including ones with zero candidates. Missing reference candidates
  count as errors. Candidate Miss Rate includes those same zero-candidate cases.
- Unlabelled observations are counted separately. Candidate metrics and observation
  metrics have different denominators; report them using these explicit names.
- Inference time is probability scoring over the supplied partition, including
  fitted preprocessing, excluding file loading, HMM decoding and network latency.
- Route continuity and decoder sequence comparisons require Laptop 1's graph and
  decoder. They remain N/A with an explanation until integration.
- Plots use measured experiment outputs: CV Top-1/F1 with fold standard deviation,
  student-test Top-1/F1/Log Loss, latency and confusion matrices. No artificial
  epoch loss curves are produced. MLP training curves use its actual history.

## Checks and UI

```powershell
.\.venv\Scripts\python.exe tests/test_laptop2.py
.\.venv\Scripts\python.exe tests/test_frontend_laptop2.py
cd frontend
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Tests use temporary synthetic fixtures: candidate-miss denominators, label/schema
validation, trip separation, GP acquisition, all five models, plotting, tuning,
artifact export and reload. Synthetic metrics are never mixed into real results.
Use the UI dark/light toggle and Backend connection to connect to Laptop 1.
Theme choice persists within the session; new sessions start dark.

To regenerate the written EDA/UI sections and source hashes:

```powershell
.\.venv\Scripts\python.exe scripts/write_eda_report.py
```

The report draft is reports/EDA_and_UI_progress.md. The final consolidated PDF
must wait for real model and decoder results. The decorative UI hero is never
presented as dataset evidence.

Remaining dependent execution: real model fitting, tuned artifact delivery,
classifier-versus-decoder comparison, ten-trip network integration, API evidence
and final report conclusions. The PDF needs actual measured model outcomes and
Laptop 1's decoder analysis; the EDA evidence is already available separately.
