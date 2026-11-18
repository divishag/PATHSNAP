# Laptop 2 implementation and proposed integration

Status: proposed handoff for agreement between Laptop 1 and Laptop 2.

The existing shared_contract.md is preserved as the team's authoritative proposed
contract. This document describes Laptop 2's independently built interfaces.
Integration is pending agreement/adapters for these differences:

| Shared v0.2 field | Current Laptop 2 input | Required decision |
| --- | --- | --- |
| candidate_arc_id | arc_id plus candidate_id | Map arc identity; preserve original IDs |
| label | is_match | Map only aligned valid observation labels |
| alignment_status | inventory label_valid | Derive validity from accepted alignment status |
| GET /trips: trip_ids | GET /trips: trips objects | Agree response shape or add client adapter |

No changes to Laptop 1's modules are included in this handoff.
The API section matches the implemented Streamlit client. Feature, label,
and model interfaces below are proposals pending agreement with Laptop 1.
Laptop 2 training, evaluation, plotting and Bayesian-search code is implemented;
real model runs require the candidate/observation tables described below.

## Ownership and boundaries

| Laptop 1 provides | Laptop 2 provides |
| --- | --- |
| Raw-file loading, validation, reference-route alignment | Dataset EDA and statistics |
| Candidate generation, features, labels | Five-model training, evaluation and comparison |
| HMM/Viterbi decoder | Bayesian tuning and exported trained model |
| FastAPI server and server-side scoring | Streamlit UI, Folium visualisations and plots |

Shared: raw data, dependency list, project README, this contract and report.
Do not change another owner's interface without updating this contract together.

## 1. API: Laptop 1 to the implemented Laptop 2 UI

All requests use JSON. Laptop 2 sends requests from its Streamlit Python
process; browser CORS is not required for this implementation.

### GET /trips

Return all ten reserved evaluation trips with unique nonempty string IDs:

```json
{
  "trips": [
    {"trip_id": "real-trip-id", "label": "Evaluation trip 1"}
  ]
}
```

The single item above illustrates the schema, not the actual trip catalogue.
The label is optional. Load real IDs from the backend's dataset catalogue.

### POST /predict

Request:

```json
{"trip_id": "real-trip-id"}
```

Successful response (illustrative geometry only):

```json
{
  "trip_id": "real-trip-id",
  "gps_coordinates": [[77.59, 12.97], [77.591, 12.971]],
  "predicted_route": [[77.5901, 12.9701], [77.5911, 12.9711]],
  "road_segments": [[[77.59, 12.97], [77.591, 12.971]]],
  "metrics": {
    "top1_accuracy": null,
    "macro_f1": null,
    "route_continuity": null,
    "inference_time_ms": null
  }
}
```

| Field | Meaning |
| --- | --- |
| trip_id | Exact string ID from the request |
| gps_coordinates | Ordered original GPS positions; at least one pair |
| predicted_route | Ordered reconstructed road geometry, including connecting road segments; at least two pairs |
| road_segments | Optional list of road geometries for the network layer |
| metrics | Optional aggregate evaluation results, calculated server-side |

Every coordinate pair is `[longitude, latitude]`, finite and within geographic
bounds. Never return latitude first. Arc IDs alone cannot render a route.
Scores are fractions from 0 to 1, not percentages. Inference time is milliseconds
and must be nonnegative. Missing or null metrics display as “Not supplied”.
Return only public fields; do not send reference routes or per-observation
reference labels to the UI. Reference comparison remains server-side after inference.
Define the Top-1 denominator, Macro F1 averaging scheme and route-continuity formula
before reporting scores. The current UI does not assume those definitions.

Unknown trip: HTTP 404. Invalid request: HTTP 422. Server/model unavailable:
appropriate HTTP 5xx. Return JSON error details for backend diagnostics; the
UI currently shows the HTTP status and asks the evaluator to check server logs.
Client timeout: 5 seconds to connect, 120 seconds to read. Do not return a
job ID or asynchronous job response without extending the UI contract first.

## 2. Feature-table handoff: Laptop 1 to Laptop 2 (proposed)

Provide separate training and student-test candidate tables, plus a manifest
that defines the schema, feature order, units, label method and generation settings.
Agree final filenames before implementing the training scripts.

One row represents one candidate road segment for one GPS observation:

| Column | Requirement |
| --- | --- |
| trip_id | Stable string; cross-validation group |
| observation_id | Original observation index; never renumber silently after cleaning |
| timestamp_s | Observation timestamp in seconds |
| candidate_id | Unique within this observation; stable candidate ordering |
| arc_id | Original trip-local arc row index; never a global class label |
| feature columns | Named numeric predictors, with documented units and missing-value policy |
| is_match | Proposed binary candidate label: 1 for the aligned reference arc, 0 otherwise |
| label_valid | Whether reference alignment supports a usable observation label |

Candidate identity is `(trip_id, observation_id, candidate_id)`. Include an
observation inventory even for observations with zero candidates, and retain
the aligned reference arc in a separate permitted evaluation table. This is
needed to measure candidate misses without dropping failed observations.

Reference `.route` rows are chronological traversed arc IDs, not pointwise GPS
labels. Laptop 1 must implement and document route alignment before candidate
labels can be used. Record ambiguous/unlabelled observations and the chosen policy.
Candidate misses must remain visible in observation-level evaluation, even
when no positive candidate row is available for supervised training.

IDs, timestamps, partition indicators and reference-derived fields are metadata
unless explicitly justified as predictors. Never include label_valid, is_match,
aligned reference IDs, future reference information or reference-derived distance
as inference features. Inference must work without reference labels.

Use whole-trip grouping for cross-validation on the 80 training trips. Fit
imputation/scaling/calibration inside each training fold. Use the ten student-test
trips for development comparisons. Evaluator data is excluded from model choice,
feature choice, tuning, calibration and thresholds.

## 3. Model handoff: Laptop 2 to Laptop 1 (proposed)

The configured five families are Logistic Regression, distance-weighted kNN,
Random Forest, HistGradientBoosting and a shallow MLP. Model selection and Bayesian
tuning maximise mean grouped-CV observation Top-1 accuracy, retaining candidate
Macro F1, binary Log Loss and Brier score in the comparison. Calibration is optional
and uses inner trip-grouped splits; weighted models are not assumed calibrated.

Export a model bundle in `artifacts/` containing:

- `model.joblib`: fitted preprocessing and classifier pipeline.
- `model_manifest.json`: schema version, exact ordered feature names, feature
  units, positive class value, class ordering, model family/configuration,
  Python/package versions, random seed, training split and artifact version.
- Experiment metrics and tuned configuration, linked to reproducible scripts.

Proposed scoring interface: feed a table with the manifest's feature columns
to `predict_proba`, then explicitly locate positive class `1` using `classes_`.
Do not assume it is always probability column 1. Scoring must preserve row
order and the candidate-to-arc mapping. No model is exported at this stage.

For a binary candidate classifier, `P(is_match=1)` is an individual score,
not automatically a distribution over the observation's candidate set.
Laptop 1 and Laptop 2 must agree and document candidate-score normalization,
probability clipping and any calibration before using scores as HMM emissions.
Dividing all candidate scores at a timestep by a common positive constant does
not change the optimal Viterbi sequence under the usual fixed-length product
objective. Such normalisation is not probability calibration or a derivation
of a generative emission likelihood. Document classifier-derived scores as
emission potentials unless an explicit likelihood model is justified.
Compare classifier-only argmax predictions with HMM/Viterbi results using the
same candidate sets and observation eligibility rules.

Laptop 1 owns graph transitions, disconnected-state handling, GPS gap policy,
decoder configuration and reconstruction of connected geometry. Laptop 2 owns
model preprocessing and score generation. Both record the boundary assumptions.

## 4. Run and integrate

Laptop 1: implement the API, load the trained bundle once at startup, bind
the server to `0.0.0.0:8000`, and log each received trip request and outcome.
Keep reference evaluation separate from prediction inputs.

Laptop 2, from repository root:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
cd frontend
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Enter `http://<LAPTOP_1_LAN_IP>:8000` in Backend
connection, connect, select a trip, and reconstruct its route. Both machines
must be reachable on the local network with the API port permitted. Map tiles
and Folium browser assets also require internet access in the current setup.

## 5. Acceptance checklist

- [ ] Both owners agree the feature/label schema and model scoring interface.
- [ ] Model preprocessing produces identical features in training and inference.
- [ ] Catalogue contains ten real reserved evaluation trips.
- [ ] Every trip runs from Laptop 2 with corresponding request logs on Laptop 1.
- [ ] Geometry uses correct coordinate order and draws the actual reconstructed route.
- [ ] Metrics match server-side calculations and documented denominators.
- [ ] Reference labels remain server-side and outside inference features.
- [ ] Invalid trip, unreachable server, timeout and malformed response are handled.
- [ ] Changing API address or trip clears old results; theme changes preserve them.
- [ ] Report includes real UI screenshots, maps, experiment results and API evidence.

Current validation: UI startup, trip loading, prediction rendering, clearing old
results and theme toggling have been checked using simulated API responses.
The light theme was also visually inspected. Real dataset EDA has run on all
80 training and 10 student-test trips; results are in results/eda. A separate
synthetic workflow verifies all five models, Bayesian acquisition, plots and
artifact reload. Real model experiments and backend integration remain pending.
See laptop2_runbook.md for exact commands and metric definitions.
The EDA/UI write-up is reports/EDA_and_UI_progress.md. Genuine desktop/mobile
screenshots are under reports/screenshots; the independent checks are listed in
docs/task_ownership.md. Backend acceptance boxes remain pending real integration.
