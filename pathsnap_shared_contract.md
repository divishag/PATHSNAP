# PathSnap — Shared Data and Interface Contract (v0.2)

**Status:** Proposed team contract. Both teammates must approve before implementing. This is a development design, not an additional university requirement.

## 1. What the supplied guides establish

- Training: 80 trips; student-testing: 10 trips; evaluator-testing: 10 reserved trips. Keep trip boundaries intact; use group-aware cross-validation for training.
- `.track`: headerless whitespace-separated `longitude latitude timestamp_seconds` (nominally 1 Hz).
- `.nodes`: headerless whitespace-separated `longitude latitude`; **node_id = 0-based row number**.
- `.arcs`: headerless whitespace-separated integer `from_node_id to_node_id`; **arc_id = 0-based row number**.
- `.route`: headerless single integer `arc_id` per row, a chronological reference route *where supplied*.
- **Do not assume a `.track` row corresponds to a `.route` row.** Reference routes are not independently guaranteed error-free.
- Candidate probabilities must preserve their correspondence to road segments; HMM/Viterbi must account for topology, GPS gaps and invalid transitions.
- Reference labels may be used for permitted training and evaluation, but never as prediction features. Evaluator reference labels must stay server-side.

## 2. Raw loaders (agreed proposed Python schema)

All IDs are scoped to one trip. `trip_id` is the folder name as a **string** (preserve leading zeros).

`gps_observations`: `trip_id:str`, `observation_id:int64` (0-based row index), `longitude:float64`, `latitude:float64`, `timestamp_s:float64`.

`road_nodes`: `trip_id:str`, `node_id:int64` (0-based row index), `longitude:float64`, `latitude:float64`.

`road_arcs`: `trip_id:str`, `arc_id:int64` (0-based row index), `from_node_id:int64`, `to_node_id:int64`.

`reference_route`: `trip_id:str`, `route_step_id:int64` (0-based row index), `arc_id:int64`.

Load with whitespace parsing and explicit column names. Validate finite coordinates, index bounds, time ordering, and arc references; report anomalies rather than silently changing IDs. Preserve original rows and IDs.

## 3. Candidate table — contract between backend and model teammate

One row per (`trip_id`, `observation_id`, `candidate_arc_id`). A candidate is a road arc near a GPS observation. The number of candidates can vary by observation.

**Required metadata (never used as predictive numeric features):**
- `trip_id:str`
- `observation_id:int64`
- `candidate_arc_id:int64` (same as `.arcs` row index)

**Proposed initial model features (numeric, units specified):**
- `distance_to_arc_m:float64` — physical point-to-segment distance in meters using a justified local projection or geodesic method
- `heading_difference_deg:float64|null` — difference between observed movement bearing and candidate arc bearing, 0–180 degrees; undefined at the first point or invalid movement
- `speed_mps:float64|null` — distance between consecutive observations divided by positive time difference
- `time_gap_s:float64|null` — elapsed time since previous observation
- `arc_length_m:float64` — candidate segment length in meters

**Label (only when independently justified):**
- `label:int8|null` — 1 if candidate is established as the correct road segment for that observation; 0 if established incorrect; null if not reliably alignable
- `label_source:str|null` — provenance/method of alignment (not a predictive feature)

**Important:** Presence of an arc anywhere in `.route` does **not** establish that it is correct for a particular observation. Do not train on fabricated observation labels. Before supervised training, implement and inspect an explicit temporal/spatial alignment method, document its uncertainty, and exclude uncertain observations from supervised scoring. Record the number excluded and candidate miss rate.

**Not yet fixed:** candidate search radius, maximum candidate count, projection, heading derivation, feature imputation, and alignment method. These must be agreed from measured EDA and training data, not guessed or chosen using evaluator results.

## 4. Reference-route alignment — training-only contract

**Input:** `gps_observations`, `road_nodes`, `road_arcs`, and `reference_route` from the same trip. The `.route` sequence describes traversed arcs, **not one label per GPS sample**. Repeated arc IDs may represent repeated traversal and must not be collapsed blindly.

**Proposed alignment output (`reference_alignment`):**
- `trip_id:str`
- `observation_id:int64`
- `reference_arc_id:int64|null` — selected reference arc, if supportable
- `reference_route_step_id:int64|null` — position in the supplied ordered route; distinct from arc ID
- `alignment_distance_m:float64|null` — point-to-selected-arc distance
- `alignment_status:str` — `valid`, `ambiguous`, `unmatched`, or `invalid_input`
- `alignment_method:str` — name/version of the method used

**Alignment procedure to investigate (not a mandated or verified solution):** reconstruct segment geometry by joining route arc IDs to arcs and nodes; project coordinates appropriately; generate nearby route-step candidates for each GPS point; find a spatially plausible, order-aware alignment (e.g., dynamic programming), permitting multiple GPS points on a segment, repeated arc traversals, and documented gap handling. Inspect real trips visually and report ambiguous cases. Do not assume all route arcs form one uninterrupted connected chain without validating topology.

**Label rule:** create binary candidate labels only for observations with validated `alignment_status=valid`: `label=1` when `candidate_arc_id == reference_arc_id`, otherwise `label=0`. For all other observations set `label=null` and exclude them from supervised training and observation-level reference metrics. A route arc's presence somewhere in the `.route` file is **not** enough to label a GPS point. If the reference arc is absent from the candidate set, record a **candidate miss**; do not invent a positive candidate or quietly discard the miss from coverage reporting.

**Leakage rule:** alignment and reference-route fields are **training/evaluation-only** and must never be input features, candidate generators, decoder inputs, or inference-time tie breakers. At inference, generate candidates using GPS and road-network geometry alone. Evaluator labels remain private to server-side scoring; the UI receives only predictions, display geometry, and aggregate metrics.

**Acceptance before supervised training:** inspect several real trips, quantify valid/ambiguous/unmatched rates and candidate misses, verify IDs and spatial geometry, and obtain both teammates' approval. Treat resulting labels as *derived reference-alignment labels*, not independently verified truth.

## 5. Prediction output — model to decoder

One row per candidate with `trip_id`, `observation_id`, `candidate_arc_id`, `candidate_score:float64` and `score_type:str` (`probability` or `decision_score`). For probabilities, each score must be finite and in [0,1]. Calibration/normalization and interpretation must be documented; a binary candidate probability need not sum to 1 across a GPS observation's candidates.

The decoder groups candidates by `(trip_id, observation_id)`, sorts by `observation_id`, uses road connectivity and transition costs, and returns one selected arc ID per relevant observation. Define and log a fallback for observations with no candidates and for disconnected roads. No ground-truth columns may enter the prediction pipeline.

`decoded_observations`: `trip_id:str`, `observation_id:int64`, `predicted_arc_id:int64|null`, `status:str` (e.g. `matched`, `no_candidate`, `unreachable`). Report coverage separately if any observations remain unmatched.

## 6. API contract — Laptop 1 and Laptop 2

**Proposed endpoints** (team design; guidelines mandate FastAPI and a two-laptop workflow, not these exact URLs):
- `GET /health` → `{ "status": "ok" }`
- `GET /trips` → `{ "trip_ids": ["..."] }` (reserved evaluator trips only during final evaluation)
- `POST /predict` with `{ "trip_id": "00000000" }` → trip ID, prediction/route geometry suitable for UI, and aggregate evaluation metrics if available

The server resolves the trip ID against an allowlisted server-side partition. It does **not** accept arbitrary filesystem paths. The UI receives no reference route IDs or ground-truth labels. The server may use reference routes internally for permitted evaluation and return aggregate metrics, but **not** use them to compute predictions. Define metric denominators and label provenance explicitly; observation-level accuracy is not justified without defensible reference alignment. API error responses should distinguish unknown trip IDs, missing files, and inference failures. During final evaluation, expose only the 10 reserved evaluator trips in the UI.

## 7. Split of ownership

**Laptop 1 / backend owner:** raw loaders, validation, candidate generation, feature generation, label-alignment investigation, HMM/Viterbi, FastAPI, model loading, backend tests.

**Laptop 2 / ML and UI owner:** dataset EDA and Folium maps, five model experiments and comparisons, Bayesian optimisation with group-aware CV, model export, Streamlit UI, frontend tests.

**Joint decisions:** approve columns, label alignment, candidate generation parameters, preprocessing pipeline, output model artifact format, API JSON shape and integration tests. Both contribute to the report.

## 8. Validation and handoff checklist

- [ ] Compare loaded counts with source file row counts for multiple trips
- [ ] Validate node/arc ID bounds and preserve 0-based indexing
- [ ] Measure timestamp irregularities and invalid GPS values
- [ ] Demonstrate one Folium map from actual data
- [ ] Prove label alignment on an inspected example; document unalignable cases
- [ ] Agree on a tiny sample candidate CSV with the exact schema
- [ ] Agree on a saved `joblib` model **plus its preprocessing pipeline**, ordered feature list, and version metadata
- [ ] Compare 5 models using group-aware training CV and 10 student-test trips
- [ ] Run Bayesian tuning and record trial history
- [ ] Compare classifier-only against HMM/Viterbi
- [ ] Verify Laptop 2 UI calls Laptop 1 API over the network and API logs show requests
- [ ] Keep evaluator labels out of the inference pipeline and UI
- [ ] Commit relevant real experiment results, artifacts and documentation for final submission

## 9. Open decisions to resolve before model training

1. How to align `.route` arc sequences with GPS observations without assuming equal row counts?
2. What candidate search algorithm, radius and maximum candidate count are justified by EDA?
3. Which spatial coordinate/distance method fits the actual geography?
4. How are missing/undefined features represented and imputed inside cross-validation folds?
5. What exact scorer and saved-model interface will be used by the decoder?
6. How will route accuracy be defined when reference data is sequence-level rather than observation-level?

**Do not claim observation-level ground truth or metrics until the alignment issue is resolved.**
