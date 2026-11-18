# Feature engineering v2: what changed and why

Partition used for all diagnostics below: **train (80 trips)** unless stated.
Labels are unchanged: same candidate generation (k=8, 100 m), same monotonic
reference-route alignment and the same `review_required` filter as v1. Only
the predictor columns changed, so v1 and v2 numbers are directly comparable.

## Diagnosis of v1 (6 features)

1. **Reverse-direction twin arcs.** 82–100% of arcs in each trip's network
   have a reverse arc on the same two nodes (A→B and B→A). Their geometry is
   identical, so `distance_to_arc_m` is tied, and the reference arc was
   candidate rank 0 or rank 1 about equally often. Most v1 Top-1 errors were
   picking the wrong direction of the correct road.
2. **Heading came from one 1-second GPS step**, and was NaN whenever that step
   was under 1 m. On noisy or slow segments this is close to a coin flip.
   On twin-pair observations (first 20 train trips) the v1 heading picked the
   correct direction 79% of the time (mean over trips); a displacement taken
   over ±5 observations picked it 92% of the time.
3. `candidate_rank` broke twin ties by arc index, which is arbitrary.
4. No temporal context and nothing relative to the other candidates of the
   same GPS fix.

## v2 features (`src/pathsnap/features/engineering.py`)

| Group | Feature | Why |
|---|---|---|
| Geometry | `distance_to_arc_m`, `arc_length_m` | kept from v1 |
| | `distance_delta_m`, `distance_ratio` | distance relative to the best candidate of the same fix; GPS noise level differs by trip, relative values transfer better |
| | `distance_rank_tied` | dense rank on distance rounded to 0.1 m, so twins share a rank instead of an arbitrary 0/1 |
| | `projection_fraction`, `endpoint_overshoot_m` | whether the fix projects inside the arc or beyond its end (next arc likely) |
| Direction | `align_cos_w2/w5/w10` | cosine between GPS displacement from t−w to t+w and arc direction; signed, so it separates A→B from B→A. Set to 0 (unknown) if displacement < 2 m |
| | `align_cos_delta_w5` | cosine minus best cosine in the fix |
| | `along_track_w5_m` | signed metres moved along the arc direction |
| | `heading_difference_deg` | now from a ±3 window (v1 used one step) |
| Context | `context_dist_w3_m`, `context_dist_w10_m`, `context_dist_delta_w10_m` | mean distance of neighbouring fixes to the arc: stays small for the road actually driven, separates parallel streets |
| Observation / network | `speed_mps` (±2 window), `time_gap_s`, `n_candidates`, `has_twin_candidate`, `to_node_out_degree` | motion state, ambiguity of the fix, intersection density |

All features use only GPS + road network. They use fixes before and after t,
which is legitimate here because the whole trip is decoded offline (the API
receives the full trip). No reference data is read.

## Effect (HistGradientBoosting, same settings for both feature sets)

| Feature set | Grouped 5-fold CV Top-1 (train) | Student-test Top-1 |
|---|---|---|
| v1 (6 features) | 0.802 ± 0.063 | 0.818 |
| v2 (21 features) | 0.924 ± 0.021 | 0.915 |

Fold standard deviation also dropped by about 3x, so the model is more stable
across trips. Full per-model results are in `results/experiments/laptop1/`.

## API change

`src/pathsnap/api/main.py` used to read `data/processed/student_test_candidates.csv`,
which only covers student-test trips and also carries label columns. It now
builds candidates and features from the raw trip at request time
(`generate_candidates` → `add_features`), so it works for the 10 evaluator trips
and keeps reference labels out of the prediction path.
