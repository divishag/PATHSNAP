# Ownership and handoff status

## Completed Laptop 2 work

| Area | Files / evidence | Completion meaning |
| --- | --- | --- |
| Raw-data EDA | data/explore.py, scripts/explore_dataset.py, results/eda | Actually run on 80 train and 10 student-test trips |
| Model experiments | models/train.py, models/neural.py, scripts/train_models.py | Five-family code checked using temporary synthetic data |
| Metrics | models/evaluate.py | Candidate binary metrics, Top-1, candidate misses, Brier and scoring time |
| Comparison | models/compare.py, visualization/plots.py | Measured-file comparison tables, graphs, confusion matrices and real MLP history support |
| Bayesian search | optimization/bayesian.py, scripts/tune_model.py | GP expected improvement; actual acquisition and export checked on synthetic fixtures |
| Export | models/train.py: export_bundle | Pipeline/bundle export code, version manifest and hashes; no real weights yet |
| Development maps | visualization/folium_maps.py, scripts/generate_maps.py | Actual raw GPS/reference maps produced; separate from evaluator UI |
| Evaluator UI | frontend/ | Dark/light modes, catalogue, prediction request, metric cards, map and downloads |
| UI regression checks | tests/test_frontend_laptop2.py | Eight independent checks with simulated API responses |
| Workflow regression checks | tests/test_laptop2.py | Seven checks including leakage guards, grouped calibration and all five models |
| Report sections | reports/EDA_and_UI_progress.md | Actual EDA findings, figures, UI implementation and remaining dependencies |
| Provenance | reports/eda_source_manifest.json | SHA-256 for all 360 inspected raw trip files |
| Integration | docs/laptop2_integration.md, docs/laptop2_runbook.md | Proposed schema, model scoring boundary, API fields and commands |

Source paths under data/models/visualization above are relative to src/pathsnap/.

## New five-model plan

1. Logistic Regression: interpretable linear/probabilistic baseline.
2. Distance-weighted kNN: instance-based local similarity; weighting is not class balancing.
3. Random Forest: bagging ensemble for nonlinear interactions.
4. HistGradientBoosting: boosting model for tabular candidate features.
5. Shallow MLP: neural network with training-fold sample balancing and scaling.

No model is declared best. CV groups are whole trips. The primary selection
metric is observation Top-1, with candidate Macro F1, Log Loss, Brier and latency
reported alongside it. Optional sigmoid calibration uses inner grouped folds.
The old Decision Tree/GaussianNB shortlist is superseded; no real experiments
were run under that shortlist.

## Laptop 1 ownership (not implemented by this handoff)

- data/loader.py, validate.py and route_alignment.py.
- features/candidates.py, engineering.py and labels.py.
- decoding/hmm.py and viterbi.py.
- api/main.py, schemas.py and pipeline.py.
- scripts/build_features.py and evaluate_decoder.py.
- Their loader/candidate/features/alignment/decoder/API tests.
- configs/data.yaml, decoder.yaml and api.yaml.

## Deferred until Laptop 1 handoff

1. Agree feature names, units, aligned-label semantics and observation inventory.
2. Receive train/student-test candidates and inventory CSVs.
3. Run actual five-model experiments, inspect metrics and choose the model.
4. Run Bayesian tuning, compare calibration if justified, and deliver real weights.
5. Compare classifier-only predictions with HMM/Viterbi reconstruction.
6. Run all ten reserved evaluator trips through the real two-laptop API.
7. Capture server logs and real route/metric evidence, then complete the final PDF.

## Independent validation evidence

- Seven model/workflow tests and eight frontend tests pass.
- Test scores are synthetic and remain in temporary directories, not report results.
- MLP's short synthetic fit can reach its iteration limit; this is not evidence
  of convergence on real data. Real runs must inspect warnings and measured curves.
- Desktop dark and light themes were inspected in the browser.
- Mobile width 390 x 844 was inspected in both themes. Measured document width
  equalled viewport width (390 px); no page-level horizontal overflow was found.
- Mobile review identified a light-mode expander header contrast issue; fixed
  and checked in the rendered page. Screenshots are under reports/screenshots/.
- Screenshots show the genuine interface connection screen, not a real backend run.
- Simulated API checks cover requests, malformed data, failed refresh, timeout,
  public download contents and preservation/clearing of current results.

## Merge instructions

The handoff ZIP includes Laptop 2 source, configuration, tests, documentation,
actual EDA evidence and report sections. Review shared README/docs and
configs/models.yaml when merging. Preserve Laptop 1 modules. Source files are
laid out at their repository-relative paths inside the ZIP. HandoffManifest.json
records included files and their SHA-256 values. Raw data and virtual environments
are intentionally excluded. Run scripts/package_laptop2.py to regenerate it.

These local changes have not been committed or pushed to GitHub in this session.
