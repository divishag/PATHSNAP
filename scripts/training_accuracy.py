"""Training-set Top-1 for each saved model (overfitting check, report section 3.2 C.9).

    PYTHONPATH=src python scripts/training_accuracy.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.features.engineering import FEATURE_COLUMNS
from pathsnap.models.evaluate import evaluate_model

MODELS = sys.argv[1:] or ["hist_gradient_boosting", "random_forest", "mlp", "logistic_regression",
                          "hist_gradient_boosting_no_class_weight", "weighted_knn"]
cands = pd.read_csv(ROOT / "data/processed/train_candidates.csv", dtype={"trip_id": str})
obs = pd.read_csv(ROOT / "data/processed/train_observations.csv", dtype={"trip_id": str})
out = ROOT / "results/comparison/training_accuracy.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
for name in MODELS:
    model = joblib.load(ROOT / f"artifacts/laptop1/{name}.joblib")
    m = evaluate_model(model, cands, obs, list(FEATURE_COLUMNS))
    exp = json.loads((ROOT / f"results/experiments/laptop1/{name}.json").read_text())
    row = {"model": name, "train_top1": m["top1_accuracy"], "cv_top1": exp["cv_top1_accuracy"],
           "student_test_top1": exp["student_test"]["top1_accuracy"],
           "train_minus_test_gap_pp": 100 * (m["top1_accuracy"] - exp["student_test"]["top1_accuracy"]),
           "train_macro_f1": m["candidate_macro_f1"]}
    rows = [r for r in rows if r["model"] != name] + [row]
    pd.DataFrame(rows).to_csv(out, index=False)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
