"""Train PATHSNAP Logistic Regression with trip-grouped validation."""

import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathsnap.models.train import make_model
from pathsnap.models.evaluate import evaluate_model, labelled_rows

from pathsnap.features.engineering import FEATURE_COLUMNS

FEATURES = list(FEATURE_COLUMNS)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
OUTPUT = ROOT / "results/experiments/laptop1"
ARTIFACTS = ROOT / "artifacts/laptop1"
OUTPUT.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def load_partition(partition):
    candidates = pd.read_csv(
        DATA / f"{partition}_candidates.csv",
        dtype={"trip_id": str},
    )
    observations = pd.read_csv(
        DATA / f"{partition}_observations.csv",
        dtype={"trip_id": str},
    )

    # Use partition-qualified IDs to avoid train/test collisions.
    candidates["trip_id"] = partition + ":" + candidates["trip_id"]
    observations["trip_id"] = partition + ":" + observations["trip_id"]

    return candidates, observations


def main():
    print("Loading PATHSNAP data...", flush=True)
    train, train_obs = load_partition("train")
    test, test_obs = load_partition("student_test")

    print("Training trips:", train_obs.trip_id.nunique(), flush=True)
    print("Student-test trips:", test_obs.trip_id.nunique(), flush=True)
    print("Training candidates:", len(train), flush=True)

    assert train_obs.trip_id.nunique() == 80
    assert test_obs.trip_id.nunique() == 10
    assert not set(train_obs.trip_id) & set(test_obs.trip_id)

    for frame in (train, test):
        missing = set(FEATURES) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

    model = make_model("logistic_regression", seed=42)

    print("\nStarting 5-fold grouped cross-validation...", flush=True)
    splitter = GroupKFold(n_splits=5)
    fold_results = []

    for fold, (tr_idx, va_idx) in enumerate(
        splitter.split(train_obs, groups=train_obs.trip_id), start=1
    ):
        start = perf_counter()
        tr_obs = train_obs.iloc[tr_idx]
        va_obs = train_obs.iloc[va_idx]

        tr_rows = train[train.trip_id.isin(tr_obs.trip_id)]
        va_rows = train[train.trip_id.isin(va_obs.trip_id)]

        fitted = clone(model)
        labelled = labelled_rows(tr_rows, tr_obs)
        fitted.fit(labelled[FEATURES], labelled.is_match.astype(int))

        metrics = evaluate_model(fitted, va_rows, va_obs, FEATURES)
        fold_results.append(metrics)

        print(
            f"Fold {fold}/5 | "
            f"Top-1: {metrics['top1_accuracy']:.4f} | "
            f"Macro F1: {metrics['candidate_macro_f1']:.4f} | "
            f"Time: {perf_counter() - start:.1f}s",
            flush=True,
        )

    print("\nTraining final model on all training trips...", flush=True)
    labelled = labelled_rows(train, train_obs)
    model.fit(labelled[FEATURES], labelled.is_match.astype(int))

    print("Evaluating reserved student-test trips...", flush=True)
    test_metrics = evaluate_model(model, test, test_obs, FEATURES)

    results = {
        "model": "logistic_regression",
        "features": FEATURES,
        "seed": 42,
        "cv_top1_accuracy": sum(
            x["top1_accuracy"] for x in fold_results
        ) / len(fold_results),
        "cv_macro_f1": sum(
            x["candidate_macro_f1"] for x in fold_results
        ) / len(fold_results),
        "folds": fold_results,
        "student_test": test_metrics,
    }

    results_path = OUTPUT / "logistic_regression.json"
    results_path.write_text(json.dumps(results, indent=2))

    model_path = ARTIFACTS / "logistic_regression.joblib"
    joblib.dump(model, model_path)

    print("\nTRAINING COMPLETE")
    print("CV Top-1:", results["cv_top1_accuracy"])
    print("CV Macro F1:", results["cv_macro_f1"])
    print("Student-test Top-1:", test_metrics["top1_accuracy"])
    print("Model saved:", model_path)
    print("Results saved:", results_path)


if __name__ == "__main__":
    main()
