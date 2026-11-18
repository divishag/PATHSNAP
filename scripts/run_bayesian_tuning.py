"""Bayesian hyperparameter optimisation for the selected PathSnap model.

Search:     Gaussian-process surrogate + expected improvement
            (pathsnap.optimization.bayesian.bayesian_search), 5 random
            start-up trials, then GP-guided trials.
Objective:  mean observation Top-1 accuracy over trip-grouped CV folds on
            the 80 training trips (GroupKFold on trip_id). Student-test trips
            are only used once, after the search, to compare tuned vs
            untuned. Evaluator-testing data is never touched.

Run from the repository root:
    PYTHONPATH=src python scripts/run_bayesian_tuning.py --model hist_gradient_boosting
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.features.engineering import FEATURE_COLUMNS
from pathsnap.models.evaluate import evaluate_model, labelled_rows
from pathsnap.models.train import make_model
from pathsnap.optimization.bayesian import bayesian_search

FEATURES = list(FEATURE_COLUMNS)
DATA = ROOT / "data/processed"
OUTPUT = ROOT / "results/tuning"
ARTIFACTS = ROOT / "artifacts/tuned"

# Search spaces: (kind, low, high). "log" bounds are log10 values.
SPACES = {
    "hist_gradient_boosting": {
        "learning_rate": ("log", -2, np.log10(0.3)),
        "max_leaf_nodes": ("int", 15, 63),
        "min_samples_leaf": ("int", 10, 200),
        "l2_regularization": ("log", -4, 1),
        "max_iter": ("int", 100, 300),
        "class_weight": ("choice", [None, "balanced"], None),
    },
    "random_forest": {
        "n_estimators": ("int", 60, 200),
        "max_depth": ("int", 8, 40),
        "min_samples_leaf": ("int", 1, 30),
        "max_features": ("choice", ["sqrt", 0.5, 0.8], None),
    },
}
FIXED = {
    "hist_gradient_boosting": {"early_stopping": False},
    "random_forest": {"max_samples": 0.25, "n_jobs": -1,
                      "class_weight": "balanced_subsample"},
}


def load_partition(partition):
    candidates = pd.read_csv(DATA / f"{partition}_candidates.csv", dtype={"trip_id": str})
    observations = pd.read_csv(DATA / f"{partition}_observations.csv", dtype={"trip_id": str})
    candidates["trip_id"] = partition + ":" + candidates["trip_id"]
    observations["trip_id"] = partition + ":" + observations["trip_id"]
    return candidates, observations


def grouped_cv(model, train, train_obs, folds):
    results = []
    splitter = GroupKFold(n_splits=folds)
    for tr_idx, va_idx in splitter.split(train_obs, groups=train_obs.trip_id):
        tr_obs, va_obs = train_obs.iloc[tr_idx], train_obs.iloc[va_idx]
        fitted = clone(model)
        labelled = labelled_rows(train[train.trip_id.isin(tr_obs.trip_id)], tr_obs)
        fitted.fit(labelled[FEATURES], labelled.is_match.astype(int))
        metrics = evaluate_model(
            fitted, train[train.trip_id.isin(va_obs.trip_id)], va_obs, FEATURES)
        results.append(metrics)
    top1 = [m["top1_accuracy"] for m in results]
    f1 = [m["candidate_macro_f1"] for m in results]
    return {"cv_top1_accuracy": float(np.mean(top1)), "cv_top1_std": float(np.std(top1)),
            "cv_macro_f1": float(np.mean(f1)), "fold_top1": top1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hist_gradient_boosting", choices=list(SPACES))
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    train, train_obs = load_partition("train")
    test, test_obs = load_partition("student_test")
    assert train_obs.trip_id.nunique() == 80 and test_obs.trip_id.nunique() == 10

    name, space, fixed = args.model, SPACES[args.model], FIXED[args.model]
    start_all = perf_counter()

    def objective(parameters):
        t0 = perf_counter()
        model = make_model(name, args.seed, {**fixed, **parameters})
        cv = grouped_cv(model, train, train_obs, args.folds)
        print(f"  trial | CV Top-1 {cv['cv_top1_accuracy']:.4f} | "
              f"{perf_counter() - t0:.0f}s | {parameters}", flush=True)
        return cv["cv_top1_accuracy"], cv

    # Untuned reference: the exact settings used in the 5-model comparison,
    # evaluated with the same fold count so the comparison is fair.
    untuned_params = {"hist_gradient_boosting": {},
                      "random_forest": {"n_estimators": 100, "min_samples_leaf": 5}}[name]
    print("Untuned reference ...", flush=True)
    untuned_cv = grouped_cv(make_model(name, args.seed, {**fixed, **untuned_params}),
                            train, train_obs, args.folds)
    print(f"  untuned CV Top-1 {untuned_cv['cv_top1_accuracy']:.4f}", flush=True)

    print(f"Bayesian search: {args.trials} trials, {args.folds}-fold grouped CV", flush=True)
    records = bayesian_search(objective, space, args.trials, seed=args.seed, startup=5)

    (OUTPUT / f"{name}_trials.json").write_text(json.dumps(records, indent=2, default=str))
    pd.DataFrame([{"trial": r["trial"], "objective_cv_top1": r["objective"],
                   "acquisition": r["acquisition"], "elapsed_s": r["elapsed_s"],
                   **{f"param_{k}": v for k, v in r["parameters"].items()}}
                  for r in records]).to_csv(OUTPUT / f"{name}_trials.csv", index=False)

    best = max(records, key=lambda r: r["objective"])
    best_params = {**fixed, **best["parameters"]}

    def fit_full(params):
        model = make_model(name, args.seed, params)
        labelled = labelled_rows(train, train_obs)
        t0 = perf_counter()
        model.fit(labelled[FEATURES], labelled.is_match.astype(int))
        return model, perf_counter() - t0

    tuned_model, tuned_fit_s = fit_full(best_params)
    untuned_model, untuned_fit_s = fit_full({**fixed, **untuned_params})
    tuned_test = evaluate_model(tuned_model, test, test_obs, FEATURES)
    untuned_test = evaluate_model(untuned_model, test, test_obs, FEATURES)

    joblib.dump(tuned_model, ARTIFACTS / f"{name}_tuned.joblib")
    summary = {
        "model": name,
        "method": "Gaussian-process Bayesian optimisation, expected improvement, 5 random start-up trials",
        "objective": f"mean observation Top-1 accuracy, {args.folds}-fold GroupKFold over the 80 training trips",
        "search_space": {k: list(v) for k, v in space.items()},
        "fixed_parameters": fixed,
        "trials": len(records),
        "seed": args.seed,
        "best_trial": best["trial"],
        "best_parameters": best_params,
        "untuned_parameters": {**fixed, **untuned_params},
        "untuned_cv": untuned_cv,
        "tuned_cv": best["details"],
        "untuned_student_test": untuned_test,
        "tuned_student_test": tuned_test,
        "untuned_fit_time_s": untuned_fit_s,
        "tuned_fit_time_s": tuned_fit_s,
        "total_search_time_s": perf_counter() - start_all,
    }
    (OUTPUT / f"{name}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nTUNING COMPLETE")
    print(f"CV Top-1   untuned {untuned_cv['cv_top1_accuracy']:.4f} -> tuned {best['objective']:.4f}")
    print(f"Test Top-1 untuned {untuned_test['top1_accuracy']:.4f} -> tuned {tuned_test['top1_accuracy']:.4f}")
    print("Best parameters:", best_params)


if __name__ == "__main__":
    main()
