"""Reproducible five-family experiments using trip-grouped validation."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from .neural import CandidateMLP

from .evaluate import KEY, labelled_rows, positive_scores, evaluate_scores, evaluate_model, load_tables

FAMILIES = {
    "logistic_regression": "linear/probabilistic", "weighted_knn": "instance-based",
    "random_forest": "bagging ensemble", "hist_gradient_boosting": "boosting",
    "mlp": "neural network",
}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding="utf-8")


def make_model(name, seed=42, parameters=None):
    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
        "weighted_knn": KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="auto", n_jobs=-1),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        # Disable internal row-wise early stopping: validation must remain trip-grouped.
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, early_stopping=False, class_weight="balanced", random_state=seed),
        "mlp": CandidateMLP(random_state=seed),
    }
    if name not in models:
        raise ValueError(f"Unknown model {name}; choose from {list(models)}")
    classifier = models[name]
    classifier.set_params(**(parameters or {}))
    if getattr(classifier, "early_stopping", False):
        raise ValueError("Internal row-based validation is disabled; tune the epoch budget using grouped CV.")
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()), ("classifier", classifier)])


def fit_model(model, candidates, observations, features, calibration_folds=0):
    labelled = labelled_rows(candidates, observations)
    if set(labelled.is_match.unique()) != {0, 1}:
        raise ValueError("Each training split must contain both candidate classes 0 and 1.")
    if calibration_folds:
        splits = list(GroupKFold(n_splits=calibration_folds).split(labelled, groups=labelled.trip_id))
        for train_idx, valid_idx in splits:
            if any(labelled.iloc[idx].is_match.nunique() != 2 for idx in [train_idx, valid_idx]):
                raise ValueError("Both classes are required in each grouped calibration split.")
        model = CalibratedClassifierCV(model, method="sigmoid", cv=splits, ensemble=False)
    model.fit(labelled[features], labelled.is_match.astype(int))
    return model


def cross_validate(model, candidates, observations, features, folds=5, calibration_folds=0):
    if observations.trip_id.nunique() < folds or folds < 2:
        raise ValueError("Need at least one distinct trip per fold and at least two folds.")
    results = []
    splitter = GroupKFold(n_splits=folds)
    # Split the inventory so observations without candidates remain in validation.
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(observations, groups=observations.trip_id)):
        train_obs, valid_obs = observations.iloc[train_idx], observations.iloc[valid_idx]
        train_rows = candidates[candidates.trip_id.isin(train_obs.trip_id)]
        valid_rows = candidates[candidates.trip_id.isin(valid_obs.trip_id)]
        fitted = fit_model(clone(model), train_rows, train_obs, features, calibration_folds)
        metrics = evaluate_model(fitted, valid_rows, valid_obs, features)
        if metrics["candidate_macro_f1"] is None:
            raise ValueError("A validation fold has no labelled candidate rows.")
        results.append({"fold": fold, "train_trips": sorted(train_obs.trip_id.unique()),
                        "validation_trips": sorted(valid_obs.trip_id.unique()), **metrics})
    values = [r["candidate_macro_f1"] for r in results]
    top1 = [r["top1_accuracy"] for r in results]
    return {"cv_macro_f1": float(np.mean(values)), "cv_macro_f1_std": float(np.std(values)),
            "cv_top1_accuracy": float(np.mean(top1)), "cv_top1_accuracy_std": float(np.std(top1)), "folds": results}


def read_inputs(config):
    features = config["features"]
    if not isinstance(features, list) or not features:
        raise ValueError("Set features to Laptop 1's agreed numeric feature names in the YAML config.")
    train, train_obs = load_tables(config["train_candidates"], config["train_observations"], features)
    test, test_obs = load_tables(config["student_test_candidates"], config["student_test_observations"], features)
    if set(train_obs.trip_id) & set(test_obs.trip_id):
        raise ValueError("Training and student-test trip IDs overlap. Supply globally unique IDs.")
    expected = config.get("expected_trips", {})
    for partition, count in [("train", train_obs.trip_id.nunique()), ("student_test", test_obs.trip_id.nunique())]:
        if partition in expected and expected[partition] != count:
            raise ValueError(f"Expected {expected[partition]} {partition} trips, received {count}.")
    return features, train, train_obs, test, test_obs


def run_experiments(config):
    features, train, train_obs, test, test_obs = read_inputs(config)
    results = []
    names = config["models"]
    if len(set(names)) < 5 or len(set(names)) != len(names):
        raise ValueError("Compare at least five distinct models; do not repeat entries.")
    for name in names:
        start = perf_counter()
        model = make_model(name, config.get("seed", 42), config.get("parameters", {}).get(name))
        parameters = model.named_steps["classifier"].get_params()
        calibration_folds = config.get("calibration_folds", 0)
        cv = cross_validate(model, train, train_obs, features, config.get("folds", 5), calibration_folds)
        fit_start = perf_counter()
        model = fit_model(model, train, train_obs, features, calibration_folds)
        fitting = perf_counter() - fit_start
        result = {"model": name, "family": FAMILIES[name], "seed": config.get("seed", 42),
                  "parameters": parameters, "calibration_folds": calibration_folds, **cv,
                  "training_time_s": fitting,
                  "train": evaluate_model(model, train, train_obs, features),
                  "student_test": evaluate_model(model, test, test_obs, features),
                  "total_time_s": perf_counter() - start}
        fitted_pipeline = model if hasattr(model, "named_steps") else model.calibrated_classifiers_[0].estimator
        classifier = fitted_pipeline.named_steps["classifier"]
        result["training_loss_history"] = list(getattr(classifier, "loss_curve_", []))
        result["training_class_counts"] = {str(k): int(v) for k, v in labelled_rows(train, train_obs).is_match.value_counts().items()}
        save_json(Path(config["experiment_dir"]) / f"{name}.json", result)
        results.append(result)
    best = max(results, key=lambda r: (r["cv_top1_accuracy"], -r["cv_top1_accuracy_std"], r["cv_macro_f1"]))
    save_json(Path(config["experiment_dir"]) / "selection.json", {
        "selected_model": best["model"], "objective": "mean trip-grouped CV observation Top-1 accuracy",
        "tie_break": "lower Top-1 fold standard deviation, then higher candidate Macro F1", "config": config})
    return results


def export_bundle(model, directory, config, name, parameters, metrics):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.joblib"
    joblib.dump(model, model_path)
    manifest = {"schema_version": 1, "model": name, "parameters": parameters,
                "features": config["features"], "feature_units": config.get("feature_units", {}),
                "positive_class": 1, "classes": [int(v) for v in model.classes_],
                "preprocessing": "median imputation and standard scaling fitted within CV folds; refitted on all training trips",
                "score_semantics": "binary candidate P(is_match=1); not normalized across an observation",
                "seed": config.get("seed", 42), "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["scikit-learn", "numpy", "pandas", "joblib"]},
                "training_candidates_sha256": hashlib.sha256(Path(config["train_candidates"]).read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "config": config,
                "metrics": metrics}
    save_json(directory / "model_manifest.json", manifest)
