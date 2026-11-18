"""Gaussian-process Bayesian optimisation with expected improvement acquisition."""
import json
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning
from pathsnap.models.train import make_model, cross_validate, read_inputs, fit_model, export_bundle, save_json
from pathsnap.models.evaluate import evaluate_model, labelled_rows
from sklearn.model_selection import GroupKFold

# Bounds are converted to a unit cube for the GP; logarithmic parameters use log10.
SPACES = {
    "logistic_regression": {"C": ("log", -3, 2), "class_weight": ("choice", [None, "balanced"], None)},
    "weighted_knn": {"n_neighbors": ("int", 3, 51), "p": ("int", 1, 2), "leaf_size": ("int", 15, 60)},
    "random_forest": {"n_estimators": ("int", 100, 400), "max_depth": ("int", 5, 30), "min_samples_leaf": ("int", 1, 20),
                      "class_weight": ("choice", [None, "balanced_subsample"], None)},
    "hist_gradient_boosting": {"learning_rate": ("log", -2, np.log10(.3)), "max_leaf_nodes": ("int", 8, 63),
                               "l2_regularization": ("log", -6, 1), "max_iter": ("int", 75, 300),
                               "class_weight": ("choice", [None, "balanced"], None)},
    "mlp": {"hidden_layer_sizes": ("choice", [(32,), (64,), (64, 32)], None),
            "alpha": ("log", -5, -1), "learning_rate_init": ("log", -4, -2),
            "max_iter": ("int", 75, 250), "balance_classes": ("choice", [False, True], None)},
}


def decode_point(point, space):
    parameters = {}
    for coordinate, (key, (kind, low, high)) in zip(point, space.items()):
        if kind == "choice":
            parameters[key] = low[min(int(float(coordinate) * len(low)), len(low) - 1)]
            continue
        value = low + float(coordinate) * (high - low)
        parameters[key] = int(round(value)) if kind == "int" else float(10 ** value if kind == "log" else value)
    return parameters


def bayesian_search(objective, space, trials, seed=42, startup=5):
    if trials < 1: raise ValueError("At least one trial required.")
    rng = np.random.default_rng(seed)
    points, values, records, seen = [], [], [], set()
    for number in range(trials):
        pool = rng.uniform(0, 1, (1024, len(space)))
        pool = np.asarray([p for p in pool if json.dumps(decode_point(p, space), sort_keys=True) not in seen])
        if not len(pool): break
        method = "random_initialization"
        if len(points) >= startup:
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(noise_level=1e-5),
                                          normalize_y=True, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(np.asarray(points), np.asarray(values))
            mean, std = gp.predict(pool, return_std=True)
            improvement = mean - max(values) - .001
            z = improvement / np.maximum(std, 1e-12)
            acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
            point = pool[int(np.argmax(acquisition))]
            method = "gaussian_process_expected_improvement"
        else:
            point = pool[0]
        parameters = decode_point(point, space)
        seen.add(json.dumps(parameters, sort_keys=True))
        start = perf_counter()
        value, details = objective(parameters)
        if not np.isfinite(value): raise ValueError("Optimisation objective is not finite.")
        points.append(point); values.append(float(value))
        records.append({"trial": number, "objective": float(value), "parameters": parameters,
                        "acquisition": method, "elapsed_s": perf_counter() - start, "details": details})
    return records


def tune(config, trials=30):
    features, train, train_obs, test, test_obs = read_inputs(config)
    selection = json.loads((Path(config["experiment_dir"]) / "selection.json").read_text(encoding="utf-8"))
    if selection["config"] != json.loads(json.dumps(config)):
        raise ValueError("Experiment config changed; rerun experiments before tuning.")
    name, seed = selection["selected_model"], config.get("seed", 42)
    baseline = config.get("parameters", {}).get(name, {})
    calibration_folds = config.get("calibration_folds", 0)
    space = dict(SPACES[name])
    if name == "weighted_knn":
        minimum = len(labelled_rows(train, train_obs))
        for indices, _ in GroupKFold(config.get("folds", 5)).split(train_obs, groups=train_obs.trip_id):
            subset = train_obs.iloc[indices]
            labelled = labelled_rows(train[train.trip_id.isin(subset.trip_id)], subset)
            minimum = min(minimum, len(labelled))
            if calibration_folds:
                for inner, _ in GroupKFold(calibration_folds).split(labelled, groups=labelled.trip_id):
                    minimum = min(minimum, len(inner))
        space["n_neighbors"] = ("int", min(3, minimum), min(51, minimum))
    output = Path(config["experiment_dir"]) / "tuning"; output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    def objective(parameters):
        model = make_model(name, seed, {**baseline, **parameters})
        cv = cross_validate(model, train, train_obs, features, config.get("folds", 5), calibration_folds)
        return cv["cv_top1_accuracy"], cv
    records = bayesian_search(objective, space, trials, seed)
    save_json(output / "trials.json", records)
    pd.DataFrame([{k: v for k, v in r.items() if k != "details"} for r in records]).to_csv(output / "trials.csv", index=False)
    best = max(records, key=lambda r: r["objective"])
    parameters = {**baseline, **best["parameters"]}
    fit_start = perf_counter()
    model = fit_model(make_model(name, seed, parameters), train, train_obs, features, calibration_folds)
    fit_time = perf_counter() - fit_start
    tuned_metrics = evaluate_model(model, test, test_obs, features)
    untuned = json.loads((Path(config["experiment_dir"]) / f"{name}.json").read_text(encoding="utf-8"))
    result = {"model": name, "method": "Gaussian process with expected improvement; 5 random startup trials",
              "search_space": space, "objective": "mean grouped CV observation Top-1 accuracy",
              "requested_trials": trials, "completed_trials": len(records), "best_parameters": parameters,
              "tuned_cv_top1_accuracy": best["objective"], "untuned_cv_top1_accuracy": untuned["cv_top1_accuracy"],
              "tuned_cv_macro_f1": best["details"]["cv_macro_f1"], "untuned_cv_macro_f1": untuned["cv_macro_f1"],
              "tuned_training_time_s": fit_time, "untuned_training_time_s": untuned["training_time_s"],
              "tuned_student_test": tuned_metrics, "untuned_student_test": untuned["student_test"],
              "elapsed_s": perf_counter() - start}
    if best["objective"] < untuned["cv_top1_accuracy"]:
        parameters = baseline
        model = fit_model(make_model(name, seed, parameters), train, train_obs, features, calibration_folds)
        result["exported_variant"] = "untuned"
    else: result["exported_variant"] = "tuned"
    export_bundle(model, config["artifact_dir"], config, name, parameters, result)
    save_json(output / "summary.json", result)
    return result
