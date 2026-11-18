"""Independent Laptop 2 checks. All fixtures are synthetic, never report results."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np
import pandas as pd
from pathsnap.models.evaluate import validate_tables, evaluate_scores, positive_scores
from pathsnap.models.train import run_experiments, make_model, cross_validate, fit_model, FAMILIES
from pathsnap.optimization.bayesian import SPACES, decode_point
from pathsnap.models.compare import compare_experiments
from pathsnap.optimization.bayesian import bayesian_search, tune


def fixture(prefix, trips):
    rows, observations = [], []
    for trip in range(trips):
        for obs in range(4):
            observations.append({"trip_id": f"{prefix}{trip}", "observation_id": obs,
                                 "label_valid": 1, "reference_arc_id": 0 if obs < 2 else 99})
            if obs == 3: continue  # No candidates, but remains in metric denominator.
            for candidate in range(3):
                rows.append({"trip_id": f"{prefix}{trip}", "observation_id": obs,
                             "candidate_id": candidate, "arc_id": candidate,
                             "is_match": int(obs < 2 and candidate == 0),
                             "distance_m": float(candidate * 20 + trip % 2),
                             "heading_difference": float(candidate * 30)})
    return pd.DataFrame(rows), pd.DataFrame(observations)


class Laptop2Checks(unittest.TestCase):
    def test_five_distinct_families_and_valid_search_spaces(self):
        self.assertEqual(len(FAMILIES), 5)
        self.assertEqual(len(set(FAMILIES.values())), 5)
        self.assertEqual(set(FAMILIES), set(SPACES))
        for name, space in SPACES.items():
            model = make_model(name, parameters=decode_point([.5] * len(space), space))
            self.assertFalse(getattr(model.named_steps["classifier"], "early_stopping", False))

    def test_grouped_calibration_produces_usable_probabilities(self):
        rows, obs = fixture("a", 6)
        calibrated = fit_model(make_model("logistic_regression"), rows, obs, ["distance_m"], calibration_folds=3)
        scores = positive_scores(calibrated, rows, ["distance_m"])
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores > 0) & (scores < 1)).all())
        result = cross_validate(make_model("logistic_regression"), rows, obs, ["distance_m"], folds=2, calibration_folds=2)
        self.assertIn("cv_top1_accuracy", result)

    def test_missing_candidates_count_as_misses(self):
        rows, obs = fixture("a", 2)
        validate_tables(rows, obs, ["distance_m"])
        metrics = evaluate_scores(rows, obs, np.where(rows.candidate_id.eq(0), .9, .1))
        self.assertEqual(metrics["top1_accuracy"], .5)
        self.assertEqual(metrics["candidate_miss_rate"], .5)
        self.assertEqual(metrics["labelled_observations"], 8)

    def test_rejects_reference_leakage_and_wrong_labels(self):
        rows, obs = fixture("a", 2)
        with self.assertRaises(ValueError): validate_tables(rows, obs, ["is_match"])
        rows.loc[0, "is_match"] = 0
        with self.assertRaises(ValueError): validate_tables(rows, obs, ["distance_m"])

    def test_grouped_folds_do_not_share_trips(self):
        rows, obs = fixture("a", 4)
        result = cross_validate(make_model("logistic_regression"), rows, obs, ["distance_m"], folds=2)
        for fold in result["folds"]:
            self.assertFalse(set(fold["train_trips"]) & set(fold["validation_trips"]))

    def test_bayesian_acquisition_runs_after_startup(self):
        runs = bayesian_search(lambda p: (-(p["x"] - .3) ** 2, {}), {"x": ("float", 0., 1.)}, 7)
        self.assertEqual(len(runs), 7)
        self.assertEqual(runs[-1]["acquisition"], "gaussian_process_expected_improvement")

    def test_full_synthetic_workflow_exports_reloadable_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            config = {"features": ["distance_m", "heading_difference"], "seed": 42, "folds": 2,
                      "models": ["logistic_regression", "weighted_knn", "random_forest", "hist_gradient_boosting", "mlp"],
                      "parameters": {"random_forest": {"n_estimators": 5}, "hist_gradient_boosting": {"max_iter": 5},
                                     "weighted_knn": {"n_neighbors": 3}, "mlp": {"max_iter": 100, "hidden_layer_sizes": (8,)}},
                      "experiment_dir": str(target / "experiments"), "artifact_dir": str(target / "artifacts")}
            for partition, prefix, count in [("train", "a", 4), ("student_test", "b", 2)]:
                rows, obs = fixture(prefix, count)
                for kind, frame in [("candidates", rows), ("observations", obs)]:
                    path = target / f"{partition}_{kind}.csv"; frame.to_csv(path, index=False)
                    config[f"{partition}_{kind}"] = str(path)
            results = run_experiments(config)
            self.assertEqual(len(results), 5)
            for result in results:
                self.assertIn("cv_top1_accuracy", result)
            self.assertTrue(next(r for r in results if r["model"] == "mlp")["training_loss_history"])
            compare_experiments(config["experiment_dir"], target / "metrics", target / "plots")
            tuned = tune(config, trials=2)
            model = joblib.load(target / "artifacts" / "model.joblib")
            rows, _ = fixture("c", 1)
            self.assertEqual(len(positive_scores(model, rows, config["features"])), len(rows))
            manifest = json.loads((target / "artifacts" / "model_manifest.json").read_text())
            self.assertEqual(manifest["features"], config["features"])
            self.assertIn(tuned["exported_variant"], ["tuned", "untuned"])
            self.assertTrue((target / "plots" / "cv_macro_f1.png").exists())
            self.assertTrue((target / "plots" / "mlp_training_loss.png").exists())


if __name__ == "__main__": unittest.main()
