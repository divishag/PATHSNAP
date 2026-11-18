"""Candidate and observation metrics; reference inventory prevents survivor bias."""
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss, confusion_matrix

KEY = ["trip_id", "observation_id"]
METADATA = set(KEY + ["candidate_id", "arc_id", "timestamp_s", "is_match", "label_valid", "reference_arc_id", "partition"])


def load_tables(candidates_path, observations_path, features):
    candidates = pd.read_csv(candidates_path, dtype={"trip_id": str})
    observations = pd.read_csv(observations_path, dtype={"trip_id": str})
    validate_tables(candidates, observations, features)
    return candidates, observations


def validate_tables(candidates, observations, features):
    if not features or len(set(features)) != len(features) or set(features) & METADATA:
        raise ValueError("Provide unique, explicit predictor columns, excluding IDs and reference metadata.")
    required = set(KEY + ["candidate_id", "arc_id", "is_match"]) | set(features)
    if not required <= set(candidates):
        raise ValueError(f"Missing candidate columns: {sorted(required - set(candidates))}")
    if not set(KEY + ["label_valid", "reference_arc_id"]) <= set(observations):
        raise ValueError("Observation inventory needs trip_id, observation_id, label_valid, reference_arc_id.")
    if observations.empty or candidates.empty:
        raise ValueError("Candidate table and observation inventory must be nonempty.")
    if observations[KEY].isna().any().any() or candidates[KEY + ["candidate_id", "arc_id"]].isna().any().any():
        raise ValueError("Identity columns must not be missing.")
    if observations.duplicated(KEY).any() or candidates.duplicated(KEY + ["candidate_id"]).any():
        raise ValueError("Duplicate observation/candidate identity.")
    if candidates.duplicated(KEY + ["arc_id"]).any():
        raise ValueError("Duplicate arc candidates for an observation.")
    if not observations.label_valid.isin([0, 1]).all():
        raise ValueError("label_valid must be encoded as 0/1.")
    valid = observations[observations.label_valid.eq(1)]
    if valid.empty or valid.reference_arc_id.isna().any():
        raise ValueError("Valid observations require reference_arc_id; at least one valid label is needed.")
    joined = candidates.merge(observations, on=KEY, how="left", validate="many_to_one", indicator=True, suffixes=("", "_inventory"))
    if not joined._merge.eq("both").all():
        raise ValueError("Candidate observations are absent from the observation inventory.")
    usable = joined.label_valid_inventory.eq(1) if "label_valid_inventory" in joined else joined.label_valid.eq(1)
    rows = joined.loc[usable]
    if not rows.is_match.isin([0, 1]).all():
        raise ValueError("Valid observations need binary candidate is_match labels.")
    if not rows.is_match.eq(rows.arc_id.eq(rows.reference_arc_id).astype(int)).all():
        raise ValueError("Candidate labels disagree with reference arc alignment.")
    for feature in features:
        candidates[feature] = pd.to_numeric(candidates[feature], errors="raise")
    if np.isinf(candidates[features].to_numpy(dtype=float)).any():
        raise ValueError("Infinite predictors are invalid; use NaN for documented missing data.")


def labelled_rows(candidates, observations):
    keys = observations.loc[observations.label_valid.eq(1), KEY]
    return candidates.merge(keys, on=KEY, how="inner", validate="many_to_one")


def positive_scores(model, frame, features):
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Trained classifier has no positive class 1.")
    if frame.empty:
        return np.array([], dtype=float)
    scores = np.asarray(model.predict_proba(frame[features]))[:, classes.index(1)]
    if not np.isfinite(scores).all():
        raise ValueError("Model returned nonfinite probabilities.")
    return np.clip(scores, 1e-12, 1 - 1e-12)


def evaluate_scores(candidates, observations, scores):
    frame = candidates.copy().reset_index(drop=True)
    if len(scores) != len(frame):
        raise ValueError("Scores must preserve candidate row count and order.")
    frame["score"] = scores
    if not np.isfinite(frame.score).all() or not frame.score.between(0, 1).all():
        raise ValueError("Candidate scores must be finite probabilities between 0 and 1.")
    usable = labelled_rows(frame, observations)
    y = usable.is_match.to_numpy(dtype=int)
    prediction = usable.score.ge(.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, labels=[0, 1], average="macro", zero_division=0)
    winners = frame.loc[frame.groupby(KEY, sort=False).score.idxmax(), KEY + ["arc_id"]]
    reference = observations.loc[observations.label_valid.eq(1)]
    compared = reference.merge(winners, on=KEY, how="left", validate="one_to_one")
    covered_keys = frame.loc[frame.is_match.eq(1), KEY].drop_duplicates().assign(candidate_hit=1)
    coverage = reference.merge(covered_keys, on=KEY, how="left")
    return {
        "candidate_accuracy": float(accuracy_score(y, prediction)) if len(y) else None,
        "candidate_macro_precision": float(precision) if len(y) else None,
        "candidate_macro_recall": float(recall) if len(y) else None,
        "candidate_macro_f1": float(f1) if len(y) else None,
        "candidate_log_loss": float(log_loss(y, np.column_stack([1 - usable.score, usable.score]), labels=[0, 1])) if len(y) else None,
        "candidate_brier_score": float(np.mean((usable.score.to_numpy() - y) ** 2)) if len(y) else None,
        "candidate_confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "top1_accuracy": float(compared.arc_id.eq(compared.reference_arc_id).mean()),
        "candidate_miss_rate": float(coverage.candidate_hit.isna().mean()),
        "labelled_observations": len(reference),
        "unlabelled_observations": int(observations.label_valid.ne(1).sum()),
        "candidate_rows": len(frame),
        "route_continuity": None,
        "route_continuity_note": "N/A: requires Laptop 1 road graph and agreed transition definition.",
    }


def evaluate_model(model, candidates, observations, features):
    start = perf_counter()
    scores = positive_scores(model, candidates, features)
    elapsed = (perf_counter() - start) * 1000
    return {**evaluate_scores(candidates, observations, scores), "inference_time_ms": elapsed}
