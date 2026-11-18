"""Load a trained PATHSNAP classifier and score road candidates."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


from pathsnap.features.engineering import FEATURE_COLUMNS

FEATURES = list(FEATURE_COLUMNS)


def load_candidate_model(model_path):
    """Load a trusted, locally trained classifier."""
    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")

    model = joblib.load(path)

    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba")

    if not hasattr(model, "classes_"):
        raise TypeError("Model must expose classes_")

    if 1 not in model.classes_:
        raise ValueError("Model has no positive class (1)")

    return model


def score_candidates(model, candidates):
    """Add P(is_match=1) to candidate rows without changing their order."""
    missing = set(FEATURES) - set(candidates.columns)

    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    scored = candidates.copy()

    if scored.empty:
        scored["match_probability"] = pd.Series(dtype=float)
        return scored

    positive_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(
        scored[FEATURES]
    )[:, positive_index]

    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced non-finite probabilities")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model produced invalid probabilities")

    scored["match_probability"] = probabilities

    return scored
