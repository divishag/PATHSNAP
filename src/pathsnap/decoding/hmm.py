"""HMM scoring utilities for PathSnap map matching."""

import numpy as np


def emission_log_scores(probabilities, epsilon=1e-12):
    """
    Convert candidate-match probabilities into log scores.

    Parameters
    ----------
    probabilities:
        One-dimensional array of candidate probabilities.
    epsilon:
        Lower clipping threshold to avoid log(0).

    Returns
    -------
    numpy.ndarray
        Log scores with the same shape as the input.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if probabilities.ndim != 1:
        raise ValueError("Probabilities must be one-dimensional")

    if probabilities.size == 0:
        raise ValueError("At least one probability is required")

    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1")

    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between 0 and 1")

    return np.log(np.clip(probabilities, epsilon, 1.0))
