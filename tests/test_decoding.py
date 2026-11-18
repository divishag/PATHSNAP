"""Tests for PathSnap HMM scoring and decoding."""

import numpy as np
import pytest

from pathsnap.decoding.hmm import emission_log_scores


def test_emission_scores_preserve_ranking():
    probabilities = [0.75, 0.20, 0.05]

    scores = emission_log_scores(probabilities)

    assert scores[0] > scores[1] > scores[2]
    assert np.allclose(scores, np.log(probabilities))


def test_zero_probability_is_finite():
    scores = emission_log_scores([0.0, 1.0])

    assert np.isfinite(scores).all()
    assert scores[0] < scores[1]


def test_invalid_probabilities_rejected():
    with pytest.raises(ValueError):
        emission_log_scores([-0.1, 0.8])

    with pytest.raises(ValueError):
        emission_log_scores([0.5, 1.2])

    with pytest.raises(ValueError):
        emission_log_scores([float("nan"), 0.5])


def test_empty_probabilities_rejected():
    with pytest.raises(ValueError):
        emission_log_scores([])
