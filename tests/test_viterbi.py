"""Tests for Viterbi dynamic programming."""

import numpy as np
import pytest

from pathsnap.decoding.viterbi import viterbi_decode


def test_single_observation():
    path = viterbi_decode(
        [np.log([0.2, 0.8])],
        [],
    )

    assert path == [1]


def test_prefers_consistent_path():
    emissions = [
        np.log([0.9, 0.1]),
        np.log([0.4, 0.6]),
        np.log([0.9, 0.1]),
    ]

    transitions = [
        np.array([
            [0.0, -5.0],
            [-5.0, 0.0],
        ]),
        np.array([
            [0.0, -5.0],
            [-5.0, 0.0],
        ]),
    ]

    path = viterbi_decode(emissions, transitions)

    assert path == [0, 0, 0]


def test_disconnected_transition_rejected():
    emissions = [
        np.log([0.8]),
        np.log([0.9]),
    ]

    transitions = [
        np.array([[-np.inf]]),
    ]

    with pytest.raises(ValueError):
        viterbi_decode(emissions, transitions)


def test_variable_candidate_counts():
    emissions = [
        np.log([0.6, 0.4]),
        np.log([0.2, 0.3, 0.5]),
        np.log([0.7]),
    ]

    transitions = [
        np.zeros((2, 3)),
        np.zeros((3, 1)),
    ]

    path = viterbi_decode(emissions, transitions)

    assert path == [0, 2, 0]


def test_invalid_transition_shape():
    emissions = [
        np.log([0.5, 0.5]),
        np.log([0.5, 0.5]),
    ]

    transitions = [
        np.zeros((3, 2)),
    ]

    with pytest.raises(ValueError):
        viterbi_decode(emissions, transitions)


def test_empty_sequence():
    assert viterbi_decode([], []) == []
