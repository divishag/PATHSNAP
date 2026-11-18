"""Viterbi dynamic programming for PathSnap candidate sequences."""

import numpy as np


def viterbi_decode(emission_scores, transition_scores):
    """
    Find the maximum-scoring candidate sequence.

    Parameters
    ----------
    emission_scores:
        List of 1D arrays. Each array contains log emission
        scores for one GPS observation.

    transition_scores:
        List of 2D arrays. Matrix t has shape
        (candidates_at_t, candidates_at_t_plus_1).

    Returns
    -------
    list[int]
        Selected candidate index for each observation.

    Raises
    ------
    ValueError
        If inputs are invalid or no complete path exists.
    """
    if not emission_scores:
        return []

    emissions = [
        np.asarray(group, dtype=float)
        for group in emission_scores
    ]

    for group in emissions:
        if group.ndim != 1 or group.size == 0:
            raise ValueError(
                "Every observation needs at least one candidate"
            )

        if np.isnan(group).any() or np.isposinf(group).any():
            raise ValueError("Invalid emission scores")

    if len(transition_scores) != len(emissions) - 1:
        raise ValueError(
            "Expected one transition matrix between "
            "each consecutive pair of observations"
        )

    scores = emissions[0].copy()
    backpointers = []

    for t, matrix in enumerate(transition_scores):
        matrix = np.asarray(matrix, dtype=float)

        expected_shape = (
            len(emissions[t]),
            len(emissions[t + 1]),
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                f"Transition matrix {t} has shape "
                f"{matrix.shape}; expected {expected_shape}"
            )

        if np.isnan(matrix).any() or np.isposinf(matrix).any():
            raise ValueError("Invalid transition scores")

        combined = scores[:, None] + matrix

        best_previous = np.argmax(combined, axis=0)

        scores = (
            emissions[t + 1]
            + combined[
                best_previous,
                np.arange(len(emissions[t + 1]))
            ]
        )

        backpointers.append(best_previous)

    if not np.isfinite(scores).any():
        raise ValueError(
            "No complete reachable candidate sequence exists"
        )

    final_index = int(np.argmax(scores))
    path = [final_index]

    for pointers in reversed(backpointers):
        final_index = int(pointers[final_index])
        path.append(final_index)

    path.reverse()
    return path
