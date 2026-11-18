"""Tests for directed road-network transitions."""

import numpy as np
import pandas as pd

from pathsnap.decoding.transitions import RoadTransitionModel


def make_trip():
    return {
        "trip_id": "synthetic",
        "nodes": pd.DataFrame({
            "longitude": [8.0, 8.001, 8.002, 8.003],
            "latitude": [49.0, 49.0, 49.0, 49.001],
        }),
        "arcs": pd.DataFrame({
            "from_node_id": [0, 1, 2],
            "to_node_id": [1, 2, 3],
        }),
        "track": pd.DataFrame({
            "longitude": [8.0005, 8.0015, 8.0025],
            "latitude": [49.0, 49.0, 49.0005],
            "timestamp": [1.0, 2.0, 3.0],
        }),
    }


def test_forward_connected_transition():
    model = RoadTransitionModel(make_trip())

    distance = model.network_distance(0, 0, 1, 1)

    assert np.isfinite(distance)
    assert distance > 0


def test_same_arc_forward_transition():
    trip = make_trip()
    trip["track"].loc[1, "longitude"] = 8.0008

    model = RoadTransitionModel(trip)

    distance = model.network_distance(0, 0, 1, 0)

    assert distance > 0
    assert distance < model.lengths[0]


def test_reverse_unreachable_transition():
    model = RoadTransitionModel(make_trip())

    # No directed path from arc 2 back to arc 0.
    distance = model.network_distance(2, 2, 0, 0)

    assert np.isinf(distance)


def test_transition_score_is_nonpositive():
    model = RoadTransitionModel(make_trip())

    score = model.transition_log_score(0, 0, 1, 1)

    assert np.isfinite(score)
    assert score <= 0


def test_invalid_beta_rejected():
    import pytest

    with pytest.raises(ValueError):
        RoadTransitionModel(make_trip(), beta_m=0)
