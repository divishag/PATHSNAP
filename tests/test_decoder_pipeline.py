"""Integration tests for the PathSnap decoder."""

import numpy as np
import pandas as pd

from pathsnap.decoding.pipeline import decode_trip


def make_trip():
    return {
        "trip_id": "synthetic",
        "nodes": pd.DataFrame({
            "longitude": [8.0, 8.001, 8.002, 8.003],
            "latitude": [49.0, 49.0, 49.0, 49.0],
        }),
        "arcs": pd.DataFrame({
            "from_node_id": [0, 1, 2],
            "to_node_id": [1, 2, 3],
        }),
        "track": pd.DataFrame({
            "longitude": [8.0005, 8.0015, 8.0025],
            "latitude": [49.0, 49.0, 49.0],
            "timestamp": [1.0, 2.0, 3.0],
        }),
    }


def test_decodes_connected_trip():
    trip = make_trip()

    candidates = pd.DataFrame({
        "trip_id": ["synthetic"] * 3,
        "observation_id": [0, 1, 2],
        "candidate_id": [0, 0, 0],
        "arc_id": [0, 1, 2],
        "match_probability": [0.9, 0.9, 0.9],
    })

    result = decode_trip(trip, candidates)

    assert result["predicted_arc_id"].tolist() == [0, 1, 2]
    assert (result["decode_status"] == "decoded").all()


def test_missing_candidate_observation():
    trip = make_trip()

    candidates = pd.DataFrame({
        "trip_id": ["synthetic", "synthetic"],
        "observation_id": [0, 2],
        "candidate_id": [0, 0],
        "arc_id": [0, 2],
        "match_probability": [0.9, 0.8],
    })

    result = decode_trip(trip, candidates)

    assert len(result) == 3
    assert result.loc[1, "decode_status"] == "no_candidates"
    assert pd.isna(result.loc[1, "predicted_arc_id"])


def test_empty_candidates():
    trip = make_trip()

    candidates = pd.DataFrame(columns=[
        "trip_id", "observation_id", "candidate_id",
        "arc_id", "match_probability",
    ])

    result = decode_trip(trip, candidates)

    assert len(result) == 3
    assert (result["decode_status"] == "no_candidates").all()
