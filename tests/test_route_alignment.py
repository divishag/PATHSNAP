import numpy as np
import pandas as pd
import pytest

from pathsnap.data.route_alignment import align_reference_route


def make_trip():
    return {
        "trip_id": "synthetic",
        "track": pd.DataFrame({
            "longitude": [8.5001, 8.5005, 8.5012, 8.5018],
            "latitude": [50.0, 50.0, 50.0, 50.0],
            "timestamp": [0, 1, 2, 3],
        }),
        "nodes": pd.DataFrame({
            "longitude": [8.5, 8.501, 8.502],
            "latitude": [50.0, 50.0, 50.0],
        }),
        "arcs": pd.DataFrame({
            "from_node_id": [0, 1],
            "to_node_id": [1, 2],
        }),
        "route": pd.DataFrame({
            "arc_id": [0, 1],
        }),
    }


def test_alignment_preserves_order():
    trip = make_trip()
    result = align_reference_route(trip)

    assert len(result) == len(trip["track"])
    assert (
        np.diff(result["reference_route_position"]) >= 0
    ).all()
    assert set(result["reference_arc_id"]).issubset({0, 1})
    assert np.isfinite(result["alignment_distance_m"]).all()


def test_alignment_matches_simple_route():
    result = align_reference_route(make_trip())

    assert result["reference_arc_id"].iloc[0] == 0
    assert result["reference_arc_id"].iloc[-1] == 1


def test_alignment_requires_reference():
    trip = make_trip()
    del trip["route"]

    with pytest.raises(ValueError, match="Reference route"):
        align_reference_route(trip)


def test_alignment_rejects_invalid_arc_ids():
    trip = make_trip()
    trip["route"] = pd.DataFrame({"arc_id": [999]})

    with pytest.raises(ValueError, match="invalid arc IDs"):
        align_reference_route(trip)
