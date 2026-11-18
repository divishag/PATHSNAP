import pandas as pd

from pathsnap.data.validate import validate_trip


def test_validation_detects_invalid_data():
    trip = {
        "trip_id": "sample",
        "track": pd.DataFrame({
            "longitude": [77.0, 200.0, 77.1],
            "latitude": [12.0, 12.1, 12.2],
            "timestamp": [0, 1, 1],
        }),
        "nodes": pd.DataFrame({
            "longitude": [77.0, 77.1],
            "latitude": [12.0, 12.1],
        }),
        "arcs": pd.DataFrame({
            "from_node_id": [0, 0],
            "to_node_id": [1, 99],
        }),
        "route": pd.DataFrame({
            "arc_id": [0, 5],
        }),
    }

    result = validate_trip(trip)

    assert result["invalid_gps_coordinates"] == 1
    assert result["non_increasing_timestamps"] == 1
    assert result["invalid_road_arcs"] == 1
    assert result["invalid_reference_arcs"] == 1


def test_reference_route_is_optional():
    trip = {
        "trip_id": "inference",
        "track": pd.DataFrame({
            "longitude": [77.0],
            "latitude": [12.0],
            "timestamp": [0],
        }),
        "nodes": pd.DataFrame({
            "longitude": [77.0, 77.1],
            "latitude": [12.0, 12.1],
        }),
        "arcs": pd.DataFrame({
            "from_node_id": [0],
            "to_node_id": [1],
        }),
    }

    result = validate_trip(trip)

    assert result["invalid_gps_coordinates"] == 0
    assert "reference_route_length" not in result
