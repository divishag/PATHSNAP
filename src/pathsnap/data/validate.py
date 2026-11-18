import numpy as np
import pandas as pd


def validate_trip(trip):
    """Return data-quality statistics without modifying the trip."""
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    report = {"trip_id": trip["trip_id"]}

    # GPS coordinates
    lon = pd.to_numeric(track["longitude"], errors="coerce")
    lat = pd.to_numeric(track["latitude"], errors="coerce")

    valid_gps = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & lon.between(-180, 180)
        & lat.between(-90, 90)
    )

    report["gps_observations"] = len(track)
    report["invalid_gps_coordinates"] = int((~valid_gps).sum())

    # Timestamps and GPS gaps
    timestamps = pd.to_numeric(track["timestamp"], errors="coerce")
    report["invalid_timestamps"] = int(
        (~np.isfinite(timestamps)).sum()
    )

    gaps = timestamps.diff()
    valid_gaps = gaps[np.isfinite(gaps)]

    report["non_increasing_timestamps"] = int(
        (valid_gaps <= 0).sum()
    )
    report["gps_gaps_over_5s"] = int((valid_gaps > 5).sum())
    report["max_gps_gap_s"] = (
        float(valid_gaps.max()) if not valid_gaps.empty else None
    )

    # Road-network coordinates
    node_lon = pd.to_numeric(nodes["longitude"], errors="coerce")
    node_lat = pd.to_numeric(nodes["latitude"], errors="coerce")

    valid_nodes = (
        np.isfinite(node_lon)
        & np.isfinite(node_lat)
        & node_lon.between(-180, 180)
        & node_lat.between(-90, 90)
    )

    report["road_nodes"] = len(nodes)
    report["invalid_road_nodes"] = int((~valid_nodes).sum())

    # Road-network connectivity references
    from_ids = pd.to_numeric(arcs["from_node_id"], errors="coerce")
    to_ids = pd.to_numeric(arcs["to_node_id"], errors="coerce")

    valid_arcs = (
        np.isfinite(from_ids)
        & np.isfinite(to_ids)
        & from_ids.between(0, len(nodes) - 1)
        & to_ids.between(0, len(nodes) - 1)
    )

    report["road_arcs"] = len(arcs)
    report["invalid_road_arcs"] = int((~valid_arcs).sum())

    # Optional reference route
    if "route" in trip:
        route_ids = pd.to_numeric(
            trip["route"]["arc_id"], errors="coerce"
        )

        valid_route = (
            np.isfinite(route_ids)
            & route_ids.between(0, len(arcs) - 1)
        )

        report["reference_route_length"] = len(route_ids)
        report["invalid_reference_arcs"] = int(
            (~valid_route).sum()
        )

    return report
