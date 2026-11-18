"""Generate road-segment candidates without using reference labels."""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer


def generate_candidates(trip, k=8, search_radius_m=100.0):
    """
    Generate nearby road-segment candidates per GPS observation.

    Uses a KD-tree of road-segment midpoints, supplemented by
    segment length to avoid missing long roads.

    Returns candidate rows with original trip-local arc IDs.
    """
    if k < 1 or search_radius_m <= 0:
        raise ValueError("k and search_radius_m must be positive")

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    if track.empty or nodes.empty or arcs.empty:
        return pd.DataFrame(columns=[
            "trip_id", "observation_id", "timestamp_s",
            "candidate_id", "arc_id", "distance_to_arc_m",
            "arc_length_m"
        ])

    lon0 = float(track["longitude"].median())
    lat0 = float(track["latitude"].median())

    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )

    nx, ny = transformer.transform(
        nodes["longitude"].to_numpy(),
        nodes["latitude"].to_numpy()
    )
    node_xy = np.column_stack((nx, ny))

    gx, gy = transformer.transform(
        track["longitude"].to_numpy(),
        track["latitude"].to_numpy()
    )
    gps_xy = np.column_stack((gx, gy))

    from_ids = arcs["from_node_id"].to_numpy(dtype=int)
    to_ids = arcs["to_node_id"].to_numpy(dtype=int)

    starts = node_xy[from_ids]
    ends = node_xy[to_ids]
    vectors = ends - starts

    lengths = np.linalg.norm(vectors, axis=1)
    midpoints = (starts + ends) / 2
    tree = cKDTree(midpoints)

    max_half_length = float(lengths.max()) / 2

    rows = []

    for obs_id, point in enumerate(gps_xy):
        # Midpoint filtering is conservative: any segment within
        # search_radius_m of the point has its midpoint no farther
        # than search_radius_m + half its length.
        nearby = tree.query_ball_point(
            point, search_radius_m + max_half_length
        )

        if not nearby:
            continue

        indices = np.asarray(nearby, dtype=int)
        seg_starts = starts[indices]
        seg_vectors = vectors[indices]

        squared = np.sum(seg_vectors ** 2, axis=1)

        fraction = np.divide(
            np.sum((point - seg_starts) * seg_vectors, axis=1),
            squared,
            out=np.zeros(len(indices)),
            where=squared > 0
        )

        fraction = np.clip(fraction, 0, 1)
        projected = seg_starts + fraction[:, None] * seg_vectors

        distances = np.linalg.norm(point - projected, axis=1)

        eligible = np.flatnonzero(distances <= search_radius_m)
        if not len(eligible):
            continue

        order = eligible[np.argsort(distances[eligible], kind="stable")[:k]]

        for rank, local_idx in enumerate(order):
            arc_index = int(indices[local_idx])

            rows.append({
                "trip_id": trip["trip_id"],
                "observation_id": int(track.index[obs_id]),
                "timestamp_s": float(track["timestamp"].iloc[obs_id]),
                "candidate_id": rank,
                "arc_id": arc_index,
                "distance_to_arc_m": float(distances[local_idx]),
                "arc_length_m": float(lengths[arc_index]),
            })

    return pd.DataFrame(rows, columns=[
        "trip_id", "observation_id", "timestamp_s",
        "candidate_id", "arc_id", "distance_to_arc_m",
        "arc_length_m"
    ])
