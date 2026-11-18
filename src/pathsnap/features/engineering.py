"""Create numerical features for candidate road classification."""

import numpy as np
from pyproj import CRS, Transformer


FEATURE_COLUMNS = [
    "distance_to_arc_m",
    "arc_length_m",
    "heading_difference_deg",
    "speed_mps",
    "time_gap_s",
    "candidate_rank",
]


def add_features(trip, candidates):
    """
    Add movement and road-direction features.

    The function must work without access to reference routes.
    """
    result = candidates.copy()

    if result.empty:
        for name in FEATURE_COLUMNS:
            if name not in result.columns:
                result[name] = np.array([], dtype=float)
        return result

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    lon0 = float(track["longitude"].median())
    lat0 = float(track["latitude"].median())

    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )

    # Project GPS coordinates to metres.
    gx, gy = transformer.transform(
        track["longitude"].to_numpy(dtype=float),
        track["latitude"].to_numpy(dtype=float),
    )
    gps_xy = np.column_stack((gx, gy))

    # Project road nodes using the same coordinate system.
    nx, ny = transformer.transform(
        nodes["longitude"].to_numpy(dtype=float),
        nodes["latitude"].to_numpy(dtype=float),
    )
    node_xy = np.column_stack((nx, ny))

    # GPS movement vectors.
    gps_vectors = np.zeros_like(gps_xy)
    gps_vectors[1:] = gps_xy[1:] - gps_xy[:-1]

    timestamps = track["timestamp"].to_numpy(dtype=float)

    time_gaps = np.zeros(len(track), dtype=float)
    time_gaps[1:] = np.diff(timestamps)

    valid_motion = (
        (time_gaps > 0)
        & (time_gaps <= 5)
        & (np.linalg.norm(gps_vectors, axis=1) >= 1.0)
    )

    speed = np.full(len(track), np.nan)
    step_distance = np.linalg.norm(gps_vectors, axis=1)

    speed[valid_motion] = (
        step_distance[valid_motion] / time_gaps[valid_motion]
    )

    # Candidate road vectors.
    arc_ids = result["arc_id"].to_numpy(dtype=int)
    chosen_arcs = arcs.iloc[arc_ids]

    start_xy = node_xy[
        chosen_arcs["from_node_id"].to_numpy(dtype=int)
    ]
    end_xy = node_xy[
        chosen_arcs["to_node_id"].to_numpy(dtype=int)
    ]

    road_vectors = end_xy - start_xy
    road_norm = np.linalg.norm(road_vectors, axis=1)

    obs_ids = result["observation_id"].to_numpy(dtype=int)
    movement_vectors = gps_vectors[obs_ids]
    movement_norm = np.linalg.norm(movement_vectors, axis=1)

    dot = np.sum(movement_vectors * road_vectors, axis=1)

    cosine = np.divide(
        dot,
        movement_norm * road_norm,
        out=np.full(len(result), np.nan),
        where=(movement_norm > 0) & (road_norm > 0)
    )

    angles = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    # Heading is unreliable at the start, after GPS gaps,
    # or when displacement is too small.
    angles[~valid_motion[obs_ids]] = np.nan

    result["heading_difference_deg"] = angles
    result["speed_mps"] = speed[obs_ids]
    result["time_gap_s"] = time_gaps[obs_ids]
    result["candidate_rank"] = result["candidate_id"].astype(int)

    return result
