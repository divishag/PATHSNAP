"""Create numerical features for candidate road classification.

Every feature here is computed from GPS observations and the road
network only. Reference routes are never read, so the same function is
used unchanged by the training pipeline and the API.

Design notes (see reports/feature_engineering_v2.md):

* ~90% of arcs have a reverse-direction twin with identical geometry,
  so distance alone cannot separate them. Direction must come from GPS
  motion. A single 1-second step is too noisy, so direction is measured
  over centred windows of +/-2, +/-5 and +/-10 observations.
* Absolute values (distance, heading) vary a lot between trips because
  GPS noise differs. Features relative to the other candidates of the
  same observation (distance minus the best distance, cosine minus the
  best cosine) transfer better across trips.
* A candidate that stays close to the GPS trace over neighbouring
  observations is more plausible than one that is close only at one
  instant (parallel streets, crossings).
"""

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


FEATURE_COLUMNS = [
    # Geometry of the candidate relative to the GPS point
    "distance_to_arc_m",
    "distance_delta_m",
    "distance_ratio",
    "distance_rank_tied",
    "projection_fraction",
    "endpoint_overshoot_m",
    "arc_length_m",
    # Direction agreement (multi-scale, signed)
    "align_cos_w2",
    "align_cos_w5",
    "align_cos_w10",
    "align_cos_delta_w5",
    "along_track_w5_m",
    "heading_difference_deg",
    # Temporal context along the trace
    "context_dist_w3_m",
    "context_dist_w10_m",
    "context_dist_delta_w10_m",
    # Observation / network context
    "speed_mps",
    "time_gap_s",
    "n_candidates",
    "has_twin_candidate",
    "to_node_out_degree",
]

# Kept so the report can compare the original six-feature baseline.
BASELINE_FEATURE_COLUMNS = [
    "distance_to_arc_m",
    "arc_length_m",
    "heading_difference_deg_v1",
    "speed_mps_v1",
    "time_gap_s",
    "candidate_rank",
]

MIN_DISPLACEMENT_M = 2.0
KEY = ["trip_id", "observation_id"]


def _project(trip):
    track = trip["track"]
    nodes = trip["nodes"]
    lon0 = float(track["longitude"].median())
    lat0 = float(track["latitude"].median())
    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    gps = np.column_stack(transformer.transform(
        track["longitude"].to_numpy(float), track["latitude"].to_numpy(float)
    ))
    node_xy = np.column_stack(transformer.transform(
        nodes["longitude"].to_numpy(float), nodes["latitude"].to_numpy(float)
    ))
    return gps, node_xy


def _segment_distance(points, starts, vectors, sq_len):
    """Point-to-segment distance and unclipped projection fraction."""
    offsets = points - starts
    raw = np.divide(
        np.einsum("ij,ij->i", offsets, vectors), sq_len,
        out=np.zeros(len(points)), where=sq_len > 0,
    )
    clipped = np.clip(raw, 0.0, 1.0)
    projected = starts + clipped[:, None] * vectors
    return np.linalg.norm(points - projected, axis=1), raw


def _window_displacement(gps, positions, w):
    n = len(gps)
    lo = np.clip(positions - w, 0, n - 1)
    hi = np.clip(positions + w, 0, n - 1)
    return gps[hi] - gps[lo], lo, hi


def _cosine(disp, unit):
    norm = np.linalg.norm(disp, axis=1)
    cos = np.einsum("ij,ij->i", disp, unit) / np.maximum(norm, 1e-9)
    # Too little movement: direction is unknown, not "perpendicular".
    cos[norm < MIN_DISPLACEMENT_M] = 0.0
    return cos


def _baseline_features(result, gps, timestamps, positions, road_vec):
    """The original v1 single-step heading/speed, for comparison only."""
    step = np.zeros_like(gps)
    step[1:] = gps[1:] - gps[:-1]
    gaps = np.zeros(len(gps))
    gaps[1:] = np.diff(timestamps)
    dist = np.linalg.norm(step, axis=1)
    valid = (gaps > 0) & (gaps <= 5) & (dist >= 1.0)
    speed = np.full(len(gps), np.nan)
    speed[valid] = dist[valid] / gaps[valid]
    mv = step[positions]
    cos = np.divide(
        np.einsum("ij,ij->i", mv, road_vec),
        np.linalg.norm(mv, axis=1) * np.linalg.norm(road_vec, axis=1),
        out=np.full(len(positions), np.nan),
        where=(np.linalg.norm(mv, axis=1) > 0) & (np.linalg.norm(road_vec, axis=1) > 0),
    )
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang[~valid[positions]] = np.nan
    result["heading_difference_deg_v1"] = ang
    result["speed_mps_v1"] = speed[positions]


def add_features(trip, candidates):
    """Add geometry, direction and context features to candidate rows."""
    result = candidates.copy()
    all_columns = FEATURE_COLUMNS + [
        c for c in BASELINE_FEATURE_COLUMNS if c not in FEATURE_COLUMNS
    ]
    if result.empty:
        for name in all_columns:
            result[name] = np.array([], dtype=float)
        return result

    track = trip["track"]
    arcs = trip["arcs"]
    gps, node_xy = _project(trip)
    timestamps = track["timestamp"].to_numpy(float)
    n_obs = len(gps)

    # observation_id is the 0-based track row index (see loader).
    positions = result["observation_id"].to_numpy(int)
    arc_ids = result["arc_id"].to_numpy(int)
    from_nodes = arcs["from_node_id"].to_numpy(int)
    to_nodes = arcs["to_node_id"].to_numpy(int)

    starts = node_xy[from_nodes[arc_ids]]
    ends = node_xy[to_nodes[arc_ids]]
    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    sq_len = lengths ** 2
    unit = vectors / np.maximum(lengths, 1e-9)[:, None]

    # --- geometry -----------------------------------------------------
    points = gps[positions]
    dist, raw_fraction = _segment_distance(points, starts, vectors, sq_len)
    result["distance_to_arc_m"] = dist
    result["arc_length_m"] = lengths
    result["projection_fraction"] = np.clip(raw_fraction, 0.0, 1.0)
    overshoot = np.maximum(raw_fraction - 1.0, 0.0) + np.maximum(-raw_fraction, 0.0)
    result["endpoint_overshoot_m"] = overshoot * lengths

    group = result.groupby(KEY, sort=False)
    min_dist = group["distance_to_arc_m"].transform("min").to_numpy()
    result["distance_delta_m"] = dist - min_dist
    result["distance_ratio"] = (dist + 1.0) / (min_dist + 1.0)
    # Twins have identical distance: give them the same rank instead of
    # an arbitrary 0/1 order decided by arc index.
    result["_d_round"] = np.round(dist, 1)
    result["distance_rank_tied"] = (
        result.groupby(KEY, sort=False)["_d_round"]
        .rank(method="dense").to_numpy() - 1
    )
    result = result.drop(columns="_d_round")
    if "candidate_id" in result:
        result["candidate_rank"] = result["candidate_id"].astype(int)

    # --- direction ----------------------------------------------------
    for w in (2, 5, 10):
        disp, _, _ = _window_displacement(gps, positions, w)
        result[f"align_cos_w{w}"] = _cosine(disp, unit)
        if w == 5:
            result["along_track_w5_m"] = np.einsum("ij,ij->i", disp, unit)
    result["align_cos_delta_w5"] = (
        result["align_cos_w5"]
        - result.groupby(KEY, sort=False)["align_cos_w5"].transform("max")
    )
    disp3, _, _ = _window_displacement(gps, positions, 3)
    cos3 = np.einsum("ij,ij->i", disp3, unit) / np.maximum(
        np.linalg.norm(disp3, axis=1), 1e-9)
    heading = np.degrees(np.arccos(np.clip(cos3, -1, 1)))
    heading[np.linalg.norm(disp3, axis=1) < MIN_DISPLACEMENT_M] = np.nan
    result["heading_difference_deg"] = heading

    # --- temporal context: mean distance of neighbouring fixes --------
    for w in (3, 10):
        total = np.zeros(len(result))
        count = np.zeros(len(result))
        for offset in range(-w, w + 1):
            idx = positions + offset
            ok = (idx >= 0) & (idx < n_obs)
            d, _ = _segment_distance(
                gps[np.clip(idx, 0, n_obs - 1)], starts, vectors, sq_len)
            total += np.where(ok, d, 0.0)
            count += ok
        result[f"context_dist_w{w}_m"] = total / np.maximum(count, 1)
    result["context_dist_delta_w10_m"] = (
        result["context_dist_w10_m"]
        - result.groupby(KEY, sort=False)["context_dist_w10_m"].transform("min")
    )

    # --- observation-level motion ------------------------------------
    disp2, lo, hi = _window_displacement(gps, np.arange(n_obs), 2)
    dt = timestamps[hi] - timestamps[lo]
    speed = np.divide(np.linalg.norm(disp2, axis=1), dt,
                      out=np.full(n_obs, np.nan), where=dt > 0)
    gaps = np.zeros(n_obs)
    gaps[1:] = np.diff(timestamps)
    result["speed_mps"] = speed[positions]
    result["time_gap_s"] = gaps[positions]
    result["n_candidates"] = group["arc_id"].transform("size").to_numpy()

    # --- network context ---------------------------------------------
    pair_to_arc = pd.Series(
        np.arange(len(arcs)),
        index=pd.MultiIndex.from_arrays([from_nodes, to_nodes]),
    )
    pair_to_arc = pair_to_arc[~pair_to_arc.index.duplicated()]
    twin = pair_to_arc.reindex(
        pd.MultiIndex.from_arrays([to_nodes[arc_ids], from_nodes[arc_ids]])
    ).to_numpy()
    candidate_pairs = set(zip(positions.tolist(), arc_ids.tolist()))
    result["has_twin_candidate"] = np.array([
        (not np.isnan(t)) and ((p, int(t)) in candidate_pairs)
        for p, t in zip(positions, twin)
    ], dtype=int)
    out_degree = np.bincount(from_nodes, minlength=len(node_xy))
    result["to_node_out_degree"] = out_degree[to_nodes[arc_ids]]

    _baseline_features(result, gps, timestamps, positions, vectors)
    return result
