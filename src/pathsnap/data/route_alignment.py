"""Reference-route alignment for training and evaluation only.

Never call this module inside the model's prediction pipeline.
"""

import numpy as np
import pandas as pd
from pyproj import Transformer


def align_reference_route(trip, skip_penalty=0.5,
                          review_distance_m=20.0,
                          review_extra_m=10.0):
    """
    Align GPS observations to an ordered reference arc sequence.

    Uses metric point-to-segment distances and dynamic programming.
    The alignment is provisional and must be validated before
    generating supervised training labels.
    """
    if "route" not in trip:
        raise ValueError("Reference route required for alignment")

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route_ids = trip["route"]["arc_id"].to_numpy(dtype=int)

    if len(track) == 0 or len(route_ids) == 0:
        raise ValueError("Track and reference route must be nonempty")

    if skip_penalty < 0:
        raise ValueError("skip_penalty must be nonnegative")

    if (route_ids < 0).any() or (route_ids >= len(arcs)).any():
        raise ValueError("Reference route contains invalid arc IDs")

    # Local metric coordinate system appropriate for this dataset
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32632", always_xy=True
    )

    gps = np.column_stack(transformer.transform(
        track["longitude"].to_numpy(dtype=float),
        track["latitude"].to_numpy(dtype=float),
    ))

    node_xy = np.column_stack(transformer.transform(
        nodes["longitude"].to_numpy(dtype=float),
        nodes["latitude"].to_numpy(dtype=float),
    ))

    selected_arcs = arcs.iloc[route_ids]
    start_ids = selected_arcs["from_node_id"].to_numpy(dtype=int)
    end_ids = selected_arcs["to_node_id"].to_numpy(dtype=int)

    starts = node_xy[start_ids]
    ends = node_xy[end_ids]
    vectors = ends - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)

    n_obs = len(gps)
    n_arcs = len(route_ids)
    positions = np.arange(n_arcs)

    # Store route-position backpointers for Viterbi-style backtracking.
    backpointers = np.zeros((n_obs, n_arcs), dtype=np.int32)

    def segment_distances(point):
        offsets = point - starts

        fractions = np.divide(
            np.einsum("ij,ij->i", offsets, vectors),
            lengths_sq,
            out=np.zeros(n_arcs, dtype=float),
            where=lengths_sq > 0,
        )

        fractions = np.clip(fractions, 0.0, 1.0)
        projected = starts + fractions[:, None] * vectors

        return np.linalg.norm(point - projected, axis=1)

    if not np.isfinite(gps).all() or not np.isfinite(node_xy).all():
        raise ValueError("Invalid coordinates: validate trip first")

    # Each observation can stay at its current route position or move
    # forward, but cannot go backward in reference-route order.
    previous_cost = segment_distances(gps[0])

    for t in range(1, n_obs):
        distances = segment_distances(gps[t])

        # Prefix minima give the best predecessor k <= j.
        adjusted = previous_cost - skip_penalty * positions

        best_cost = np.empty(n_arcs, dtype=float)
        best_index = np.empty(n_arcs, dtype=np.int32)

        running_cost = float("inf")
        running_index = 0

        for j in range(n_arcs):
            if adjusted[j] < running_cost:
                running_cost = adjusted[j]
                running_index = j

            best_cost[j] = running_cost
            best_index[j] = running_index

        previous_cost = (
            distances + best_cost + skip_penalty * positions
        )
        backpointers[t] = best_index

    # Recover the selected route position for each GPS observation.
    aligned_positions = np.empty(n_obs, dtype=np.int32)
    aligned_positions[-1] = int(np.argmin(previous_cost))

    for t in range(n_obs - 1, 0, -1):
        aligned_positions[t - 1] = backpointers[
            t, aligned_positions[t]
        ]

    aligned_distances = np.empty(n_obs, dtype=float)
    nearest_distances = np.empty(n_obs, dtype=float)

    for t, point in enumerate(gps):
        distances = segment_distances(point)
        aligned_distances[t] = distances[aligned_positions[t]]
        nearest_distances[t] = distances.min()

    extra_distances = aligned_distances - nearest_distances

    # Diagnostic flags, not certified ground-truth labels.
    review_required = (
        (aligned_distances > review_distance_m)
        | (extra_distances > review_extra_m)
    )

    return pd.DataFrame({
        "trip_id": trip["trip_id"],
        "observation_id": track.index.to_numpy(),
        "reference_route_position": aligned_positions,
        "reference_arc_id": route_ids[aligned_positions],
        "alignment_distance_m": aligned_distances,
        "nearest_reference_distance_m": nearest_distances,
        "alignment_extra_distance_m": extra_distances,
        "review_required": review_required,
    })
