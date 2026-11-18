"""Trip-level HMM/Viterbi decoder for PathSnap."""

import numpy as np
import pandas as pd

from pathsnap.decoding.hmm import emission_log_scores
from pathsnap.decoding.transitions import RoadTransitionModel
from pathsnap.decoding.viterbi import viterbi_decode


def decode_trip(
    trip,
    candidates,
    probability_column="match_probability",
    beta_m=200.0,  # chosen by scripts/tune_decoder.py sweep (results/decoder/)
):
    """Decode candidate probabilities into a sequence of road arcs.

    Disconnected transitions start a new independent decoding segment.
    Observations with no candidates remain unassigned.
    """
    required = {
        "trip_id", "observation_id", "candidate_id",
        "arc_id", probability_column,
    }

    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")

    if candidates.duplicated(
        ["observation_id", "candidate_id"]
    ).any():
        raise ValueError("Duplicate candidate IDs within an observation")

    if not candidates.empty and not (
        candidates["trip_id"].astype(str) == str(trip["trip_id"])
    ).all():
        raise ValueError("Candidate trip_id does not match the trip")

    model = RoadTransitionModel(trip, beta_m=beta_m)

    grouped = {
        int(obs_id): group.sort_values("candidate_id")
        for obs_id, group in candidates.groupby("observation_id")
    }

    output = []
    segment_id = 0
    pending_groups = []
    pending_emissions = []
    pending_transitions = []

    def flush_segment():
        nonlocal segment_id

        if not pending_groups:
            return

        chosen = viterbi_decode(
            pending_emissions,
            pending_transitions,
        )

        for group, candidate_index in zip(
            pending_groups, chosen
        ):
            row = group.iloc[candidate_index]

            output.append({
                "trip_id": str(trip["trip_id"]),
                "observation_id": int(row["observation_id"]),
                "predicted_arc_id": int(row["arc_id"]),
                "selected_candidate_id": int(row["candidate_id"]),
                "decode_segment_id": segment_id,
                "decode_status": "decoded",
            })

        segment_id += 1
        pending_groups.clear()
        pending_emissions.clear()
        pending_transitions.clear()

    for obs_id in trip["track"].index:
        obs_id = int(obs_id)
        group = grouped.get(obs_id)

        if group is None or group.empty:
            flush_segment()

            output.append({
                "trip_id": str(trip["trip_id"]),
                "observation_id": obs_id,
                "predicted_arc_id": pd.NA,
                "selected_candidate_id": pd.NA,
                "decode_segment_id": pd.NA,
                "decode_status": "no_candidates",
            })
            continue

        probabilities = group[
            probability_column
        ].to_numpy(dtype=float)

        emissions = emission_log_scores(probabilities)

        if pending_groups:
            previous = pending_groups[-1]

            matrix = np.empty(
                (len(previous), len(group)),
                dtype=float,
            )

            previous_obs_id = int(
                previous["observation_id"].iloc[0]
            )

            for i, prev_arc in enumerate(
                previous["arc_id"].to_numpy(dtype=int)
            ):
                for j, curr_arc in enumerate(
                    group["arc_id"].to_numpy(dtype=int)
                ):
                    matrix[i, j] = model.transition_log_score(
                        previous_obs_id,
                        prev_arc,
                        obs_id,
                        curr_arc,
                    )

            # If every candidate pair is disconnected,
            # start a new decoding segment.
            if not np.isfinite(matrix).any():
                flush_segment()
            else:
                # A locally possible transition may still be
                # unreachable from the start of the current segment.
                # Check the accumulated Viterbi state before extending.
                scores = pending_emissions[0].copy()

                for step, previous_matrix in enumerate(
                    pending_transitions
                ):
                    combined = scores[:, None] + previous_matrix
                    scores = (
                        pending_emissions[step + 1]
                        + np.max(combined, axis=0)
                    )

                next_scores = (
                    emissions
                    + np.max(scores[:, None] + matrix, axis=0)
                )

                if not np.isfinite(next_scores).any():
                    flush_segment()
                else:
                    pending_transitions.append(matrix)

        pending_groups.append(group)
        pending_emissions.append(emissions)

    flush_segment()

    result = pd.DataFrame(output)

    return result.sort_values(
        "observation_id"
    ).reset_index(drop=True)
