"""Create candidate labels from reference-route alignment.

This module is for supervised training and evaluation preparation.
It must not be called by the prediction pipeline.
"""

import numpy as np
import pandas as pd


def add_labels(candidates, alignment):
    """
    Join candidate roads with reference-route alignment.

    is_match:
        1 if candidate arc equals aligned reference arc
        0 otherwise
        <NA> if alignment is flagged as uncertain

    label_valid:
        True only when the provisional alignment passes review
        thresholds.

    Note: passing thresholds does not independently verify ground truth.
    """
    required_candidates = {
        "trip_id", "observation_id", "arc_id"
    }
    required_alignment = {
        "trip_id", "observation_id", "reference_arc_id",
        "review_required"
    }

    if not required_candidates.issubset(candidates.columns):
        raise ValueError("Candidate table is missing required columns")

    if not required_alignment.issubset(alignment.columns):
        raise ValueError("Alignment table is missing required columns")

    alignment_info = alignment[
        [
            "trip_id",
            "observation_id",
            "reference_arc_id",
            "review_required",
        ]
    ].copy()

    if alignment_info.duplicated(
        ["trip_id", "observation_id"]
    ).any():
        raise ValueError("Duplicate observation IDs in alignment")

    # Preserve candidate row order and reject unexpected duplicates.
    if candidates.duplicated(
        ["trip_id", "observation_id", "candidate_id"]
    ).any():
        raise ValueError("Duplicate candidate identifiers")

    result = candidates.merge(
        alignment_info,
        on=["trip_id", "observation_id"],
        how="left",
        validate="many_to_one",
        sort=False,
        indicator=True,
    )

    result["label_valid"] = (
        result["_merge"].eq("both")
        & result["review_required"].eq(False).fillna(False)
        & result["reference_arc_id"].notna()
    )

    result["is_match"] = pd.Series(
        pd.NA,
        index=result.index,
        dtype="Int64",
    )

    valid = result["label_valid"]

    result.loc[valid, "is_match"] = (
        result.loc[valid, "arc_id"].to_numpy()
        == result.loc[valid, "reference_arc_id"].to_numpy()
    ).astype(int)

    result = result.drop(columns=["_merge"])

    return result


def build_observation_inventory(trip, alignment, labelled_candidates):
    """
    Preserve every GPS observation, including zero-candidate cases.
    """
    track = trip["track"]

    inventory = pd.DataFrame({
        "trip_id": trip["trip_id"],
        "observation_id": track.index.to_numpy(),
        "timestamp_s": track["timestamp"].to_numpy(),
    })

    inventory = inventory.merge(
        alignment[
            [
                "trip_id",
                "observation_id",
                "reference_arc_id",
                "reference_route_position",
                "alignment_distance_m",
                "alignment_extra_distance_m",
                "review_required",
            ]
        ],
        on=["trip_id", "observation_id"],
        how="left",
        validate="one_to_one",
    )

    counts = (
        labelled_candidates.groupby("observation_id")
        .size()
        .rename("candidate_count")
    )

    inventory["candidate_count"] = (
        inventory["observation_id"]
        .map(counts)
        .fillna(0)
        .astype(int)
    )

    inventory["label_valid"] = (
        inventory["reference_arc_id"].notna()
        & inventory["review_required"].eq(False).fillna(False)
    )

    # Determine whether the reference arc exists among candidates.
    valid_rows = labelled_candidates[
        labelled_candidates["label_valid"]
    ]

    matched_observations = set(
        valid_rows.loc[
            valid_rows["is_match"].eq(1),
            "observation_id"
        ].tolist()
    )

    inventory["reference_in_candidates"] = (
        inventory["observation_id"]
        .isin(matched_observations)
    )

    inventory["candidate_miss"] = (
        inventory["label_valid"]
        & ~inventory["reference_in_candidates"]
    )

    return inventory
