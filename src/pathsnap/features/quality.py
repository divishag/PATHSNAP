"""Quality checks for GPS-to-road candidate training data."""

import numpy as np
import pandas as pd


def assess_observation_quality(
    observations: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    close_road_threshold_m: float = 30.0,
) -> pd.DataFrame:
    """Assess road coverage and label usability without deleting observations.

    This function assumes candidate distances were computed in metres.
    Missing candidates are recorded explicitly rather than treated as
    infinitely distant observations.
    """
    required_obs = {
        "trip_id",
        "observation_id",
        "label_valid",
        "candidate_miss",
    }
    required_candidates = {
        "trip_id",
        "observation_id",
        "distance_to_arc_m",
    }

    missing_obs = required_obs - set(observations.columns)
    missing_candidates = required_candidates - set(candidates.columns)

    if missing_obs or missing_candidates:
        raise ValueError(
            f"Missing columns: observations={sorted(missing_obs)}, "
            f"candidates={sorted(missing_candidates)}"
        )

    if close_road_threshold_m <= 0:
        raise ValueError("close_road_threshold_m must be positive")

    result = observations.copy()

    nearest = (
        candidates.groupby(["trip_id", "observation_id"])[
            "distance_to_arc_m"
        ]
        .min()
        .rename("nearest_candidate_distance_m")
        .reset_index()
    )

    result = result.merge(
        nearest,
        on=["trip_id", "observation_id"],
        how="left",
        validate="one_to_one",
    )

    result["has_candidate"] = (
        result["nearest_candidate_distance_m"].notna()
    )

    result["close_to_network"] = (
        result["has_candidate"]
        & (
            result["nearest_candidate_distance_m"]
            <= close_road_threshold_m
        )
    )

    result["network_quality"] = np.select(
        [
            ~result["has_candidate"],
            result["close_to_network"],
        ],
        [
            "no_candidate",
            "near_road",
        ],
        default="far_from_road",
    )

    valid_alignment = (
        result["label_valid"].fillna(False).astype(bool)
    )

    candidate_miss = (
        result["candidate_miss"].fillna(False).astype(bool)
    )

    result["usable_for_training"] = (
        valid_alignment
        & ~candidate_miss
        & result["has_candidate"]
    )

    result["training_exclusion_reason"] = np.select(
        [
            ~result["has_candidate"],
            ~valid_alignment,
            candidate_miss,
        ],
        [
            "no_candidate",
            "uncertain_reference_alignment",
            "reference_not_in_candidates",
        ],
        default="none",
    )

    return result
