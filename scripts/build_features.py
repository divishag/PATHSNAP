"""Export PathSnap candidate features and provisional labels."""

import argparse
import json
from pathlib import Path

import pandas as pd

from pathsnap.data.loader import read_trip
from pathsnap.data.validate import validate_trip
from pathsnap.data.route_alignment import align_reference_route
from pathsnap.features.quality import assess_observation_quality
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.engineering import add_features, FEATURE_COLUMNS
from pathsnap.features.labels import (
    add_labels,
    build_observation_inventory,
)


DATA_ROOT = Path("Student folder")
OUTPUT_ROOT = Path("data/processed")


def process_partition(partition, limit=None):
    source = DATA_ROOT / partition

    if not source.is_dir():
        raise FileNotFoundError(f"Dataset not found: {source}")

    trip_dirs = sorted(
        p for p in source.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if limit is not None:
        trip_dirs = trip_dirs[:limit]

    if not trip_dirs:
        raise ValueError(f"No trips found in {source}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    candidate_path = OUTPUT_ROOT / f"{partition}_candidates.csv"
    observation_path = OUTPUT_ROOT / f"{partition}_observations.csv"

    summaries = []

    # Write each trip incrementally to avoid excessive memory usage.
    candidate_path.unlink(missing_ok=True)
    observation_path.unlink(missing_ok=True)

    wrote_candidates = False
    wrote_observations = False
    total_candidates = 0
    total_observations = 0

    for index, folder in enumerate(trip_dirs, start=1):
        print(
            f"[{index}/{len(trip_dirs)}] Processing {folder.name}",
            flush=True,
        )

        trip = read_trip(folder, include_reference=True)

        validation = validate_trip(trip)

        if any(validation.get(key, 0) > 0 for key in [
            "invalid_gps_coordinates",
            "invalid_timestamps",
            "non_increasing_timestamps",
            "invalid_road_nodes",
            "invalid_road_arcs",
            "invalid_reference_arcs",
        ]):
            raise ValueError(
                f"Structural validation failed for {folder.name}: "
                f"{validation}"
            )

        # The predictor only receives GPS and road-network data.
        inference_trip = {
            key: value
            for key, value in trip.items()
            if key != "route"
        }

        candidates = generate_candidates(
            inference_trip,
            k=8,
            search_radius_m=100.0,
        )

        features = add_features(inference_trip, candidates)

        # Reference information is used only for label preparation.
        alignment = align_reference_route(trip)

        labelled = add_labels(features, alignment)

        observations = build_observation_inventory(
            trip, alignment, labelled
        )

        observations = assess_observation_quality(
            observations,
            labelled,
        )

        labelled.to_csv(
            candidate_path,
            mode="a",
            header=not wrote_candidates,
            index=False,
        )

        observations.to_csv(
            observation_path,
            mode="a",
            header=not wrote_observations,
            index=False,
        )

        wrote_candidates = True
        wrote_observations = True

        total_candidates += len(labelled)
        total_observations += len(observations)

        summary = {
            "trip_id": trip["trip_id"],
            "observations": len(observations),
            "candidates": len(labelled),
            "valid_alignments": int(observations["label_valid"].sum()),
            "candidate_misses": int(observations["candidate_miss"].sum()),
            "flagged_alignments": int(
                observations["review_required"].sum()
            ),
        }

        summaries.append(summary)

        print(
            f"  candidates={summary['candidates']} "
            f"valid={summary['valid_alignments']} "
            f"misses={summary['candidate_misses']} "
            f"flagged={summary['flagged_alignments']}",
            flush=True,
        )

    print(f"\nSaved: {candidate_path}")
    print(f"Saved: {observation_path}")
    print(f"Candidate rows: {total_candidates}")
    print(f"Observations: {total_observations}")

    pd.DataFrame(summaries).to_csv(
        OUTPUT_ROOT / f"{partition}_summary.csv",
        index=False,
    )


def write_manifest():
    manifest = {
        "schema_version": "0.1",
        "task": "binary_candidate_classification",
        "feature_columns": FEATURE_COLUMNS,
        "feature_units": {
            "distance_to_arc_m": "metres",
            "arc_length_m": "metres",
            "heading_difference_deg": "degrees",
            "speed_mps": "metres_per_second",
            "time_gap_s": "seconds",
            "candidate_rank": "zero_based_distance_rank",
        },
        "label_column": "is_match",
        "label_valid_column": "label_valid",
        "group_column": "trip_id",
        "observation_column": "observation_id",
        "candidate_arc_column": "arc_id",
        "candidate_generation": {
            "k": 8,
            "search_radius_m": 100.0,
        },
        "alignment": {
            "method": "monotonic_reference_route_dynamic_programming",
            "skip_penalty": 0.5,
            "review_distance_m": 20.0,
            "review_extra_m": 10.0,
            "status": "provisional_not_independently_verified",
        },
        "missing_value_policy": (
            "NaN values are retained. Fit imputation "
            "inside training cross-validation folds."
        ),
        "training_policy": (
            "Join candidates to observations on trip_id and "
            "observation_id. Train only on observations where "
            "usable_for_training is True. Use is_match as the "
            "candidate label. Preserve all observations for "
            "coverage diagnostics. Fit preprocessing inside "
            "trip-grouped cross-validation folds."
        ),
        "quality_assessment": {
            "module": "pathsnap.features.quality",
            "close_road_threshold_m": 30.0,
            "training_eligibility_column": "usable_for_training",
            "network_quality_column": "network_quality",
            "exclusion_reason_column": "training_exclusion_reason",
            "status": "provisional_reference_alignment",
        },
        "inference_policy": (
            "Do not use reference_arc_id, review_required, "
            "label_valid or is_match as predictive features."
        ),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_ROOT / "feature_manifest.json"

    with path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--partition",
        choices=["train", "student_test"],
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N trips for a quick test",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    process_partition(args.partition, args.limit)
    write_manifest()
