"""End-to-end PATHSNAP evaluation using an existing trained model."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.loader import read_trip
from pathsnap.decoding.pipeline import decode_trip

from pathsnap.features.engineering import FEATURE_COLUMNS

FEATURES = list(FEATURE_COLUMNS)


def find_test_folder():
    locations = [
        ROOT / "data/raw/student_test",
        ROOT / "data/student_test",
        ROOT / "Student folder/student_test",
        ROOT / "student_test",
    ]

    for location in locations:
        if location.is_dir():
            return location

    raise FileNotFoundError(
        "Student-test trip folders not found. "
        "Pass --trip-root with their location."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="hist_gradient_boosting",
    )
    parser.add_argument("--trip-root", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    model_path = (
        ROOT / "artifacts/laptop1" / f"{args.model}.joblib"
    )
    model = joblib.load(model_path)

    if 1 not in model.classes_:
        raise ValueError("Model lacks positive class")

    positive_index = list(model.classes_).index(1)

    candidates = pd.read_csv(
        ROOT / "data/processed/student_test_candidates.csv",
        dtype={"trip_id": str},
    )

    observations = pd.read_csv(
        ROOT / "data/processed/student_test_observations.csv",
        dtype={"trip_id": str},
    )

    # Match numeric CSV trip IDs to zero-padded folder names.
    candidates["trip_key"] = (
        candidates["trip_id"].str.zfill(8)
    )
    observations["trip_key"] = (
        observations["trip_id"].str.zfill(8)
    )

    trip_root = args.trip_root or find_test_folder()
    trip_dirs = sorted(
        p for p in trip_root.iterdir()
        if p.is_dir() and list(p.glob("*.track"))
    )

    output_dir = ROOT / "results/end_to_end"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []
    all_metrics = []

    print(f"Model: {args.model}", flush=True)
    print(f"Trip root: {trip_root}", flush=True)
    print(f"Trips found: {len(trip_dirs)}", flush=True)

    for trip_dir in trip_dirs[:args.limit]:
        start = perf_counter()
        trip = read_trip(trip_dir, include_reference=False)

        trip_candidates = candidates.loc[
            candidates.trip_key == trip_dir.name
        ].copy()

        trip_obs = observations.loc[
            observations.trip_key == trip_dir.name
        ].copy()

        if trip_candidates.empty:
            print(f"Skipping {trip_dir.name}: no candidates")
            continue

        # Decoder requires exactly the trip's own ID.
        trip_candidates["trip_id"] = str(trip["trip_id"])

        probabilities = model.predict_proba(
            trip_candidates[FEATURES]
        )[:, positive_index]

        trip_candidates["match_probability"] = probabilities

        print(
            f"Decoding {trip_dir.name}: "
            f"{len(trip_candidates)} candidates",
            flush=True,
        )

        decoded = decode_trip(
            trip,
            trip_candidates,
            probability_column="match_probability",
        )

        decoded["model"] = args.model
        all_predictions.append(decoded)

        reference = trip_obs.loc[
            trip_obs.label_valid.eq(1),
            ["observation_id", "reference_arc_id"],
        ].drop_duplicates("observation_id")

        evaluated = decoded.merge(
            reference,
            on="observation_id",
            how="inner",
            validate="one_to_one",
        )

        correct = (
            evaluated.predicted_arc_id
            == evaluated.reference_arc_id
        ).fillna(False)

        metrics = {
            "trip_id": trip_dir.name,
            "evaluated_observations": len(evaluated),
            "correct_observations": int(correct.sum()),
            "decoded_observations": int(
                decoded.predicted_arc_id.notna().sum()
            ),
            "total_observations": len(decoded),
            "elapsed_s": round(perf_counter() - start, 2),
        }

        metrics["arc_accuracy"] = (
            float(correct.mean()) if len(correct) else None
        )

        all_metrics.append(metrics)
        print("Result:", metrics, flush=True)

    if not all_predictions:
        raise RuntimeError("No trips decoded")

    predictions = pd.concat(
        all_predictions, ignore_index=True
    )
    predictions.to_csv(
        output_dir / f"{args.model}_predictions.csv",
        index=False,
    )

    total = sum(x["evaluated_observations"] for x in all_metrics)
    correct = sum(x["correct_observations"] for x in all_metrics)

    summary = {
        "model": args.model,
        "trips_processed": len(all_metrics),
        "evaluated_observations": total,
        "correct_observations": correct,
        "micro_arc_accuracy": correct / total if total else None,
        "trips": all_metrics,
    }

    (output_dir / f"{args.model}_metrics.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\nEND-TO-END COMPLETE", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
