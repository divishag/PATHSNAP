"""Apply observation quality assessment to existing feature exports."""

import argparse
from pathlib import Path

import pandas as pd

from pathsnap.features.quality import assess_observation_quality


def process_partition(partition: str) -> None:
    root = Path("data/processed")
    candidate_path = root / f"{partition}_candidates.csv"
    observation_path = root / f"{partition}_observations.csv"

    observations = pd.read_csv(
        observation_path,
        dtype={"trip_id": str},
    )

    candidates = pd.read_csv(
        candidate_path,
        dtype={"trip_id": str},
        usecols=[
            "trip_id",
            "observation_id",
            "distance_to_arc_m",
        ],
    )

    quality = assess_observation_quality(
        observations,
        candidates,
    )

    assert len(quality) == len(observations)
    assert not quality.duplicated(
        ["trip_id", "observation_id"]
    ).any()

    temporary_path = observation_path.with_suffix(".tmp.csv")
    quality.to_csv(temporary_path, index=False)
    temporary_path.replace(observation_path)

    print(f"\n{partition.upper()}")
    print("Observations:", len(quality))
    print("Usable for training:", int(quality["usable_for_training"].sum()))
    print("Network quality:")
    print(quality["network_quality"].value_counts().to_string())
    print("Exclusion reasons:")
    print(
        quality["training_exclusion_reason"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition",
        required=True,
        choices=["train", "student_test"],
    )
    args = parser.parse_args()
    process_partition(args.partition)
