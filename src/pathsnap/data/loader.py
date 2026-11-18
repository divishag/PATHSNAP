from pathlib import Path

import pandas as pd


def read_trip(trip_dir, include_reference=False):
    """
    Load one PathSnap trip.

    include_reference=False prevents ground-truth routes
    from entering the prediction pipeline.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name

    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip folder not found: {trip_dir}")

    def find_file(extension):
        files = list(trip_dir.glob(f"*{extension}"))

        if len(files) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {extension} file "
                f"in {trip_dir}, found {len(files)}"
            )

        return files[0]

    # GPS observations
    track = pd.read_csv(
        find_file(".track"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude", "timestamp"],
    )
    track.index.name = "observation_id"

    # Road-network nodes
    nodes = pd.read_csv(
        find_file(".nodes"),
        sep=r"\s+",
        header=None,
        names=["longitude", "latitude"],
    )
    nodes.index.name = "node_id"

    # Road-network arcs
    arcs = pd.read_csv(
        find_file(".arcs"),
        sep=r"\s+",
        header=None,
        names=["from_node_id", "to_node_id"],
        dtype="int64",
    )
    arcs.index.name = "arc_id"

    result = {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
    }

    # Reference route is only loaded when explicitly requested
    if include_reference:
        route = pd.read_csv(
            find_file(".route"),
            sep=r"\s+",
            header=None,
            names=["arc_id"],
            dtype="int64",
        )
        result["route"] = route

    return result
