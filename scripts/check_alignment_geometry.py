from pathlib import Path

import numpy as np
from pyproj import Transformer

from pathsnap.data.loader import read_trip


def inspect_alignment(trip):
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route_ids = trip["route"]["arc_id"].to_numpy()

    # Convert geographic coordinates to metres using UTM Zone 32N.
    transformer = Transformer.from_crs(
        "EPSG:4326", "EPSG:32632", always_xy=True
    )

    gps = np.column_stack(
        transformer.transform(
            track["longitude"].to_numpy(),
            track["latitude"].to_numpy()
        )
    )

    node_xy = np.column_stack(
        transformer.transform(
            nodes["longitude"].to_numpy(),
            nodes["latitude"].to_numpy()
        )
    )

    route_arcs = arcs.iloc[route_ids]
    starts = node_xy[
        route_arcs["from_node_id"].to_numpy(dtype=int)
    ]
    ends = node_xy[
        route_arcs["to_node_id"].to_numpy(dtype=int)
    ]

    # Calculate point-to-segment distance in manageable batches.
    nearest_indices = []
    nearest_distances = []

    for batch_start in range(0, len(gps), 100):
        points = gps[batch_start:batch_start + 100]

        vectors = ends - starts
        lengths_squared = np.sum(vectors ** 2, axis=1)

        offsets = points[:, None, :] - starts[None, :, :]

        projection = np.divide(
            np.sum(offsets * vectors[None, :, :], axis=2),
            lengths_squared[None, :],
            out=np.zeros((len(points), len(starts))),
            where=lengths_squared[None, :] > 0
        )

        projection = np.clip(projection, 0, 1)

        closest = (
            starts[None, :, :]
            + projection[:, :, None] * vectors[None, :, :]
        )

        distances = np.linalg.norm(
            points[:, None, :] - closest, axis=2
        )

        indices = np.argmin(distances, axis=1)

        nearest_indices.extend(indices.tolist())
        nearest_distances.extend(
            distances[np.arange(len(points)), indices].tolist()
        )

    indices = np.asarray(nearest_indices)
    distances = np.asarray(nearest_distances)

    print("Trip:", trip["trip_id"])
    print("GPS observations:", len(gps))
    print("Reference arcs:", len(route_ids))

    print("\nDistance to nearest reference segment (metres):")
    print("Median:", round(float(np.median(distances)), 2))
    print("90th percentile:", round(float(np.percentile(distances, 90)), 2))
    print("95th percentile:", round(float(np.percentile(distances, 95)), 2))
    print("Maximum:", round(float(np.max(distances)), 2))

    print("\nNearest reference segment sequence:")
    print("First 20 route positions:", indices[:20].tolist())
    print("Last 20 route positions:", indices[-20:].tolist())

    changes = np.diff(indices)

    print("\nNearest-segment ordering:")
    print("Backward movements:", int(np.sum(changes < 0)))
    print("Forward movements:", int(np.sum(changes > 0)))
    print("Unchanged:", int(np.sum(changes == 0)))

    print("\nNote: These are geometric nearest-segment")
    print("matches, NOT validated training labels.")


if __name__ == "__main__":
    trip = read_trip(
        Path("Student folder/train/00000001"),
        include_reference=True
    )
    inspect_alignment(trip)
