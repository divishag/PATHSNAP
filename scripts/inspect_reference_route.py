from pathlib import Path
import numpy as np

from pathsnap.data.loader import read_trip


def inspect_route(trip):
    arcs = trip["arcs"]
    nodes = trip["nodes"]
    route = trip["route"]["arc_id"].to_numpy()

    # Retrieve the node pairs for each reference arc.
    route_arcs = arcs.iloc[route]

    starts = route_arcs["from_node_id"].to_numpy()
    ends = route_arcs["to_node_id"].to_numpy()

    # Check whether consecutive reference arcs share endpoints.
    direct = ends[:-1] == starts[1:]

    # Also check whether they share either endpoint.
    shared = (
        (starts[:-1] == starts[1:])
        | (starts[:-1] == ends[1:])
        | (ends[:-1] == starts[1:])
        | (ends[:-1] == ends[1:])
    )

    print("Trip:", trip["trip_id"])
    print("GPS observations:", len(trip["track"]))
    print("Reference arc entries:", len(route))
    print("Unique reference arcs:", len(np.unique(route)))
    print("Repeated arc entries:", len(route) - len(np.unique(route)))
    print("Consecutive arc pairs:", len(direct))
    print("Directed endpoint matches:", int(direct.sum()))
    print("Pairs sharing either endpoint:", int(shared.sum()))
    print("Pairs without shared endpoints:", int((~shared).sum()))

    print("\nFirst 10 reference arc IDs:")
    print(route[:10])

    print("\nFirst 5 reference arc connections:")
    for arc_id in route[:5]:
        row = arcs.iloc[int(arc_id)]
        print(
            f"Arc {arc_id}: "
            f"node {int(row['from_node_id'])} -> "
            f"node {int(row['to_node_id'])}"
        )

    print("\nGPS coordinate bounds:")
    track = trip["track"]
    print("Longitude:", track.longitude.min(), track.longitude.max())
    print("Latitude:", track.latitude.min(), track.latitude.max())


if __name__ == "__main__":
    folder = Path("Student folder/train/00000001")
    trip = read_trip(folder, include_reference=True)
    inspect_route(trip)
