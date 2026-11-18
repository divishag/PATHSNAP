"""Directed road-network transition scoring for PathSnap."""

import numpy as np
from collections import OrderedDict
from pyproj import CRS, Transformer
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


class RoadTransitionModel:
    """Compute network-aware transitions between candidate road arcs."""

    def __init__(self, trip, beta_m=50.0, backtrack_tolerance_m=15.0, uturn_penalty=6.0):
        if not np.isfinite(beta_m) or beta_m <= 0:
            raise ValueError("beta_m must be positive and finite")

        self.beta_m = float(beta_m)
        # GPS noise can move consecutive fixes slightly backwards along the
        # same arc; treat that as noise instead of a loop around the network.
        self.backtrack_tolerance_m = float(backtrack_tolerance_m)
        # Log-score cost of switching to the reverse-direction twin arc.
        self.uturn_penalty = float(uturn_penalty)
        self.track = trip["track"]
        self.nodes = trip["nodes"]
        self.arcs = trip["arcs"]

        lon0 = float(self.track["longitude"].median())
        lat0 = float(self.track["latitude"].median())

        crs = CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
            "+datum=WGS84 +units=m +no_defs"
        )

        transformer = Transformer.from_crs(
            "EPSG:4326", crs, always_xy=True
        )

        nx, ny = transformer.transform(
            self.nodes["longitude"].to_numpy(dtype=float),
            self.nodes["latitude"].to_numpy(dtype=float),
        )
        self.node_xy = np.column_stack((nx, ny))

        gx, gy = transformer.transform(
            self.track["longitude"].to_numpy(dtype=float),
            self.track["latitude"].to_numpy(dtype=float),
        )
        self.gps_xy = np.column_stack((gx, gy))

        self.from_nodes = self.arcs["from_node_id"].to_numpy(dtype=int)
        self.to_nodes = self.arcs["to_node_id"].to_numpy(dtype=int)

        self.starts = self.node_xy[self.from_nodes]
        self.ends = self.node_xy[self.to_nodes]
        self.vectors = self.ends - self.starts
        self.lengths = np.linalg.norm(self.vectors, axis=1)

        # Sparse directed graph. Duplicate endpoint pairs are
        # consolidated using their minimum edge length.
        edges = {}
        for u, v, length in zip(
            self.from_nodes, self.to_nodes, self.lengths
        ):
            key = (int(u), int(v))
            if key not in edges or length < edges[key]:
                edges[key] = float(length)

        rows = []
        cols = []
        weights = []

        for (u, v), length in edges.items():
            rows.append(u)
            cols.append(v)
            weights.append(max(length, 1e-9))

        n = len(self.nodes)

        self.graph = csr_matrix(
            (weights, (rows, cols)),
            shape=(n, n),
        )

        self._distance_cache = OrderedDict()
        self.max_cached_sources = 32

    def candidate_fraction(self, observation_id, arc_id):
        """Fraction along a directed arc nearest to the GPS point."""
        point = self.gps_xy[int(observation_id)]
        arc_id = int(arc_id)

        start = self.starts[arc_id]
        vector = self.vectors[arc_id]

        squared = float(np.dot(vector, vector))

        if squared <= 0:
            return 0.0

        fraction = float(np.dot(point - start, vector) / squared)
        return float(np.clip(fraction, 0.0, 1.0))

    def shortest_node_distance(self, source_node, target_node):
        """Directed shortest-path distance, cached by source node."""
        source_node = int(source_node)
        target_node = int(target_node)

        if source_node in self._distance_cache:
            self._distance_cache.move_to_end(source_node)
        else:
            distances = dijkstra(
                self.graph,
                directed=True,
                indices=source_node,
            )

            self._distance_cache[source_node] = distances

            if len(self._distance_cache) > self.max_cached_sources:
                self._distance_cache.popitem(last=False)

        return float(self._distance_cache[source_node][target_node])

    def network_distance(
        self,
        previous_observation_id,
        previous_arc_id,
        current_observation_id,
        current_arc_id,
    ):
        """Network distance between projected positions on two arcs."""
        prev_arc = int(previous_arc_id)
        curr_arc = int(current_arc_id)

        prev_fraction = self.candidate_fraction(
            previous_observation_id, prev_arc
        )
        curr_fraction = self.candidate_fraction(
            current_observation_id, curr_arc
        )

        # Forward movement on the same directed segment.
        if prev_arc == curr_arc and curr_fraction >= prev_fraction:
            return float(
                (curr_fraction - prev_fraction)
                * self.lengths[prev_arc]
            )

        if prev_arc == curr_arc:
            backwards = (prev_fraction - curr_fraction) * self.lengths[prev_arc]
            if backwards <= self.backtrack_tolerance_m:
                return float(backwards)

        remaining_previous = (
            (1.0 - prev_fraction) * self.lengths[prev_arc]
        )

        entering_current = (
            curr_fraction * self.lengths[curr_arc]
        )

        middle_distance = self.shortest_node_distance(
            self.to_nodes[prev_arc],
            self.from_nodes[curr_arc],
        )

        return float(
            remaining_previous
            + middle_distance
            + entering_current
        )

    def transition_log_score(
        self,
        previous_observation_id,
        previous_arc_id,
        current_observation_id,
        current_arc_id,
    ):
        """Score a transition using GPS and network travel distances."""
        previous_observation_id = int(previous_observation_id)
        current_observation_id = int(current_observation_id)

        gps_distance = float(np.linalg.norm(
            self.gps_xy[current_observation_id]
            - self.gps_xy[previous_observation_id]
        ))

        network_distance = self.network_distance(
            previous_observation_id,
            previous_arc_id,
            current_observation_id,
            current_arc_id,
        )

        if not np.isfinite(network_distance):
            return float("-inf")

        score = -abs(network_distance - gps_distance) / self.beta_m

        prev_arc, curr_arc = int(previous_arc_id), int(current_arc_id)
        if (self.from_nodes[curr_arc] == self.to_nodes[prev_arc]
                and self.to_nodes[curr_arc] == self.from_nodes[prev_arc]):
            score -= self.uturn_penalty

        return float(score)
