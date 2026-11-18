"""Tests for bounded shortest-path caching."""

import numpy as np

from test_transitions import make_trip
from pathsnap.decoding.transitions import RoadTransitionModel


def test_cache_reuses_existing_source():
    model = RoadTransitionModel(make_trip())

    first = model.shortest_node_distance(0, 2)
    second = model.shortest_node_distance(0, 2)

    assert np.isclose(first, second)
    assert len(model._distance_cache) == 1


def test_cache_has_bounded_size():
    model = RoadTransitionModel(make_trip())
    model.max_cached_sources = 2

    for source in [0, 1, 2, 3]:
        model.shortest_node_distance(source, 3)

    assert len(model._distance_cache) <= 2


def test_cache_eviction_preserves_correctness():
    model = RoadTransitionModel(make_trip())
    model.max_cached_sources = 1

    expected = model.shortest_node_distance(0, 2)

    model.shortest_node_distance(1, 3)

    actual = model.shortest_node_distance(0, 2)

    assert np.isclose(actual, expected)
