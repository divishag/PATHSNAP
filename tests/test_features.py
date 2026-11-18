"""Feature engineering must not depend on reference routes."""
from pathlib import Path

import numpy as np
import pytest

from pathsnap.data.loader import read_trip
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.engineering import FEATURE_COLUMNS, add_features

ROOT = Path(__file__).resolve().parents[1]
TRIPS = sorted((ROOT / "Student folder" / "train").glob("0*"))


@pytest.mark.skipif(not TRIPS, reason="raw data not present")
def test_v2_features_complete_and_finite():
    trip = read_trip(TRIPS[0])  # include_reference=False: no labels available
    assert "route" not in trip
    features = add_features(trip, generate_candidates(trip))
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    values = features[FEATURE_COLUMNS].drop(columns="heading_difference_deg").to_numpy(float)
    assert np.isfinite(values).all()
    assert features["align_cos_w5"].between(-1, 1).all()
