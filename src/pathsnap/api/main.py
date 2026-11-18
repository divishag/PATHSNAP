"""PATHSNAP API compatible with the existing Laptop 2 frontend."""

import json
import logging
import os
import time
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pathsnap.data.loader import read_trip
from pathsnap.decoding.pipeline import decode_trip
from pathsnap.data.route_alignment import align_reference_route
from pathsnap.features.candidates import generate_candidates
from pathsnap.features.engineering import add_features

ROOT = Path(__file__).resolve().parents[3]
# Point this at the evaluator-testing folder on demo day:
#   PATHSNAP_TRIP_ROOT="/path/to/evaluator_test" uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000
TRIP_ROOT = Path(os.environ.get("PATHSNAP_TRIP_ROOT", ROOT / "Student folder" / "student_test"))
MODEL_PATH = ROOT / "artifacts" / "final" / "model.joblib"

from pathsnap.features.engineering import FEATURE_COLUMNS

FEATURES = list(FEATURE_COLUMNS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PATHSNAP", version="2.0")


class PredictionRequest(BaseModel):
    trip_id: str


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model missing: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if 1 not in model.classes_:
        raise RuntimeError("Model has no positive class")
    return model


def available_trips():
    if not TRIP_ROOT.is_dir():
        raise HTTPException(500, "Student-test trip folder not found")

    return sorted(
        p.name
        for p in TRIP_ROOT.iterdir()
        if p.is_dir() and any(p.glob("*.track"))
    )


@app.get("/")
def health():
    return {"name": "PATHSNAP", "status": "ready"}


@app.get("/trips")
def trips():
    return {
        "trips": [
            {"trip_id": trip_id, "label": f"Trip {trip_id}"}
            for trip_id in available_trips()
        ]
    }


@lru_cache(maxsize=10)
def run_prediction(trip_id):
    start_time = time.perf_counter()

    if trip_id not in available_trips():
        raise HTTPException(404, "Unknown trip ID")

    trip = read_trip(TRIP_ROOT / trip_id)

    # Candidates and features are built from GPS + road network only,
    # at request time. Reference labels never enter this path.
    candidates = add_features(
        trip, generate_candidates(trip, k=8, search_radius_m=100.0)
    )

    if candidates.empty:
        raise HTTPException(404, "No candidates for this trip")

    model = get_model()
    positive_index = list(model.classes_).index(1)

    candidates["trip_id"] = trip["trip_id"]
    candidates["match_probability"] = model.predict_proba(
        candidates[FEATURES]
    )[:, positive_index]

    decoded = decode_trip(
        trip,
        candidates,
        probability_column="match_probability",
    )

    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    gps_coordinates = [
        [float(row.longitude), float(row.latitude)]
        for row in track.itertuples()
    ]

    # Actual road-arc geometry, using the supplied graph.
    # Each arc is represented by its two endpoints.
    road_segments = []
    arc_geometry = {}

    for arc_id in decoded["predicted_arc_id"].dropna().unique():
        arc_id = int(arc_id)
        if arc_id not in arcs.index:
            continue

        arc = arcs.loc[arc_id]
        first = nodes.loc[int(arc["from_node_id"])]
        second = nodes.loc[int(arc["to_node_id"])]

        geometry = [
            [float(first.longitude), float(first.latitude)],
            [float(second.longitude), float(second.latitude)],
        ]
        arc_geometry[arc_id] = geometry
        road_segments.append(geometry)

    # Ordered arc-midpoint trace for the frontend's red line.
    # This is an approximate visualization, NOT a fully
    # connected network route or shortest-path reconstruction.
    predicted_route = []

    for arc_id in decoded["predicted_arc_id"]:
        if pd.isna(arc_id):
            continue

        geometry = arc_geometry.get(int(arc_id))
        if geometry is None:
            continue

        midpoint = [
            (geometry[0][0] + geometry[1][0]) / 2,
            (geometry[0][1] + geometry[1][1]) / 2,
        ]
        if not predicted_route or midpoint != predicted_route[-1]:
            predicted_route.append(midpoint)

    if len(predicted_route) < 2:
        raise HTTPException(
            422, "Not enough predicted road geometry to draw a route"
        )

    # ---- Server-side evaluation (after prediction is finished) ----
    # The reference route is loaded only here, separately from the
    # prediction path above, and never returned to the UI.
    arc_accuracy = None
    macro_f1 = None
    evaluated_count = 0
    try:
        reference_trip = read_trip(TRIP_ROOT / trip_id, include_reference=True)
        alignment = align_reference_route(reference_trip)
        reference = alignment.loc[
            ~alignment["review_required"],
            ["observation_id", "reference_arc_id"],
        ]
        evaluated = decoded.merge(reference, on="observation_id", how="inner")
        evaluated_count = len(evaluated)
        if evaluated_count:
            arc_accuracy = float(
                (evaluated["predicted_arc_id"] == evaluated["reference_arc_id"])
                .fillna(False).mean()
            )
        # Candidate-level macro F1 at threshold 0.5 (same definition as the report).
        scored = candidates.merge(reference, on="observation_id", how="inner")
        if len(scored):
            from sklearn.metrics import f1_score
            macro_f1 = float(f1_score(
                (scored["arc_id"] == scored["reference_arc_id"]).astype(int),
                (scored["match_probability"] >= 0.5).astype(int),
                average="macro", zero_division=0,
            ))
    except FileNotFoundError:
        log.warning("trip %s: no .route file, accuracy not computed", trip_id)

    # Route continuity: share of consecutive distinct predicted arcs that are
    # directly connected in the road graph (to_node of one = from_node of next).
    sequence = [int(a) for a in decoded["predicted_arc_id"].dropna()]
    changes = [(x, y) for x, y in zip(sequence, sequence[1:]) if x != y]
    if changes:
        to_n = arcs["to_node_id"].to_numpy()
        from_n = arcs["from_node_id"].to_numpy()
        route_continuity = float(np.mean([to_n[x] == from_n[y] for x, y in changes]))
    else:
        route_continuity = 1.0

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    return {
        "trip_id": trip_id,
        "gps_coordinates": gps_coordinates,
        "predicted_route": predicted_route,
        "road_segments": road_segments,
        "metrics": {
            "top1_accuracy": arc_accuracy,
            "macro_f1": macro_f1,
            "route_continuity": route_continuity,
            "evaluated_observations": evaluated_count,
            "total_observations": int(len(track)),
            "inference_time_ms": round(elapsed_ms, 2),
        },
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    log.info("POST /predict trip_id=%s (trip root: %s)", request.trip_id, TRIP_ROOT)
    result = run_prediction(request.trip_id)
    m = result["metrics"]
    log.info("trip %s done: accuracy=%s macro_f1=%s continuity=%.3f time=%.0f ms",
             request.trip_id, m["top1_accuracy"], m["macro_f1"], m["route_continuity"], m["inference_time_ms"])
    return result
