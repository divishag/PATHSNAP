"""Sweep the HMM transition scale beta_m (Models guideline section 9).

Decoder parameters may be optimised on training and student-testing data
only. Uses the final classifier and the student-testing partition.

    PYTHONPATH=src python scripts/tune_decoder.py --betas 10 25 50 100 200
"""
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

TRIPS = ROOT / "Student folder/student_test"
OUT = ROOT / "results/decoder"


def continuity(sequence, arcs):
    seq = [int(a) for a in sequence if pd.notna(a)]
    changes = [(a, b) for a, b in zip(seq, seq[1:]) if a != b]
    if not changes:
        return 1.0
    to_n, from_n = arcs.to_node_id.to_numpy(), arcs.from_node_id.to_numpy()
    return float(np.mean([to_n[a] == from_n[b] for a, b in changes]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(ROOT / "artifacts/final/model.joblib"))
    parser.add_argument("--betas", type=float, nargs="+", default=[10, 25, 50, 100, 200])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    model = joblib.load(args.model)
    cands = pd.read_csv(ROOT / "data/processed/student_test_candidates.csv", dtype={"trip_id": str})
    obs = pd.read_csv(ROOT / "data/processed/student_test_observations.csv", dtype={"trip_id": str})
    cands["match_probability"] = model.predict_proba(cands[FEATURE_COLUMNS])[:, 1]
    ref = obs.loc[obs.label_valid.astype(bool), ["trip_id", "observation_id", "reference_arc_id"]]
    trips = {t: read_trip(TRIPS / t) for t in sorted(ref.trip_id.unique())}

    rows = []
    for beta in args.betas:
        t0 = perf_counter()
        correct = total = 0
        conts = []
        for trip_id, trip in trips.items():
            c = cands[cands.trip_id == trip_id].copy()
            decoded = decode_trip(trip, c, probability_column="match_probability", beta_m=beta)
            r = ref[ref.trip_id == trip_id].merge(decoded[["observation_id", "predicted_arc_id"]], on="observation_id")
            correct += int((r.predicted_arc_id == r.reference_arc_id).sum())
            total += len(r)
            conts.append(continuity(decoded.sort_values("observation_id").predicted_arc_id, trip["arcs"]))
        row = {"beta_m": beta, "arc_accuracy": correct / total, "mean_route_continuity": float(np.mean(conts)),
               "evaluated_observations": total, "elapsed_s": perf_counter() - t0}
        rows.append(row)
        print(row, flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "beta_sweep.csv", index=False)
    best = frame.loc[frame.arc_accuracy.idxmax()].to_dict()
    (OUT / "beta_sweep_best.json").write_text(json.dumps(best, indent=2))
    print("best:", best)


if __name__ == "__main__":
    main()
