"""Laptop 2 command. Run from repository root."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
parser = argparse.ArgumentParser(description=__doc__)
from pathsnap.data.explore import inspect_trip
from pathsnap.visualization.folium_maps import trip_map
parser.add_argument("--track", required=True, help="A train/student_test .track file")
parser.add_argument("--output", required=True)
args = parser.parse_args()
record, data = inspect_trip(Path(args.track))
if any(record.get(k, 0) for k in ["invalid_gps_rows", "invalid_nodes", "invalid_arcs", "invalid_reference_arc_ids"]):
    parser.error("Invalid geometry or reference IDs; inspect EDA and choose a cleaning policy first.")
trip_map(data["track"], data["nodes"], data["arcs"], args.output, reference=data.get("route", []).reshape(-1) if "route" in data else None)
print(args.output)

