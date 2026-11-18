"""Laptop 2 command. Run from repository root."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
parser = argparse.ArgumentParser(description=__doc__)
from pathsnap.data.explore import explore_dataset
parser.add_argument("--data-root", default="Student folder")
parser.add_argument("--output", default="results/eda")
parser.add_argument("--maps-per-partition", type=int, default=1)
args = parser.parse_args()
print(json.dumps(explore_dataset(args.data_root, args.output, args.maps_per_partition), indent=2))

