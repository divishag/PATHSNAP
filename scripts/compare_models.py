"""Laptop 2 command. Run from repository root."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
parser = argparse.ArgumentParser(description=__doc__)
from pathsnap.models.compare import compare_experiments
parser.add_argument("--experiments", default="results/experiments")
parser.add_argument("--metrics", default="results/metrics")
parser.add_argument("--plots", default="results/plots")
args = parser.parse_args()
print(compare_experiments(args.experiments, args.metrics, args.plots).to_string(index=False))

