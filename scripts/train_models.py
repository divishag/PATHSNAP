"""Laptop 2 command. Run from repository root."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
parser = argparse.ArgumentParser(description=__doc__)
from pathsnap.models.train import run_experiments
parser.add_argument("--config", default="configs/models.yaml")
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
results = run_experiments(config)
print(json.dumps([{"model": r["model"], "cv_top1_accuracy": r["cv_top1_accuracy"], "cv_macro_f1": r["cv_macro_f1"]} for r in results], indent=2))

