"""Laptop 2 command. Run from repository root."""
import argparse
import json
import sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", default="configs/models.yaml")
parser.add_argument("--trials", type=int, default=30)
args = parser.parse_args()
try:
    from pathsnap.optimization.bayesian import tune
except ImportError as exc:
    parser.exit(2, f"Missing dependency: {exc}. Install requirements.txt in your virtual environment.\n")
config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
print(json.dumps(tune(config, args.trials), indent=2))

