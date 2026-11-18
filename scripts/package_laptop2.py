"""Create a curated Laptop 2 overlay; exclude raw data and Laptop 1 modules."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FILES = ["README.md", "requirements-laptop2.txt", "configs/models.yaml",
         "src/pathsnap/data/explore.py", "src/pathsnap/models/train.py",
         "src/pathsnap/models/evaluate.py", "src/pathsnap/models/compare.py",
         "src/pathsnap/models/neural.py", "src/pathsnap/optimization/bayesian.py",
         "src/pathsnap/visualization/folium_maps.py", "src/pathsnap/visualization/plots.py",
         "scripts/explore_dataset.py", "scripts/train_models.py", "scripts/compare_models.py",
         "scripts/tune_model.py", "scripts/generate_maps.py", "scripts/write_eda_report.py",
         "scripts/package_laptop2.py", "tests/test_laptop2.py", "tests/test_frontend_laptop2.py",
         "docs/laptop2_integration.md", "docs/laptop2_runbook.md", "docs/task_ownership.md",
         "reports/EDA_and_UI_progress.md", "reports/eda_source_manifest.json"]
TREES = ["frontend", "results/eda", "reports/screenshots"]


def package(output):
    paths = {ROOT / name for name in FILES}
    for directory in TREES:
        paths.update(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
    if missing: raise ValueError(f"Missing handoff files: {missing}")
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            data = path.read_bytes(); name = path.relative_to(ROOT).as_posix()
            archive.writestr(name, data)
            manifest.append({"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("HandoffManifest.json", json.dumps({"type": "Laptop 2 overlay", "trained_model_included": False,
                          "raw_data_included": False, "files": manifest}, indent=2))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None: raise ValueError("Handoff archive failed integrity check.")
    print(f"{output}: {len(manifest)} files, {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="handoff/PATHSNAP_Laptop2.zip")
    package(parser.parse_args().output)
