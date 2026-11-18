from pathlib import Path


def test_required_directories_exist():
    root = Path(__file__).resolve().parents[1]

    required = [
        "src/pathsnap/data",
        "src/pathsnap/features",
        "src/pathsnap/models",
        "src/pathsnap/optimization",
        "src/pathsnap/decoding",
        "src/pathsnap/api",
        "src/pathsnap/visualization",
        "frontend",
        "scripts",
        "configs",
        "results",
    ]

    for directory in required:
        assert (root / directory).is_dir()
