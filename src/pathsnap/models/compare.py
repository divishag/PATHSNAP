"""Build comparison tables from persisted measured experiments."""
import json
from pathlib import Path

import pandas as pd
from pathsnap.visualization.plots import comparison_plots


def compare_experiments(experiment_dir, output_dir, plot_dir):
    records = []
    histories = {}
    for path in sorted(Path(experiment_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not {"model", "cv_macro_f1", "student_test"} <= data.keys():
            continue
        records.append({"model": data["model"], "family": data["family"],
                        "cv_top1_accuracy": data["cv_top1_accuracy"], "cv_top1_accuracy_std": data["cv_top1_accuracy_std"],
                        "cv_macro_f1": data["cv_macro_f1"], "cv_macro_f1_std": data["cv_macro_f1_std"],
                        "training_time_s": data["training_time_s"], **data["student_test"]})
        histories[data["model"]] = data.get("training_loss_history", [])
    if not records:
        raise ValueError("No measured experiment JSON files found. Run training first.")
    frame = pd.DataFrame(records).sort_values(["cv_top1_accuracy", "cv_top1_accuracy_std", "cv_macro_f1"], ascending=[False, True, False])
    target = Path(output_dir); target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "model_comparison.csv", index=False)
    comparison_plots(frame, plot_dir, histories)
    return frame
