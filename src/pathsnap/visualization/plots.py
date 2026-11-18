"""Plots drawn exclusively from supplied observations or measured metrics."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def comparison_plots(frame, directory, histories=None):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True)
    metrics = [("cv_top1_accuracy", "Grouped CV observation Top-1 accuracy"),
               ("cv_macro_f1", "Grouped CV candidate Macro F1"),
               ("top1_accuracy", "Student-test observation Top-1 accuracy"),
               ("candidate_macro_f1", "Student-test candidate Macro F1"),
               ("candidate_log_loss", "Student-test candidate Log Loss"),
               ("inference_time_ms", "Student-test inference time (ms)")]
    for key, title in metrics:
        values = frame[key]
        if values.isna().all():
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        errors = frame[key + "_std"] if key in ["cv_macro_f1", "cv_top1_accuracy"] else None
        ax.bar(frame.model, values, yerr=errors, color="#1258a2", capsize=4)
        ax.set_title(title); ax.tick_params(axis="x", labelrotation=25)
        if key.endswith("f1") or key.endswith("top1_accuracy"): ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(target / f"{key}.png", dpi=160); plt.close(fig)
    for row in frame.itertuples():
        matrix = np.asarray(row.candidate_confusion_matrix)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix): ax.text(j, i, str(value), ha="center", va="center")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted candidate label", ylabel="Reference candidate label", title=row.model)
        fig.tight_layout(); fig.savefig(target / f"{row.model}_confusion.png", dpi=160); plt.close(fig)
    for name, history in (histories or {}).items():
        if not history: continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(history) + 1), history, color="#1258a2")
        ax.set(xlabel="Epoch", ylabel="Recorded training objective", title=f"{name}: actual training history (no validation-loss history)")
        fig.tight_layout(); fig.savefig(target / f"{name}_training_loss.png", dpi=160); plt.close(fig)


def eda_plots(inventory, directory):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True)
    for column in ["gps_rows", "duration_s", "gap_steps_gt_1s", "weak_components", "invalid_gps_rows"]:
        if column not in inventory: continue
        fig, ax = plt.subplots(figsize=(8, 4))
        bins = np.histogram_bin_edges(inventory[column].dropna(), bins=20)
        for partition, rows in inventory.groupby("partition"):
            ax.hist(rows[column].dropna(), bins=bins, alpha=.6, label=partition)
        ax.set(title=column.replace("_", " "), ylabel="Trips"); ax.legend()
        fig.tight_layout(); fig.savefig(target / f"eda_{column}.png", dpi=150); plt.close(fig)


def trajectory_plot(gps, nodes, arcs, output):
    fig, ax = plt.subplots(figsize=(8, 6))
    low, high = gps[:, :2].min(axis=0) - .005, gps[:, :2].max(axis=0) + .005
    geometry = nodes[arcs, :2]
    visible = ((geometry.max(axis=1) >= low) & (geometry.min(axis=1) <= high)).all(axis=1)
    ax.add_collection(LineCollection(geometry[visible], colors="#cbd5e1", linewidths=.5))
    ax.plot(gps[:, 0], gps[:, 1], color="#2563eb", linewidth=.8, label="GPS trajectory")
    ax.set(xlabel="Longitude", ylabel="Latitude", title="GPS and road-network geometry (angular coordinates)")
    ax.legend(); fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); plt.close(fig)
