"""Model comparison table, plots, classifier-vs-HMM analysis and Folium maps.

Reads only results already produced by the training / end-to-end scripts and
the student-testing partition. Run from the repository root:

    PYTHONPATH=src python scripts/make_comparison.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.data.loader import read_trip
from pathsnap.features.engineering import FEATURE_COLUMNS

EXP = ROOT / "results/experiments/laptop1"
E2E = ROOT / "results/end_to_end"
OUT = ROOT / "results/comparison"
MAPS = ROOT / "results/maps"
TRIPS = ROOT / "Student folder/student_test"
OUT.mkdir(parents=True, exist_ok=True)
MAPS.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("logistic_regression", "Logistic Regression", "Linear / probabilistic",
     "L2, C=1, class_weight=balanced, median impute + scaling"),
    ("weighted_knn", "Weighted KNN", "Instance-based",
     "k=15, distance weights, fit on 150k sampled rows"),
    ("random_forest", "Random Forest", "Bagging ensemble",
     "100 trees, 25% bootstrap, min_leaf=5, balanced_subsample"),
    ("hist_gradient_boosting", "HistGradientBoosting", "Boosting",
     "150 iters, lr=0.1, class_weight=balanced"),
    ("mlp", "MLP", "Neural network",
     "64-32 ReLU, Adam, 40 epochs, batch 2048, balanced weights"),
]
ORANGE, AQUA = "#eb6834", "#1baf7a"      # validated categorical slots 2 and 3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- table
rows = []
for key, name, family, config in MODELS:
    d = json.loads((EXP / f"{key}.json").read_text())
    folds = [f["top1_accuracy"] for f in d["folds"]]
    t = d["student_test"]
    rows.append({
        "model": name, "family": family, "key_configuration": config,
        "cv_top1_mean": np.mean(folds), "cv_top1_std": np.std(folds),
        "cv_macro_f1": d["cv_macro_f1"],
        "test_top1": t["top1_accuracy"],
        "test_macro_precision": t["candidate_macro_precision"],
        "test_macro_recall": t["candidate_macro_recall"],
        "test_macro_f1": t["candidate_macro_f1"],
        "test_log_loss": t["candidate_log_loss"],
        "test_candidate_miss_rate": t["candidate_miss_rate"],
        "test_inference_time_s": t["inference_time_ms"] / 1000,
        "train_top1_gap_vs_cv": None,
        "_cm": np.array(t["candidate_confusion_matrix"]),
    })
table = pd.DataFrame(rows).drop(columns="train_top1_gap_vs_cv")

# HMM end-to-end results, where they exist
for key, name, *_ in MODELS:
    path = E2E / f"{key}_metrics.json"
    if path.exists():
        table.loc[table.model == name, "test_hmm_arc_accuracy"] = json.loads(path.read_text())["micro_arc_accuracy"]

table.drop(columns="_cm").to_csv(OUT / "model_comparison.csv", index=False)
print(table.drop(columns=["_cm", "key_configuration"]).round(4).to_string(index=False))

# ---------------------------------------------------------------- bar chart
fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(table))
w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, table.cv_top1_mean, w, color=ORANGE, label="CV Top-1 (80 train trips, grouped 5-fold)",
            yerr=table.cv_top1_std, error_kw={"ecolor": MUTED, "lw": 1, "capsize": 3})
b2 = ax.bar(x + w / 2 + 0.01, table.test_top1, w, color=AQUA, label="Student-test Top-1 (10 trips)")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, 0.851, f"{b.get_height():.3f}", ha="center", va="bottom",
                fontsize=8, color=INK, rotation=90)
ax.set_ylim(0.85, 0.96)
ax.set_xticks(x, table.model, color=INK)
ax.set_ylabel("Observation Top-1 accuracy", color=MUTED)
ax.set_title("Five-model comparison (v2 features)", color=INK, loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=8)
style(ax)
fig.tight_layout()
fig.savefig(OUT / "model_comparison_top1.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
for ax, (_, r) in zip(axes, table.iterrows()):
    cm = r["_cm"]
    norm = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(norm, cmap="Oranges", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else INK)
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(r["model"], fontsize=10, color=INK)
fig.suptitle("Candidate-level confusion matrices, student-test (threshold 0.5; row-normalised %)",
             color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrices.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- classifier-only vs HMM per trip
model = joblib.load(ROOT / "artifacts/final/model.joblib")
cands = pd.read_csv(ROOT / "data/processed/student_test_candidates.csv", dtype={"trip_id": str})
obs = pd.read_csv(ROOT / "data/processed/student_test_observations.csv", dtype={"trip_id": str})
hmm = pd.read_csv(E2E / "hist_gradient_boosting_predictions.csv", dtype={"trip_id": str})
cands["score"] = model.predict_proba(cands[FEATURE_COLUMNS])[:, 1]
clf = cands.loc[cands.groupby(["trip_id", "observation_id"]).score.idxmax(), ["trip_id", "observation_id", "arc_id"]]
ref = obs.loc[obs.label_valid.astype(bool), ["trip_id", "observation_id", "reference_arc_id"]]


def continuity(arc_sequence, arcs):
    seq = [int(a) for a in arc_sequence if pd.notna(a)]
    changes = [(a, b) for a, b in zip(seq, seq[1:]) if a != b]
    if not changes:
        return 1.0, 0
    to_n, from_n = arcs.to_node_id.to_numpy(), arcs.from_node_id.to_numpy()
    ok = [to_n[a] == from_n[b] for a, b in changes]
    return float(np.mean(ok)), len(changes)


per_trip = []
for trip_id in sorted(ref.trip_id.unique()):
    arcs = read_trip(TRIPS / trip_id)["arcs"]
    r = ref[ref.trip_id == trip_id]
    c = r.merge(clf, on=["trip_id", "observation_id"], how="left")
    h = r.merge(hmm[hmm.trip_id == trip_id][["observation_id", "predicted_arc_id"]], on="observation_id", how="left")
    c_seq = clf[clf.trip_id == trip_id].sort_values("observation_id").arc_id
    h_seq = hmm[hmm.trip_id == trip_id].sort_values("observation_id").predicted_arc_id
    c_cont, c_n = continuity(c_seq, arcs)
    h_cont, h_n = continuity(h_seq, arcs)
    per_trip.append({
        "trip_id": trip_id, "evaluated_observations": len(r),
        "classifier_top1": float((c.arc_id == c.reference_arc_id).mean()),
        "hmm_top1": float((h.predicted_arc_id == h.reference_arc_id).mean()),
        "classifier_route_continuity": c_cont, "classifier_arc_changes": c_n,
        "hmm_route_continuity": h_cont, "hmm_arc_changes": h_n,
    })
pt = pd.DataFrame(per_trip)
total = pt.evaluated_observations.sum()
summary_row = {
    "trip_id": "ALL (micro)", "evaluated_observations": total,
    "classifier_top1": float((pt.classifier_top1 * pt.evaluated_observations).sum() / total),
    "hmm_top1": float((pt.hmm_top1 * pt.evaluated_observations).sum() / total),
    "classifier_route_continuity": pt.classifier_route_continuity.mean(),
    "classifier_arc_changes": pt.classifier_arc_changes.sum(),
    "hmm_route_continuity": pt.hmm_route_continuity.mean(),
    "hmm_arc_changes": pt.hmm_arc_changes.sum(),
}
pt = pd.concat([pt, pd.DataFrame([summary_row])], ignore_index=True)
pt.to_csv(OUT / "classifier_vs_hmm.csv", index=False)
print("\n", pt.round(4).to_string(index=False))

# ---------------------------------------------------------------- Folium maps
import folium


def arc_line(nodes, arcs, a):
    u, v = arcs.loc[int(a), ["from_node_id", "to_node_id"]]
    return [(nodes.latitude[u], nodes.longitude[u]), (nodes.latitude[v], nodes.longitude[v])]


for trip_id in ["00000077", "00000000"]:
    trip = read_trip(TRIPS / trip_id, include_reference=True)
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    m = folium.Map(location=[tr.latitude.median(), tr.longitude.median()], zoom_start=14, tiles="cartodbpositron")
    ref_layer = folium.FeatureGroup(name="Reference route (green)")
    for a in trip["route"].arc_id.unique():
        folium.PolyLine(arc_line(nodes, arcs, a), color="#008300", weight=7, opacity=0.5).add_to(ref_layer)
    pred_layer = folium.FeatureGroup(name="Predicted route, HGB + HMM (orange)")
    for a in hmm[hmm.trip_id == trip_id].predicted_arc_id.dropna().unique():
        folium.PolyLine(arc_line(nodes, arcs, a), color=ORANGE, weight=3, opacity=0.95).add_to(pred_layer)
    gps_layer = folium.FeatureGroup(name="Raw GPS (grey)")
    for lat, lon in zip(tr.latitude[::3], tr.longitude[::3]):
        folium.CircleMarker((lat, lon), radius=2, color="#52514e", fill=True, weight=0).add_to(gps_layer)
    for layer in (ref_layer, pred_layer, gps_layer):
        layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(MAPS / f"prediction_{trip_id}.html"))
print("\nSaved:", OUT, MAPS)
