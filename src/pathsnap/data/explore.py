"""Independent raw-data inspection, without importing Laptop 1's loader."""
import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd

from pathsnap.visualization.folium_maps import trip_map
from pathsnap.visualization.plots import eda_plots, trajectory_plot


def inspect_numeric(path, columns):
    """Preserve bad rows as NaN for explicit EDA counts rather than silently dropping."""
    rows, malformed, blanks = [], 0, 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            blanks += 1; continue
        values = line.split()
        if len(values) != columns:
            malformed += 1
            rows.append([float("nan")] * columns)
        else:
            converted = []
            for value in values:
                try: converted.append(float(value))
                except ValueError: converted.append(float("nan"))
            rows.append(converted)
    return np.asarray(rows, dtype=float).reshape(-1, columns), malformed, blanks


def haversine_m(points):
    points = np.radians(points)
    delta = np.diff(points, axis=0)
    term = np.sin(delta[:, 1] / 2) ** 2 + np.cos(points[:-1, 1]) * np.cos(points[1:, 1]) * np.sin(delta[:, 0] / 2) ** 2
    return 6371008.8 * 2 * np.arcsin(np.sqrt(np.clip(term, 0, 1)))


def inspect_trip(track):
    track = Path(track)
    record = {"trip_id": track.stem, "track_path": str(track)}
    arrays = {}
    for suffix, columns in [("track", 3), ("nodes", 2), ("arcs", 2), ("route", 1)]:
        path = track.with_suffix("." + suffix)
        record[f"{suffix}_present"] = path.exists()
        if not path.exists(): continue
        data, malformed, blanks = inspect_numeric(path, columns)
        arrays[suffix] = data
        record.update({f"{suffix}_bytes": path.stat().st_size, f"{suffix}_rows": len(data),
                       f"{suffix}_malformed_rows": malformed, f"{suffix}_blank_lines": blanks,
                       f"{suffix}_missing_values": int(np.isnan(data).sum())})
    gps = arrays["track"]
    valid = np.isfinite(gps).all(axis=1) & (np.abs(gps[:, 0]) <= 180) & (np.abs(gps[:, 1]) <= 90)
    record.update(gps_rows=len(gps), invalid_gps_rows=int((~valid).sum()))
    clean = gps[valid]
    if len(clean):
        record.update(lon_min=float(clean[:, 0].min()), lon_max=float(clean[:, 0].max()),
                      lat_min=float(clean[:, 1].min()), lat_max=float(clean[:, 1].max()),
                      duration_s=float(clean[:, 2].max() - clean[:, 2].min()))
    dt = np.diff(gps[:, 2])
    finite_dt = dt[np.isfinite(dt)]
    record.update(non_increasing_timestamps=int((finite_dt <= 0).sum()),
                  gap_steps_gt_1s=int((finite_dt > 1.000001).sum()),
                  exact_1s_steps=int(np.isclose(finite_dt, 1, atol=1e-6, rtol=0).sum()),
                  finite_time_steps=len(finite_dt),
                  max_gap_s=float(finite_dt.max()) if len(finite_dt) else None,
                  median_step_s=float(np.median(finite_dt)) if len(finite_dt) else None)
    pair_valid = valid[:-1] & valid[1:] & (dt > 0)
    distances = haversine_m(gps[:, :2])
    record["observed_distance_m"] = float(distances[pair_valid].sum())
    speeds = distances[pair_valid] / dt[pair_valid]
    record["max_observed_speed_mps"] = float(speeds.max()) if len(speeds) else None
    if "nodes" in arrays and "arcs" in arrays:
        nodes, raw_arcs = arrays["nodes"], arrays["arcs"]
        node_valid = np.isfinite(nodes).all(axis=1) & (np.abs(nodes[:, 0]) <= 180) & (np.abs(nodes[:, 1]) <= 90)
        record["invalid_nodes"] = int((~node_valid).sum())
        arc_valid = np.isfinite(raw_arcs).all(axis=1) & (raw_arcs == np.floor(raw_arcs)).all(axis=1) & (raw_arcs >= 0).all(axis=1) & (raw_arcs < len(nodes)).all(axis=1)
        arcs = raw_arcs[arc_valid].astype(int)
        record["invalid_arcs"] = int((~arc_valid).sum())
        graph = nx.DiGraph(); graph.add_nodes_from(range(len(nodes))); graph.add_edges_from(arcs.tolist())
        record.update(weak_components=nx.number_weakly_connected_components(graph),
                      strong_components=nx.number_strongly_connected_components(graph),
                      isolated_nodes=len(list(nx.isolates(graph))),
                      duplicate_arcs=len(arcs) - len(set(map(tuple, arcs.tolist()))))
        if "route" in arrays:
            route = arrays["route"].ravel()
            route_valid = np.isfinite(route) & (route == np.floor(route)) & (route >= 0) & (route < len(raw_arcs))
            record["invalid_reference_arc_ids"] = int((~route_valid).sum())
    return record, arrays


def explore_dataset(root, output, maps_per_partition=1):
    root, output = Path(root), Path(output)
    if not all((root / p).is_dir() for p in ["train", "student_test"]):
        raise ValueError(f"Expected train/ and student_test/ under {root}")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for partition in ["train", "student_test"]:
        tracks = sorted((root / partition).rglob("*.track"))
        if not tracks: raise ValueError(f"No .track files in {partition}")
        for index, track in enumerate(tracks):
            record, arrays = inspect_trip(track)
            record["partition"] = partition; records.append(record)
            if index < maps_per_partition and all(k in arrays for k in ["track", "nodes", "arcs"]):
                if record["invalid_gps_rows"] or record["invalid_nodes"] or record["invalid_arcs"]:
                    record["map_note"] = "Skipped because invalid geometry needs explicit cleaning policy."
                    continue
                route = arrays.get("route")
                if route is not None and not record.get("invalid_reference_arc_ids"):
                    route = route.ravel().astype(int)
                else: route = None
                prefix = f"{partition}_{track.stem}"
                trip_map(arrays["track"], arrays["nodes"], arrays["arcs"], output / "maps" / f"{prefix}.html", reference=route)
                trajectory_plot(arrays["track"], arrays["nodes"], arrays["arcs"].astype(int), output / "plots" / f"{prefix}.png")
    inventory = pd.DataFrame(records)
    inventory.to_csv(output / "inventory.csv", index=False)
    inventory.groupby("partition").describe().to_csv(output / "partition_statistics.csv")
    summary = {"source": str(root.resolve()), "partitions": {}}
    for name, rows in inventory.groupby("partition"):
        summary["partitions"][name] = {"trips": len(rows), "gps_observations": int(rows.gps_rows.sum()),
            "invalid_gps_observations": int(rows.invalid_gps_rows.sum()),
            "one_second_fraction": float(rows.exact_1s_steps.sum() / rows.finite_time_steps.sum()) if rows.finite_time_steps.sum() else None,
            "trips_with_time_gaps": int(rows.gap_steps_gt_1s.gt(0).sum()),
            "trips_with_non_increasing_time": int(rows.non_increasing_timestamps.gt(0).sum())}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    eda_plots(inventory, output / "plots")
    (output / "methodology.md").write_text(
        "# EDA methodology\n\nRaw source files were inspected without modification. Row indices remain original arc/node IDs. "
        "Malformed rows are counted as missing values. Blank lines are counted separately. Coordinates use longitude then latitude. "
        "Distances use the haversine formula with mean Earth radius 6,371,008.8 m; speed and distance totals use only adjacent valid points with positive time intervals. "
        "Observed distance is not road distance; apparent speed includes GPS noise. Degree axes in trajectory plots are for display only. "
        "Road connections are inspected as directed arcs; duplicate arcs collapse only in connectivity calculations. "
        "Reference routes are sequences of arcs; no pointwise labels or model performance are inferred here. "
        "Training and student-test outputs are identified separately. No evaluator data was inspected.\n", encoding="utf-8")
    return summary
