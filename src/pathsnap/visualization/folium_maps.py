"""Reusable geographical views; reference routes are for development maps only."""
from pathlib import Path
import folium
import numpy as np


def trip_map(gps, nodes, arcs, output, reference=None, predicted=None, title="Development trip"):
    gps, nodes, arcs = np.asarray(gps), np.asarray(nodes), np.asarray(arcs, dtype=int)
    if not len(gps):
        raise ValueError("Cannot map a trip with no valid GPS points.")
    view = folium.Map(location=gps[0, :2][::-1].tolist(), tiles="OpenStreetMap", control_scale=True)
    network = folium.FeatureGroup(name="Road network - grey", show=False).add_to(view)
    # Restrict the display to the trip's bounding box; keep all arcs in EDA counts.
    low, high = gps[:, :2].min(axis=0) - .005, gps[:, :2].max(axis=0) + .005
    geometry = nodes[arcs, :2]
    mask = ((geometry.max(axis=1) >= low) & (geometry.min(axis=1) <= high)).all(axis=1)
    visible = geometry[mask]
    if len(visible):
        folium.PolyLine(visible[:, :, ::-1].tolist(), color="#94a3b8", weight=2).add_to(network)
    observations = folium.FeatureGroup(name="GPS observations - blue").add_to(view)
    for point in gps:
        folium.CircleMarker(point[:2][::-1].tolist(), radius=2, color="#2563eb", fill=True).add_to(observations)
    for sequence, label, color in [(reference, "Reference route - green", "#16a34a"), (predicted, "Predicted route - red", "#e53935")]:
        if sequence is None:
            continue
        layer = folium.FeatureGroup(name=label).add_to(view)
        for arc_id in sequence:
            a, b = arcs[int(arc_id)]
            folium.PolyLine(nodes[[a, b], :2][:, ::-1].tolist(), color=color, weight=4).add_to(layer)
    folium.Marker(gps[0, :2][::-1].tolist(), tooltip="Start").add_to(view)
    folium.Marker(gps[-1, :2][::-1].tolist(), tooltip="End").add_to(view)
    view.fit_bounds(gps[:, :2][:, ::-1].tolist())
    folium.LayerControl(collapsed=False).add_to(view)
    # A fixed title is supplied by callers; no raw dataset text is injected as HTML.
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    view.save(str(output))
    return output
