"""Laptop 2 evaluator interface. Run: streamlit run frontend/app.py."""
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse

import folium
import requests
import streamlit as st
import streamlit.components.v1 as components


class APIError(Exception):
    """An actionable backend or response error."""


def request_api(base, method, path, **kwargs):
    try:
        parsed = urlparse(base)
        valid_address = parsed.scheme in {"http", "https"} and bool(parsed.hostname)
        parsed.port  # Reject malformed ports before requesting.
    except ValueError:
        valid_address = False
    if not valid_address:
        raise APIError("Enter a complete API address, such as http://192.168.1.20:8000.")
    try:
        response = requests.request(method, base.rstrip("/") + path, timeout=(5, 600), **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise APIError("The API timed out. Check Laptop 1 and try again.") from exc
    except requests.ConnectionError as exc:
        raise APIError("Cannot reach Laptop 1. Check its address, Wi-Fi, and API server.") from exc
    except requests.HTTPError as exc:
        raise APIError(f"The API returned HTTP {response.status_code}. Check the server logs on Laptop 1.") from exc
    except (requests.RequestException, ValueError) as exc:
        raise APIError("The API did not return a valid JSON response.") from exc


def coordinates(value, name, minimum=1):
    if not isinstance(value, list) or len(value) < minimum:
        raise APIError(f"The response needs {name} with at least {minimum} coordinate(s).")
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise APIError(f"Invalid {name}: use [longitude, latitude] pairs.")
        lon, lat = point
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in point):
            raise APIError(f"Invalid numeric coordinate in {name}.")
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise APIError(f"Coordinate outside geographic bounds in {name}.")
    return [[lat, lon] for lon, lat in value]


def validate_result(result, trip_id):
    if not isinstance(result, dict) or result.get("trip_id") != trip_id:
        raise APIError("The API response trip_id does not match the selected trip.")
    coordinates(result.get("gps_coordinates"), "gps_coordinates")
    coordinates(result.get("predicted_route"), "predicted_route", 2)
    roads = result.get("road_segments", [])
    if not isinstance(roads, list):
        raise APIError("road_segments must be a list of coordinate sequences.")
    for road in roads:
        coordinates(road, "road_segments", 2)
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict):
        raise APIError("metrics must be a JSON object.")
    for key in ("top1_accuracy", "macro_f1", "route_continuity"):
        value = metrics.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
            raise APIError(f"metrics.{key} must be a number between 0 and 1, or null.")
    duration = metrics.get("inference_time_ms")
    if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0):
        raise APIError("metrics.inference_time_ms must be a nonnegative number, or null.")
    return result


def public_result(result):
    """Allow only documented geometry and aggregate metrics in downloadable JSON."""
    public = {k: result[k] for k in ("trip_id", "gps_coordinates", "predicted_route", "road_segments") if k in result}
    if "metrics" in result:
        public["metrics"] = {k: result["metrics"][k] for k in
                             ("top1_accuracy", "macro_f1", "route_continuity", "inference_time_ms")
                             if k in result["metrics"]}
    return public


def render_map(result):
    gps = coordinates(result["gps_coordinates"], "gps_coordinates")
    route = coordinates(result["predicted_route"], "predicted_route", 2)
    view = folium.Map(location=gps[0], tiles="OpenStreetMap", control_scale=True)
    roads = folium.FeatureGroup(name="Road network", show=False).add_to(view)
    for segment in result.get("road_segments", []):
        folium.PolyLine(coordinates(segment, "road_segments", 2), color="#94a3b8", weight=2).add_to(roads)
    predicted = folium.FeatureGroup(name="Predicted route · red").add_to(view)
    folium.PolyLine(route, color="#e53935", weight=5, tooltip="Predicted route").add_to(predicted)
    observed = folium.FeatureGroup(name="GPS observations · blue").add_to(view)
    for point in gps:
        folium.CircleMarker(point, radius=3, color="#2563eb", fill=True, fill_opacity=0.8).add_to(observed)
    folium.Marker(gps[0], tooltip="GPS start", icon=folium.Icon(color="green", icon="play")).add_to(view)
    folium.Marker(gps[-1], tooltip="GPS end", icon=folium.Icon(color="red", icon="stop")).add_to(view)
    view.fit_bounds(gps + route)
    folium.LayerControl(collapsed=False).add_to(view)
    return view.get_root().render()


def main():
    st.set_page_config(page_title="PathSnap | Evaluator", page_icon="📍", layout="wide")
    _, theme_control = st.columns([5, 1])
    light_theme = theme_control.toggle("Light theme", key="light_theme", help="Switch between dark and light appearance.")
    styles = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    if light_theme:
        styles += Path(__file__).with_name("light.css").read_text(encoding="utf-8")
    st.markdown("<style>" + styles + "</style>", unsafe_allow_html=True)
    st.markdown('''
    <nav class="ps-nav" aria-label="Main navigation">
      <div class="ps-brand"><span>⌁</span>PathSnap</div>
      <div class="ps-nav-links"><a href="#workspace">Workspace</a><a href="#how-it-works">How it works</a></div>
      <div class="ps-nav-tag">EVALUATOR INTERFACE ↗</div>
    </nav>
    <section class="ps-hero">
      <div class="ps-badge">From GPS noise to road clarity</div>
      <h1>Every journey,<br>a clearer path.</h1>
      <p>Turn scattered GPS observations into a connected route.<br>Explore the journey. See where every point belongs.</p>
      <div class="ps-actions"><a href="#workspace">Open workspace ↗</a><a href="#how-it-works">How it works →</a></div>
      <p class="ps-note">Map matching · Route reconstruction · Evaluation</p>
      <div class="ps-art" role="img" aria-label="Abstract blue field of points, decorative"></div>
    </section>
    <div id="workspace" class="ps-section"><h2>Route workspace</h2><span>SELECT A TRIP. FOLLOW THE ROUTE.</span></div>
    ''', unsafe_allow_html=True)

    with st.expander("Backend connection", expanded=not bool(st.session_state.get("trips"))):
        base = st.text_input("Laptop 1 API address", value=os.getenv("PATHSNAP_API_URL", "http://localhost:8000")).strip().rstrip("/")
        st.caption("Use Laptop 1’s local network IP when running on two laptops.")
        connect = st.button("Connect / refresh trips", use_container_width=True)

    if base != st.session_state.get("api_base"):
        st.session_state.api_base = base
        st.session_state.pop("trips", None)
        st.session_state.pop("result", None)
    if connect:
        st.session_state.pop("result", None)
        st.session_state.pop("trips", None)
        try:
            with st.spinner("Connecting to Laptop 1…"):
                data = request_api(base, "GET", "/trips")
                trips = data.get("trips") if isinstance(data, dict) else None
                if not isinstance(trips, list) or not trips or any(not isinstance(t, dict) or not isinstance(t.get("trip_id"), str) or not t["trip_id"].strip() for t in trips):
                    raise APIError("Expected a nonempty trips list with a string trip_id for each trip.")
                if len({t["trip_id"] for t in trips}) != len(trips):
                    raise APIError("The API returned duplicate trip IDs.")
                st.session_state.trips = trips
        except APIError as exc:
            st.error(str(exc))

    trips = st.session_state.get("trips", [])
    if not trips:
        st.markdown('<div class="ps-empty"><div class="ps-symbol">⌁</div><h3>Your next journey starts here.</h3><p>Connect to Laptop 1 to load the evaluation trips and reconstruct a route.</p></div>', unsafe_allow_html=True)
        render_explanation()
        return
    st.caption(f"Connected · {len(trips)} trips available")
    if len(trips) != 10:
        st.warning(f"The backend currently lists {len(trips)} trips. Final evaluation requires ten reserved trips.")
    labels = {t["trip_id"]: str(t.get("label") or t["trip_id"]) for t in trips}
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    selected = left.selectbox("Evaluation trip", list(labels), format_func=lambda key: labels[key])
    if st.session_state.get("selected_trip") != selected:
        st.session_state.selected_trip = selected
        st.session_state.pop("result", None)
    run = right.button("Reconstruct route", type="primary", use_container_width=True)
    if run:
        st.session_state.pop("result", None)
        try:
            with st.spinner("Reconstructing route on Laptop 1…"):
                result = request_api(base, "POST", "/predict", json={"trip_id": selected})
                st.session_state.result = validate_result(result, selected)
        except APIError as exc:
            st.error(str(exc))
    result = st.session_state.get("result")
    if result:
        st.subheader(labels[selected])
        metrics = result.get("metrics", {})
        for column, (key, title) in zip(st.columns(4), [("top1_accuracy", "Top-1 accuracy"), ("macro_f1", "Macro F1"), ("route_continuity", "Route continuity"), ("inference_time_ms", "Inference time")]):
            value = metrics.get(key)
            column.metric(title, "Not supplied" if value is None else (f"{value:,.0f} ms" if key.endswith("_ms") else f"{value:.1%}"))
        map_html = render_map(result)
        components.html(map_html, height=600, scrolling=False)
        st.caption("Blue points: GPS observations · Red line: predicted route · Use the map layer control to show road geometry.")
        st.download_button("Download interactive map", map_html, file_name="pathsnap-route.html", mime="text/html")
        # Export only the documented public response fields, never arbitrary backend fields.
        public = public_result(result)
        st.download_button("Download results", json.dumps(public, indent=2), file_name="pathsnap-result.json", mime="application/json")
    else:
        st.info("Choose a trip and select Reconstruct route to display its map and results.")
    render_explanation()


def render_explanation():
    st.markdown('<div id="how-it-works" class="ps-section"><h2>A route, reconstructed.</h2><span>HOW IT WORKS</span></div>', unsafe_allow_html=True)
    for column, title, description in zip(st.columns(3), ["01 / Select", "02 / Reconstruct", "03 / Explore"], ["Choose an evaluation trip from the connected server.", "The server matches GPS observations to roads and decodes a connected path.", "Inspect the predicted route, map layers, and returned evaluation metrics."]):
        with column:
            st.markdown(f"**{title}**")
            st.caption(description)
    st.markdown('<footer class="ps-footer">PATHSNAP &nbsp; / &nbsp; Distributed map matching &amp; route reconstruction</footer>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
