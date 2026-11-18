# Laptop 2 evaluator UI

From the repository root, run:

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -r frontend/requirements.txt
cd frontend
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501. Enter Laptop 1's API address (for example http://192.168.1.20:8000), click **Connect / refresh trips**, select a trip, then click **Reconstruct route**.

The default API address can also be set through `PATHSNAP_API_URL`. The Backend connection setting overrides it for the current session. Both laptops must be on a network that permits communication. Laptop 1 must bind its API server to `0.0.0.0` and permit the API port through its firewall. HTTP requests are issued by the Streamlit server, so browser CORS configuration is unnecessary for this UI.

Use the Light theme toggle to change appearance during the session. The shared integration handoff, including proposed feature/model interfaces, is in `docs/laptop2_integration.md` at the repository root.

No trips, predictions, or metrics are fabricated. Until Laptop 1 provides the endpoints below, the UI shows its connection screen. The full timeout for a slow prediction is 600 seconds after connection (large evaluator road networks can take about a minute).

## Proposed API contract — agree with backend owner

`GET /trips` returns only the reserved evaluation-trip catalogue:

```json
{"trips": [{"trip_id": "actual-trip-id", "label": "Evaluation trip 1"}]}
```

Return ten unique IDs from the real evaluation data; the sample ID above is illustrative. `label` is optional.

`POST /predict` accepts:

```json
{"trip_id": "actual-trip-id"}
```

Response shape (coordinates and metrics below are illustrative, not experimental results):

```json
{
  "trip_id": "actual-trip-id",
  "gps_coordinates": [[77.59, 12.97], [77.591, 12.971]],
  "predicted_route": [[77.5901, 12.9701], [77.5911, 12.9711]],
  "road_segments": [[[77.59, 12.97], [77.591, 12.971]]],
  "metrics": {
    "top1_accuracy": null,
    "macro_f1": null,
    "route_continuity": null,
    "inference_time_ms": null
  }
}
```

All coordinate pairs use `[longitude, latitude]`. The backend must return ordered road geometry for the predicted route, including connecting segments where needed; arc IDs alone cannot be drawn. `gps_coordinates` needs at least one point; `predicted_route` needs at least two. `road_segments` and `metrics` are optional. Scores use fractions from 0 to 1; latency uses milliseconds. Missing/null metrics show “Not supplied”. The backend owner must document the definition of route continuity and which observations enter each score.

The model inference pipeline receives observations and road data only. A separate server-side evaluation step uses reference labels and returns aggregate metrics. Never return reference routes, observation labels, or evaluator ground truth to this interface. The backend should reject unknown trip IDs and log requests on Laptop 1 for the evaluator.

Folium uses internet-hosted map tiles and browser assets. Internet access is needed for the map; the API can use the local network. An offline map deployment would require locally hosted assets and tiles.

## Integration checklist

- Confirm endpoint paths and JSON field names with the backend owner.
- Confirm the trip catalogue contains all ten reserved evaluation trips.
- Run a prediction for every trip from Laptop 2; confirm corresponding Laptop 1 API logs.
- Check map coordinate order, layer toggles, score units, and downloads.
- Check connection failure, timeout, invalid trip, and malformed response messages.
- Check changing trip or API address clears the previous result.
- Capture actual UI and receiving-laptop API evidence for the report.

Frontend implemented; end-to-end backend integration remains pending. This document does not claim completed integration testing.
