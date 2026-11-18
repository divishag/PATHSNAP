"""UI regression tests using simulated backend responses, never real scores."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evaluator_ui", ROOT / "frontend" / "app.py")
ui = importlib.util.module_from_spec(spec); spec.loader.exec_module(ui)


def response(data):
    value = Mock(); value.json.return_value = data
    return value


def payload(trip="0"):
    return {"trip_id": trip, "gps_coordinates": [[8.5, 49.8], [8.51, 49.81]],
            "predicted_route": [[8.5001, 49.8001], [8.5101, 49.8101]],
            "metrics": {"top1_accuracy": .8, "macro_f1": .7, "inference_time_ms": 12}}


def button(app, label):
    return next(b for b in app.button if b.label == label)


class FrontendChecks(unittest.TestCase):
    def start(self):
        app = AppTest.from_file(str(ROOT / "frontend" / "app.py")).run(timeout=20)
        self.assertFalse(app.exception)
        return app

    def connect(self, app):
        with patch("requests.request", return_value=response({"trips": [{"trip_id": str(i)} for i in range(10)]})):
            button(app, "Connect / refresh trips").click().run()
        self.assertFalse(app.exception)

    def predict(self, app):
        with patch("requests.request", return_value=response(payload())):
            button(app, "Reconstruct route").click().run()
        self.assertFalse(app.exception)

    def test_theme_preserves_results_and_trip_change_clears_them(self):
        app = self.start(); self.connect(app); self.predict(app)
        self.assertEqual(app.metric[0].value, "80.0%")
        app.toggle[0].set_value(True).run()
        self.assertFalse(app.exception); self.assertEqual(app.metric[0].value, "80.0%")
        app.toggle[0].set_value(False).run()
        self.assertEqual(app.metric[0].value, "80.0%")
        self.assertEqual(len(app.get("download_button")), 2)
        app.selectbox[0].select("1").run()
        self.assertFalse(app.metric)

    def test_failed_refresh_does_not_leave_stale_results(self):
        app = self.start(); self.connect(app); self.predict(app)
        with patch("requests.request", side_effect=requests.ConnectionError()):
            button(app, "Connect / refresh trips").click().run()
        self.assertFalse(app.exception); self.assertTrue(app.error)
        self.assertFalse(app.metric); self.assertFalse(app.selectbox)

    def test_address_change_clears_catalogue(self):
        app = self.start(); self.connect(app)
        app.text_input[0].set_value("http://192.168.1.90:8000").run()
        self.assertFalse(app.selectbox)

    def test_duplicate_trip_catalogue_is_rejected(self):
        app = self.start()
        with patch("requests.request", return_value=response({"trips": [{"trip_id": "a"}, {"trip_id": "a"}]})):
            button(app, "Connect / refresh trips").click().run()
        self.assertFalse(app.exception); self.assertIn("duplicate", app.error[0].value)

    def test_malformed_prediction_and_timeout_are_visible(self):
        app = self.start(); self.connect(app)
        for result in [response({"trip_id": "wrong"}), requests.Timeout()]:
            kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
            with patch("requests.request", **kwargs):
                button(app, "Reconstruct route").click().run()
            self.assertFalse(app.exception); self.assertTrue(app.error); self.assertFalse(app.metric)

    def test_invalid_urls_and_http_errors(self):
        for address in ["", "localhost:8000", "http://[", "http://host:bad"]:
            with self.subTest(address=address), self.assertRaises(ui.APIError): ui.request_api(address, "GET", "/trips")
        bad = response({}); bad.status_code = 500; bad.raise_for_status.side_effect = requests.HTTPError()
        with patch("requests.request", return_value=bad), self.assertRaisesRegex(ui.APIError, "500"):
            ui.request_api("http://localhost:8000", "GET", "/trips")
        bad = response({}); bad.json.side_effect = ValueError()
        with patch("requests.request", return_value=bad), self.assertRaisesRegex(ui.APIError, "valid JSON"):
            ui.request_api("http://localhost:8000", "GET", "/trips")

    def test_invalid_geometry_and_metrics_are_rejected(self):
        for change in [{"gps_coordinates": [[200, 49]]}, {"predicted_route": []},
                       {"metrics": {"macro_f1": 95}}, {"metrics": {"inference_time_ms": -1}},
                       {"road_segments": None}]:
            with self.subTest(change=change), self.assertRaises(ui.APIError):
                ui.validate_result({**payload(), **change}, "0")

    def test_download_contents_and_map_layers(self):
        result = payload()
        result["reference_route"] = ["must-not-export"]
        result["metrics"]["reference_labels"] = ["must-not-export"]
        exported = json.dumps(ui.public_result(result), allow_nan=False)
        self.assertNotIn("must-not-export", exported)
        self.assertEqual(json.loads(exported)["trip_id"], "0")
        html = ui.render_map(ui.validate_result(result, "0"))
        for label in ["Road network", "GPS observations", "Predicted route", "layer_control"]:
            self.assertIn(label, html)
        self.assertNotIn("must-not-export", html)


if __name__ == "__main__": unittest.main()
