"""API tests for Task 2 (FastAPI service). Run everything with: pytest"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_service import STATUS_LABELS
from app.schemas import CATEGORIES, OrderRequest
from detect_anomalies import score_orders
from features import read_orders, validate_orders

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model.joblib"
DATA_PATH = ROOT / "orders-sheet.csv"
SCORED_PATH = ROOT / "scored_orders.csv"
INT_FIELDS = {"quantity", "account_age_days", "customer_age"}
JSON = {"content-type": "application/json"}


# ---------------------------------------------------------------- fixtures / helpers
@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(Settings(model_path=str(MODEL_PATH), app_env="test"))) as c:
        yield c


@pytest.fixture(scope="module")
def raw():
    return read_orders(DATA_PATH)


@pytest.fixture(scope="module")
def scored():
    return pd.read_csv(SCORED_PATH)


def row_payload(row: pd.Series) -> dict:
    """One CSV row -> JSON request body (NaN -> null, ints as ints, every request field present)."""
    out = {}
    for name in OrderRequest.model_fields:
        value = row[name]
        if pd.isna(value):
            out[name] = None
        elif name in INT_FIELDS:
            out[name] = int(value)
        elif isinstance(value, (float, np.floating)):
            out[name] = float(value)
        else:
            out[name] = str(value)
    return out


@pytest.fixture(scope="module")
def normal_order(raw, scored):
    return row_payload(raw.iloc[int(scored[scored.status == "normal"].row_id.iloc[0])])


@pytest.fixture(scope="module")
def suspicious_order(raw, scored):
    return row_payload(raw.iloc[int(scored[scored.status == "suspicious"].row_id.iloc[0])])


def assert_error(resp, status: int, code: str | None = None) -> dict:
    """Every error: given status, the standard envelope, and request_id == X-Request-ID header."""
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    err = body["error"]
    assert set(err) == {"code", "message", "request_id", "details"}
    assert isinstance(err["code"], str) and isinstance(err["message"], str) and isinstance(err["details"], list)
    assert err["request_id"] and err["request_id"] == resp.headers["x-request-id"]
    for leak in ("Traceback", "model.joblib", str(ROOT), ".py"):
        assert leak not in resp.text
    if code:
        assert err["code"] == code
    return err


def error_fields(err: dict) -> set[str]:
    return {d["field"] for d in err["details"]}


# ---------------------------------------------------------------- /health and /model-info
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}
    assert r.headers["x-request-id"]


def test_model_info_matches_artifact(client, bundle):
    r = client.get("/model-info")
    assert r.status_code == 200
    info = r.json()
    assert info["model_version"] == bundle["model_version"]
    assert info["feature_names"] == list(bundle["feature_order"])
    assert info["feature_names"] == list(bundle["pipeline"].named_steps["features"].features)
    assert info["threshold"] == bundle["threshold"]
    assert info["n_training_rows"] == bundle["n_train"]
    assert info["trained_with"]["scikit-learn"] == bundle["versions"]["scikit-learn"]
    assert info["trained_with"]["pandas"] == bundle["versions"]["pandas"]
    assert info["status_labels"] == ["normal", "suspicious"]
    assert info["model_name"].startswith("IsolationForest")


def test_status_labels_and_categories_mirror_task1(bundle, raw):
    """The API constants agree with what Task 1 code and the artifact actually produce."""
    df, problems = validate_orders(raw.iloc[:2000])
    _, status = score_orders(bundle, df.drop(index=problems.index))
    assert set(status) <= set(STATUS_LABELS)
    assert set(CATEGORIES) == set(bundle["pipeline"].named_steps["features"].cat_med_price_)


# ---------------------------------------------------------------- /analyze-order: valid
def test_valid_normal_order(client, normal_order, bundle):
    r = client.post("/analyze-order", json=normal_order)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"order_id", "anomaly_score", "status", "threshold_used", "model_version", "reasons"}
    assert body["order_id"] == normal_order["order_id"]
    assert body["status"] == "normal" and body["anomaly_score"] <= bundle["threshold"]
    assert body["threshold_used"] == bundle["threshold"]
    assert body["model_version"] == bundle["model_version"]


def test_valid_suspicious_order(client, suspicious_order, bundle):
    r = client.post("/analyze-order", json=suspicious_order)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "suspicious"
    assert r.json()["anomaly_score"] > bundle["threshold"]


def test_unseen_product_uses_fallback(client, normal_order):
    r = client.post("/analyze-order", json={**normal_order, "product_id": "P99999"})
    assert r.status_code == 200, r.text
    assert math.isfinite(r.json()["anomaly_score"])


@pytest.mark.parametrize("changes", [
    {"quantity": 1, "discount_pct": 0.0, "shipping_cost": 0.0, "account_age_days": 0},
    {"discount_pct": 95.0},
    {"discount_pct": 95},                                  # int accepted for a float field
    {"discount_pct": None, "shipping_cost": None},         # imputed by OrderFeatures, as in training
    {"tax_amount": 0.0, "platform_fee": 0.0},
    {"customer_age": 18}, {"customer_age": 120}, {"customer_age": None},
    {"device_type": None, "ip_country": None},
    {"unit_price": 0.01, "order_value": 0.01},
    {"category": " toys ", "payment_method": "paypal", "sales_channel": "web "},   # dirty training labels
    {"order_timestamp": "2025-09-12T02:16:00Z"}, {"order_timestamp": "2025-09-12T02:16:00+02:00"},
    {"order_timestamp": "2025-09-12"},
    {"order_id": "  ORD1  "},                              # stripped
])
def test_boundary_values_accepted(client, normal_order, changes):
    r = client.post("/analyze-order", json={**normal_order, **changes})
    assert r.status_code == 200, r.text
    assert r.json()["status"] in STATUS_LABELS
    assert math.isfinite(r.json()["anomaly_score"])


def test_dirty_labels_score_like_clean_labels(client, normal_order):
    clean = client.post("/analyze-order", json=normal_order).json()
    dirty = client.post("/analyze-order", json={**normal_order, "category": normal_order["category"].lower() + " "}).json()
    assert dirty["anomaly_score"] == clean["anomaly_score"]


# ---------------------------------------------------------------- /analyze-order: rejected
@pytest.mark.parametrize("changes, field", [
    ({"quantity": -1}, "quantity"),
    ({"quantity": 0}, "quantity"),
    ({"quantity": 2.5}, "quantity"),
    ({"quantity": "2"}, "quantity"),                       # wrong type: no string coercion
    ({"quantity": True}, "quantity"),
    ({"unit_price": 0}, "unit_price"),
    ({"unit_price": -3.5}, "unit_price"),
    ({"unit_price": "22.74"}, "unit_price"),
    ({"discount_pct": 96}, "discount_pct"),
    ({"discount_pct": -1}, "discount_pct"),
    ({"shipping_cost": -0.01}, "shipping_cost"),
    ({"tax_amount": -1}, "tax_amount"),
    ({"platform_fee": -1}, "platform_fee"),
    ({"order_value": 0}, "order_value"),
    ({"account_age_days": -1}, "account_age_days"),
    ({"account_age_days": 10.5}, "account_age_days"),
    ({"customer_age": 5}, "customer_age"),
    ({"customer_age": 200}, "customer_age"),
    ({"order_timestamp": "yesterday"}, "order_timestamp"),
    ({"order_timestamp": "2025-13-45 99:00:00"}, "order_timestamp"),
    ({"order_timestamp": "2999-01-01T00:00:00Z"}, "order_timestamp"),
    ({"category": "Weapons"}, "category"),
    ({"payment_method": "Crypto"}, "payment_method"),
    ({"sales_channel": "Fax"}, "sales_channel"),
    ({"device_type": "Fridge"}, "device_type"),
    ({"customer_country": "Atlantis"}, "customer_country"),
    ({"coupon_used": "Maybe"}, "coupon_used"),
    ({"order_id": ""}, "order_id"),
    ({"order_id": "   "}, "order_id"),
    ({"order_id": "x" * 65}, "order_id"),
    ({"order_id": "ORD<script>"}, "order_id"),
    ({"product_id": 1005}, "product_id"),
    ({"category": None}, "category"),
    ({"product_name": "x" * 201}, "product_name"),
])
def test_invalid_fields_rejected(client, normal_order, changes, field):
    err = assert_error(client.post("/analyze-order", json={**normal_order, **changes}), 422, "validation_error")
    assert field in error_fields(err)
    assert all(d["reason"] for d in err["details"])


def test_missing_field(client, normal_order):
    order = {k: v for k, v in normal_order.items() if k != "unit_price"}
    err = assert_error(client.post("/analyze-order", json=order), 422)
    assert error_fields(err) == {"unit_price"}


def test_missing_nullable_field_is_still_required(client, normal_order):
    order = {k: v for k, v in normal_order.items() if k != "discount_pct"}
    err = assert_error(client.post("/analyze-order", json=order), 422)
    assert error_fields(err) == {"discount_pct"}


def test_extra_field(client, normal_order):
    err = assert_error(client.post("/analyze-order", json={**normal_order, "is_fraud": 1}), 422)
    assert error_fields(err) == {"is_fraud"}


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_rejected(client, normal_order, literal):
    body = json.dumps({**normal_order, "unit_price": 1.0}).replace('"unit_price": 1.0', f'"unit_price": {literal}')
    err = assert_error(client.post("/analyze-order", content=body, headers=JSON), 422)
    assert "unit_price" in error_fields(err)


def test_json_array_rejected(client, normal_order):
    assert_error(client.post("/analyze-order", json=[normal_order]), 422)


def test_empty_body(client):
    assert_error(client.post("/analyze-order", content=b"", headers=JSON), 400, "empty_body")


def test_malformed_json(client):
    assert_error(client.post("/analyze-order", content=b'{"order_id": "A",', headers=JSON), 400, "malformed_json")


@pytest.mark.parametrize("ctype", ["text/plain", "application/x-www-form-urlencoded", None])
def test_wrong_content_type(client, normal_order, ctype):
    headers = {"content-type": ctype} if ctype else {}
    r = client.post("/analyze-order", content=json.dumps(normal_order).encode(), headers=headers)
    assert_error(r, 415, "unsupported_media_type")


def test_content_type_with_charset_ok(client, normal_order):
    r = client.post("/analyze-order", content=json.dumps(normal_order),
                    headers={"content-type": "application/json; charset=utf-8"})
    assert r.status_code == 200


def test_oversized_body(client, normal_order):
    body = json.dumps({**normal_order, "product_name": "x" * 20000})
    assert_error(client.post("/analyze-order", content=body, headers=JSON), 413, "payload_too_large")


def test_oversized_chunked_body(client):
    def chunks():
        for _ in range(40):
            yield b" " * 1000
    assert_error(client.post("/analyze-order", content=chunks(), headers=JSON), 413, "payload_too_large")


@pytest.mark.parametrize("method, path", [("GET", "/analyze-order"), ("PUT", "/analyze-order"),
                                          ("POST", "/health"), ("DELETE", "/model-info")])
def test_wrong_method(client, method, path):
    r = client.request(method, path)
    assert_error(r, 405, "method_not_allowed")
    assert r.headers["allow"]


@pytest.mark.parametrize("path", ["/predict", "/docs", "/openapi.json", "/analyze-order/1", "/static/missing.js"])  # "/" is now the web UI
def test_unknown_route(client, path):
    assert_error(client.get(path), 404, "not_found")


# ---------------------------------------------------------------- request id
def test_request_id_echoed_when_valid(client):
    r = client.get("/nope", headers={"X-Request-ID": "abc-123.x_y"})
    assert r.headers["x-request-id"] == "abc-123.x_y"
    assert assert_error(r, 404)["request_id"] == "abc-123.x_y"


def test_request_id_replaced_when_invalid(client):
    r = client.get("/health", headers={"X-Request-ID": "bad id\twith spaces" + "x" * 80})
    assert r.headers["x-request-id"] != "bad id\twith spaces" + "x" * 80
    assert len(r.headers["x-request-id"]) == 32


def test_logs_no_payload(client, normal_order, caplog):
    caplog.set_level("INFO", logger="app")
    client.post("/analyze-order", json=normal_order)
    text = caplog.text
    assert f"order_id={normal_order['order_id']}" in text and "anomaly_score=" in text and "status=" in text
    assert "latency_ms=" in text and "request_id=" in text
    assert normal_order["customer_id"] not in text and normal_order["product_name"] not in text


# ---------------------------------------------------------------- failures: model / server
@pytest.mark.parametrize("make_path", [
    lambda tmp: tmp / "does_not_exist.joblib",
    lambda tmp: (tmp / "corrupt.joblib").write_bytes(b"not a pickle") and tmp / "corrupt.joblib",
    lambda tmp: joblib.dump({"pipeline": None}, tmp / "wrong.joblib") and tmp / "wrong.joblib",
])
def test_bad_model_path_degrades_to_503(tmp_path, normal_order, make_path):
    settings = Settings(model_path=str(make_path(tmp_path)), app_env="test")
    with TestClient(create_app(settings)) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": False}
        err = assert_error(c.get("/model-info"), 503, "model_unavailable")
        assert str(tmp_path) not in json.dumps(err)
        assert_error(c.post("/analyze-order", json=normal_order), 503, "model_unavailable")
        assert_error(c.post("/analyze-order", json={"bad": 1}), 422)      # validation still works
        assert c.get("/health").status_code == 200                          # still alive


def test_production_fails_fast_on_bad_model(tmp_path):
    settings = Settings(model_path=str(tmp_path / "missing.joblib"), app_env="production")
    with pytest.raises(RuntimeError, match="cannot load model"):
        with TestClient(create_app(settings)):
            pass


def test_missing_model_path_fails_fast():
    with pytest.raises(RuntimeError, match="MODEL_PATH is not set"):
        with TestClient(create_app(Settings(model_path=None))):
            pass


def test_unexpected_error_is_generic_500(normal_order, monkeypatch):
    with TestClient(create_app(Settings(model_path=str(MODEL_PATH), app_env="test"))) as c:
        def boom(order):
            raise KeyError(f"secret internal detail {MODEL_PATH}")
        monkeypatch.setattr(c.app.state.model_service, "score", boom)
        r = c.post("/analyze-order", json=normal_order)
        err = assert_error(r, 500, "internal_error")
        assert "secret" not in r.text and err["message"] == "Internal server error."
        assert c.get("/health").status_code == 200


# ---------------------------------------------------------------- consistency with detect_anomalies.py
def test_api_matches_detect_anomalies_on_real_rows(client, bundle, raw, scored):
    """30 real rows (first 20 + 10 suspicious): identical score and status to detect_anomalies.py."""
    row_ids = list(range(20)) + scored[scored.status == "suspicious"].row_id.head(10).astype(int).tolist()
    rows = raw.iloc[row_ids]
    df, problems = validate_orders(rows)
    assert problems.empty
    expected_scores, expected_status = score_orders(bundle, df)     # the exact CLI scoring function
    assert (expected_status == "suspicious").sum() >= 10 and (expected_status == "normal").sum() >= 10

    for i, row_id in enumerate(row_ids):
        r = client.post("/analyze-order", json=row_payload(raw.iloc[row_id]))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["anomaly_score"] == expected_scores[i], row_id                   # exact, not approx
        assert body["status"] == expected_status[i]
        written = scored.loc[scored.row_id == row_id].iloc[0]                         # detect_anomalies.py output file
        assert body["status"] == written.status
        assert body["anomaly_score"] == pytest.approx(written.anomaly_score, abs=1e-6)
