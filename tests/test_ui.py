"""Web UI tests: the page and its files are served, and the two additive API fields the UI relies on."""
import json
import re
from pathlib import Path

import pytest

from app.reasons import PLAIN, to_sentence
from app.schemas import ALLOWED
from features import FULL, explain, validate_orders
from tests.test_api import bundle, client, normal_order, raw, row_payload, scored, suspicious_order  # noqa: F401

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"
EXAMPLE_ROWS = {"normal": 0, "flagged": 135}           # rows of orders-sheet.csv used by "Try an example"


def ui_examples() -> dict:
    """The EXAMPLES object in app.js (JSON between the EXAMPLES:BEGIN / EXAMPLES:END markers)."""
    source = (STATIC / "app.js").read_text()
    block = re.search(r"/\* EXAMPLES:BEGIN \*/\s*const EXAMPLES = (\{.*?\});\s*/\* EXAMPLES:END \*/", source, re.S)
    assert block, "EXAMPLES markers missing from app.js"
    return json.loads(block.group(1))


# ---------------------------------------------------------------- page and static files
def test_index_is_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert '<form id="order-form"' in r.text and "/static/app.js" in r.text and "/static/style.css" in r.text
    assert r.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("path, media", [("/static/app.js", "javascript"), ("/static/batch.js", "javascript"),
                                         ("/static/style.css", "text/css"),
                                         ("/static/index.html", "text/html")])
def test_static_files_served(client, path, media):
    r = client.get(path)
    assert r.status_code == 200
    assert media in r.headers["content-type"]


def test_ui_uses_only_the_api_endpoints():
    """Single order: /health, /model-info, /analyze-order. Whole file: /analyze-batch and /reports/{id}/rows."""
    single = set(re.findall(r"callApi\(\"([^\"]+)\"", (STATIC / "app.js").read_text()))
    assert single == {"/health", "/model-info", "/analyze-order"}
    source = (STATIC / "batch.js").read_text()
    assert set(re.findall(r"callApi\(\"([^\"]+)\"", source)) == {"/analyze-batch"}
    assert re.findall(r"callApi\(`([^`$]+)", source) == ["/reports/"]


@pytest.mark.parametrize("name", ["app.js", "batch.js"])
def test_ui_never_writes_html_strings(name):
    """Server text is only ever written with textContent / createElement."""
    source = (STATIC / name).read_text()
    for sink in (".innerHTML", ".outerHTML", "insertAdjacentHTML", "document.write"):
        assert sink not in source


def test_ui_form_fields_match_schema():
    """Every request field appears in the FIELDS list of app.js, and nothing else."""
    from app.schemas import OrderRequest
    names = set(re.findall(r'\{ name: "([a-z_]+)"', (STATIC / "app.js").read_text()))
    assert names == set(OrderRequest.model_fields)


# ---------------------------------------------------------------- /model-info allowed_values
def test_model_info_allowed_values(client, bundle):
    info = client.get("/model-info").json()
    assert info["allowed_values"] == ALLOWED
    assert set(info["allowed_values"]["category"]) == set(bundle["pipeline"].named_steps["features"].cat_med_price_)


# ---------------------------------------------------------------- /analyze-order reasons
@pytest.mark.parametrize("which", ["normal_order", "suspicious_order"])
def test_analyze_returns_three_plain_reasons(client, request, which):
    body = client.post("/analyze-order", json=request.getfixturevalue(which)).json()
    reasons = body["reasons"]
    assert isinstance(reasons, list) and len(reasons) == 3
    for reason in reasons:
        assert isinstance(reason, str) and reason.endswith(".") and len(reason) > 10
        assert "log_" not in reason and "_pct" not in reason and "z=" not in reason


def test_reasons_follow_task1_explain(client, bundle, raw, suspicious_order):
    """The sentences are exactly features.explain's top 3, in order (no new ranking logic)."""
    body = client.post("/analyze-order", json=suspicious_order).json()
    df, _ = validate_orders(raw[raw.order_id == suspicious_order["order_id"]].head(1))
    X = bundle["pipeline"].named_steps["features"].transform(df)
    expected = [to_sentence(f, z) for f, z in explain(X, bundle["explain_stats"])[0]]
    assert body["reasons"] == expected


def test_every_feature_has_plain_wording(bundle):
    assert set(bundle["feature_order"]) <= set(PLAIN)
    assert set(FULL) <= set(PLAIN)


def test_sentence_wording():
    assert to_sentence("discount_pct", 6.5) == "The discount is much higher than usual."
    assert to_sentence("log_acct_age", -3.0) == "The customer account is noticeably newer than usual."
    assert to_sentence("log_fee_rate", 1.0) == "The platform fee is slightly higher than usual for an order of this size."


# ---------------------------------------------------------------- "Try an example" buttons
def ui_payload(example: dict) -> dict:
    """What app.js buildPayload() sends for an example: the datetime picker drops seconds and adds Z."""
    return {**example, "order_timestamp": example["order_timestamp"].replace(" ", "T")[:16] + ":00Z"}


@pytest.mark.parametrize("key, status", [("normal", "normal"), ("flagged", "suspicious")])
def test_ui_examples_are_real_rows(client, raw, key, status):
    example = ui_examples()[key]
    assert example == row_payload(raw.iloc[EXAMPLE_ROWS[key]])
    r = client.post("/analyze-order", json=ui_payload(example))
    assert r.status_code == 200, r.text
    assert r.json()["status"] == status
