"""Batch CSV checking: POST /analyze-batch, GET /reports/..., GET /template.csv. Run everything with: pytest"""
import csv
import io
import json
import logging
import typing
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import features
from app.batch import REPORT_COLUMNS
from app.config import Settings
from app.main import create_app
from app.schemas import (CONTEXT_REQUIRED_FIELDS, MODEL_REQUIRED_FIELDS, OPTIONAL_COLUMNS, REQUIRED_COLUMNS,
                         REQUIRED_FIELDS, OrderRequest)
from detect_anomalies import UID_COLUMNS, score_orders
from features import read_orders, validate_orders
from tests.test_api import DATA_PATH, MODEL_PATH, SCORED_PATH, assert_error, bundle, raw, row_payload, scored  # noqa: F401

CSV = "text/csv"
UUID_LIKE = "1b4e28ba-2fa1-41d2-883f-0016d3cca427"     # valid UUID4 format, never issued


# ---------------------------------------------------------------- fixtures / helpers
def make_client(tmp_path_factory, name: str, **overrides) -> TestClient:
    folder = tmp_path_factory.mktemp(name)
    options = {"model_path": str(MODEL_PATH), "app_env": "test", "reports_dir": str(folder), **overrides}
    return TestClient(create_app(Settings(**options)))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    with make_client(tmp_path_factory, "reports") as c:
        yield c


@pytest.fixture
def strict(tmp_path_factory):
    """Tiny limits and an empty REPORTS_DIR, to prove rejected uploads leave nothing behind."""
    with make_client(tmp_path_factory, "strict", max_upload_bytes=4096, max_upload_rows=5) as c:
        yield c


@pytest.fixture(scope="module")
def text_rows():
    """orders-sheet.csv exactly as text (blank cells stay blank)."""
    return pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False)


def reports_dir(c: TestClient) -> Path:
    return Path(c.app.state.settings.reports_dir)


def stored_files(c: TestClient) -> list[Path]:
    return sorted(reports_dir(c).rglob("*"))


def to_csv(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False)


def upload(c: TestClient, content, name: str = "orders.csv", ctype: str = CSV):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return c.post("/analyze-batch", files={"file": (name, content, ctype)})


def all_rows(c: TestClient, report_id: str, **params) -> list[dict]:
    out, page = [], 1
    while True:
        r = c.get(f"/reports/{report_id}/rows", params={"page": page, "page_size": 200, **params})
        assert r.status_code == 200, r.text
        body = r.json()
        out += body["rows"]
        if page >= body["total_pages"]:
            return out
        page += 1


def by_row(rows: list[dict]) -> dict[int, dict]:
    return {r["row_number"]: r for r in rows}


def download_csv(c: TestClient, report_id: str) -> pd.DataFrame:
    r = c.get(f"/reports/{report_id}/download", params={"format": "csv"})
    assert r.status_code == 200
    return pd.read_csv(io.BytesIO(r.content), dtype=str, keep_default_na=False, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def mixed(client, text_rows):
    """normal (row 0), suspicious (row 135), missing_data (blank quantity), invalid_data (discount 120)."""
    frame = text_rows.iloc[[0, 135, 1, 2]].copy()
    frame.iloc[2, frame.columns.get_loc("quantity")] = ""
    frame.iloc[3, frame.columns.get_loc("discount_pct")] = "120"
    r = upload(client, to_csv(frame), "mixed.csv")
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- required-field constants
def test_required_fields_are_what_the_schema_and_features_need():
    def nullable(name):
        return type(None) in typing.get_args(OrderRequest.model_fields[name].annotation)
    schema_required = {n for n, f in OrderRequest.model_fields.items() if f.is_required() and not nullable(n)}
    assert set(REQUIRED_FIELDS) == schema_required
    model_needs = set(features.REQUIRED_COLUMNS) - set(features.NULLABLE_INPUTS)
    assert set(MODEL_REQUIRED_FIELDS) == model_needs
    assert not set(CONTEXT_REQUIRED_FIELDS) & set(features.MODEL_INPUTS)       # context fields are not model inputs
    assert set(REQUIRED_COLUMNS) == set(REQUIRED_FIELDS) | set(features.NULLABLE_INPUTS)
    assert set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS) == set(OrderRequest.model_fields) == set(UID_COLUMNS)


def test_ui_page_lists_required_columns(client):
    page = client.get("/").text
    assert "{{BATCH_CONFIG}}" not in page
    for col in REQUIRED_COLUMNS:
        assert f"&quot;{col}&quot;" in page


# ---------------------------------------------------------------- valid and mixed files
def test_valid_file(client, text_rows):
    r = upload(client, to_csv(text_rows.iloc[:20]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["total_rows"] == 20
    assert body["summary"]["normal"] + body["summary"]["suspicious"] == 20
    assert body["summary"]["missing_data"] == body["summary"]["invalid_data"] == 0
    assert body["rows"]["total_rows"] == 20 and len(body["rows"]["rows"]) == 20
    assert body["unknown_columns"] == [] and body["ignored_columns"] == []
    assert (reports_dir(client) / body["report_id"] / "report.csv").is_file()
    assert r.headers["x-request-id"]


def test_mixed_file_has_all_four_statuses(mixed):
    s = mixed["summary"]
    assert (s["normal"], s["suspicious"], s["missing_data"], s["invalid_data"]) == (1, 1, 1, 1)
    assert s["scored_rows"] == 2 and s["suspicious_pct"] == 50.0
    rows = by_row(mixed["rows"]["rows"])
    assert rows[1]["status"] == "normal" and len(rows[1]["reasons"]) == 3
    assert rows[2]["status"] == "suspicious" and rows[2]["anomaly_score"] > mixed["threshold"]
    assert rows[3]["status"] == "missing_data" and rows[3]["anomaly_score"] is None
    assert rows[3]["missing_fields"] == ["quantity"]
    assert rows[4]["status"] == "invalid_data" and rows[4]["anomaly_score"] is None
    assert rows[4]["validation_errors"] == [{"field": "discount_pct", "reason": "must be at most 95"}]
    # default order: highest score first, unscored rows last
    assert [r["row_number"] for r in mixed["rows"]["rows"]] == [2, 1, 3, 4]


def test_blank_required_cells_are_missing_data(client, text_rows):
    frame = text_rows.iloc[:3].copy()
    frame.loc[frame.index[0], ["order_value", "payment_method"]] = ["", "  "]
    frame.loc[frame.index[1], "discount_pct"] = ""            # nullable: model imputes it, row is scored
    frame.loc[frame.index[2], "customer_age"] = ""            # optional
    rows = by_row(upload(client, to_csv(frame)).json()["rows"]["rows"])
    assert rows[1]["status"] == "missing_data"
    assert rows[1]["missing_fields"] == ["order_value", "payment_method"]
    assert rows[2]["status"] in ("normal", "suspicious") and rows[3]["status"] in ("normal", "suspicious")


@pytest.mark.parametrize("column, value, reason", [
    ("category", "Furniture", "is not one of the accepted values: Automotive, Beauty"),
    ("order_timestamp", "yesterday", "must be an ISO 8601 datetime"),
    ("order_timestamp", "2999-01-01 00:00:00", "must not be in the future"),
    ("unit_price", "NaN", "must be a real number (not NaN or infinity)"),
    ("order_value", "Infinity", "must be a real number (not NaN or infinity)"),
    ("tax_amount", "abc", "must be a number"),
    ("quantity", "1.5", "must be a whole number"),
    ("quantity", "0", "must be at least 1"),
    ("customer_age", "17", "must be at least 18"),
    ("order_id", "ORD 1", "may only contain letters, numbers"),
])
def test_invalid_values_are_invalid_data(client, text_rows, column, value, reason):
    frame = text_rows.iloc[:1].copy()
    frame[column] = value
    row = upload(client, to_csv(frame)).json()["rows"]["rows"][0]
    assert row["status"] == "invalid_data" and row["anomaly_score"] is None
    assert row["validation_errors"][0]["field"] == column
    assert row["validation_errors"][0]["reason"].startswith(reason)


def test_row_with_extra_values_is_invalid(client, text_rows):
    text = to_csv(text_rows.iloc[:2]).splitlines()
    text[2] += ",surprise"
    rows = by_row(upload(client, "\n".join(text)).json()["rows"]["rows"])
    assert rows[2]["status"] == "invalid_data" and rows[2]["validation_errors"][0]["field"] == "row"
    assert rows[1]["status"] in ("normal", "suspicious")


def test_header_normalisation_quotes_blank_lines_whitespace_bom(client, text_rows):
    frame = text_rows.iloc[:3].copy()
    frame["product_name"] = 'Chair, "deluxe"'                # quoted comma + quotes
    frame.iloc[0] = [cell + "  " if cell else cell for cell in frame.iloc[0]]    # trailing spaces
    lines = to_csv(frame).splitlines()
    lines[0] = ",".join(f"  {h.upper()} " for h in lines[0].split(","))     # messy header
    body = "﻿" + "\n".join([lines[0], "", lines[1], " , ,", *lines[2:], "", ""]) + "\n"
    r = upload(client, body)
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["total_rows"] == 3 and r.json()["summary"]["scored_rows"] == 3


def test_unknown_columns_are_listed_and_kept(client, text_rows):
    frame = text_rows.iloc[:2].copy()
    frame["warehouse"] = "north"
    body = upload(client, to_csv(frame)).json()
    assert body["unknown_columns"] == ["warehouse"]
    assert body["summary"]["scored_rows"] == 2
    assert list(download_csv(client, body["report_id"])["warehouse"]) == ["north", "north"]


def test_duplicate_order_ids_are_each_scored_and_flagged(client, text_rows):
    frame = text_rows.iloc[[0, 1, 0]].copy()                  # row 0 twice
    body = upload(client, to_csv(frame)).json()
    rows = by_row(body["rows"]["rows"])
    assert rows[1]["duplicate_order_id"] and rows[3]["duplicate_order_id"] and not rows[2]["duplicate_order_id"]
    assert rows[1]["anomaly_score"] == rows[3]["anomaly_score"]
    assert body["summary"]["duplicate_order_id_rows"] == 2 and body["summary"]["scored_rows"] == 3


def test_reupload_of_downloaded_report_gives_same_results(client, mixed):
    first = download_csv(client, mixed["report_id"])
    r = client.get(f"/reports/{mixed['report_id']}/download", params={"format": "csv"})
    again = upload(client, r.content, "mixed-report.csv").json()
    assert again["ignored_columns"] == REPORT_COLUMNS and again["unknown_columns"] == []
    assert again["summary"] == mixed["summary"]
    second = download_csv(client, again["report_id"])
    assert first.equals(second)


# ---------------------------------------------------------------- whole-file rejections (nothing stored)
def rejected(c: TestClient, resp, status: int, code: str) -> dict:
    err = assert_error(resp, status, code)
    assert stored_files(c) == [], "a rejected upload left files behind"
    return err


def test_reject_missing_required_column(strict, text_rows):
    err = rejected(strict, upload(strict, to_csv(text_rows.iloc[:2].drop(columns=["order_value", "category"]))),
                   422, "missing_columns")
    assert {d["field"] for d in err["details"]} == {"order_value", "category"}
    assert "order_value" in err["message"]


def test_reject_duplicate_headers(strict, text_rows):
    text = to_csv(text_rows.iloc[:2])
    text = text.replace("customer_age", "Quantity ", 1)        # normalises to a second 'quantity'
    err = rejected(strict, upload(strict, text), 422, "duplicate_columns")
    assert err["details"] == [{"field": "quantity", "reason": "column name appears more than once"}]


@pytest.mark.parametrize("content, code", [(b"", "empty_file"), (b" \r\n\r\n ", "empty_file"),
                                           (b"\xef\xbb\xbf", "empty_file")])
def test_reject_empty_file(strict, content, code):
    rejected(strict, upload(strict, content), 422, code)


def test_reject_header_only(strict, text_rows):
    rejected(strict, upload(strict, ",".join(text_rows.columns) + "\n\n"), 422, "header_only")


def test_reject_binary_renamed_csv(strict):
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(256))
    rejected(strict, upload(strict, png, "photo.csv"), 415, "not_text")


def test_reject_wrong_extension_and_part_type(strict, text_rows):
    body = to_csv(text_rows.iloc[:1])
    rejected(strict, upload(strict, body, "orders.xlsx"), 415, "unsupported_file_type")
    rejected(strict, upload(strict, body, "orders.csv", "image/png"), 415, "unsupported_file_type")


def test_reject_wrong_request_content_type(strict):
    r = strict.post("/analyze-batch", content=b"order_id\n1", headers={"content-type": "text/csv"})
    assert rejected(strict, r, 415, "unsupported_media_type")["message"] == "Content-Type must be multipart/form-data."
    r = strict.post("/analyze-batch", json={"file": "x"})
    rejected(strict, r, 415, "unsupported_media_type")


def test_reject_oversized_file(strict, text_rows):
    big = to_csv(pd.concat([text_rows.iloc[:1]] * 60))        # > 4096 bytes, under the multipart cap
    err = rejected(strict, upload(strict, big), 413, "payload_too_large")
    assert err["message"] == "The file is larger than the 4 KB limit."
    huge = b"x" * (200 * 1024)                                # over MAX_UPLOAD_BYTES + overhead: middleware 413
    err = rejected(strict, upload(strict, huge), 413, "payload_too_large")
    assert err["message"] == "The file is larger than the 4 KB limit."


def test_reject_too_many_rows(strict, text_rows):
    err = rejected(strict, upload(strict, to_csv(text_rows.iloc[:6])), 413, "too_many_rows")
    assert "more than 5 orders" in err["message"]


def test_reject_bad_encoding(strict, text_rows):
    latin1 = to_csv(text_rows.iloc[:1]).replace("HOM Product 1005", "Café crème").encode("latin-1")
    err = rejected(strict, upload(strict, latin1), 422, "bad_encoding")
    assert "UTF-8" in err["message"]


def test_reject_no_file_field(strict):
    r = strict.post("/analyze-batch", data={"other": "value"}, files={"not_file": ("a.csv", b"x", CSV)})
    err = rejected(strict, r, 422, "no_file")
    assert err["details"] == [{"field": "file", "reason": "no file was sent"}]


def test_reject_two_files(strict, text_rows):
    body = to_csv(text_rows.iloc[:1]).encode()
    r = strict.post("/analyze-batch", files=[("file", ("a.csv", body, CSV)), ("file", ("b.csv", body, CSV))])
    rejected(strict, r, 400, "bad_upload")


def test_reject_when_model_unavailable(tmp_path_factory, text_rows):
    with make_client(tmp_path_factory, "degraded", model_path="does-not-exist.joblib") as c:
        rejected(c, upload(c, to_csv(text_rows.iloc[:1])), 503, "model_unavailable")


def test_rejections_are_logged_without_row_contents(strict, caplog, text_rows):
    frame = text_rows.iloc[:6].copy()
    frame["product_name"] = "SECRET-PRODUCT-NAME"
    with caplog.at_level(logging.INFO):
        upload(strict, to_csv(frame), "secret.csv")
    assert "too_many_rows" in caplog.text and "secret.csv" in caplog.text
    assert "SECRET-PRODUCT-NAME" not in caplog.text


def test_success_log_has_counts_not_contents(client, caplog, text_rows):
    frame = text_rows.iloc[:2].copy()
    frame["product_name"] = "SECRET-PRODUCT-NAME"
    with caplog.at_level(logging.INFO):
        report_id = upload(client, to_csv(frame), "two.csv").json()["report_id"]
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("batch report_id="))
    assert report_id in line and "filename='two.csv'" in line and "rows=2" in line and "request_id=" in line
    assert "SECRET-PRODUCT-NAME" not in caplog.text and "ORD0020474" not in caplog.text


# ---------------------------------------------------------------- pagination, filtering, report ids
def test_pagination_filter_and_sort(client, text_rows):
    frame = text_rows.iloc[:45].copy()
    frame.loc[frame.index[:5], "quantity"] = ""
    report_id = upload(client, to_csv(frame)).json()["report_id"]
    url = f"/reports/{report_id}/rows"
    p1 = client.get(url, params={"page_size": 20}).json()
    p3 = client.get(url, params={"page_size": 20, "page": 3}).json()
    assert (p1["total_rows"], p1["total_pages"], len(p1["rows"]), len(p3["rows"])) == (45, 3, 20, 5)
    scores = [r["anomaly_score"] for r in all_rows(client, report_id) if r["anomaly_score"] is not None]
    assert scores == sorted(scores, reverse=True)
    missing = client.get(url, params={"status": "missing_data"}).json()
    assert missing["total_rows"] == 5 and {r["status"] for r in missing["rows"]} == {"missing_data"}
    assert client.get(url, params={"status": ""}).json()["total_rows"] == 45
    by_number = client.get(url, params={"sort": "row", "page_size": 3}).json()["rows"]
    assert [r["row_number"] for r in by_number] == [1, 2, 3]
    assert client.get(url, params={"page": 9}).json()["rows"] == []
    assert_error(client.get(url, params={"page_size": 201}), 422, "validation_error")
    assert_error(client.get(url, params={"page": 0}), 422, "validation_error")
    err = assert_error(client.get(url, params={"status": "fraud"}), 422, "validation_error")
    assert err["details"][0]["field"] == "status"


@pytest.mark.parametrize("path", [
    f"/reports/{UUID_LIKE}/rows", f"/reports/{UUID_LIKE}/download?format=csv",
    "/reports/not-a-uuid/rows", "/reports/12345/download",
    "/reports/..%2F..%2Fetc%2Fpasswd/rows", "/reports/../../etc/passwd/rows",
    "/reports/%2e%2e%2f%2e%2e%2fmodel.joblib/download", f"/reports/{UUID_LIKE.upper()}/rows",
])
def test_unknown_malformed_and_traversal_ids_are_404(client, path):
    assert_error(client.get(path), 404, "not_found")


def test_bad_download_format_is_422(client, mixed):
    assert_error(client.get(f"/reports/{mixed['report_id']}/download", params={"format": "xlsx"}),
                 422, "validation_error")


def test_expired_report_is_404_and_deleted(tmp_path_factory, text_rows):
    with make_client(tmp_path_factory, "ttl", report_ttl_minutes=1) as c:
        report_id = upload(c, to_csv(text_rows.iloc[:2])).json()["report_id"]
        meta_path = reports_dir(c) / report_id / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["created_at_epoch"] -= 120                       # two minutes old, TTL is one
        meta_path.write_text(json.dumps(meta))
        assert_error(c.get(f"/reports/{report_id}/rows"), 404, "not_found")
        assert not (reports_dir(c) / report_id).exists()


def test_purge_and_report_cap(tmp_path_factory, text_rows):
    with make_client(tmp_path_factory, "cap", max_reports=2) as c:
        ids = [upload(c, to_csv(text_rows.iloc[:1])).json()["report_id"] for _ in range(3)]
        assert sorted(p.name for p in reports_dir(c).iterdir()) == sorted(ids[1:])
        assert_error(c.get(f"/reports/{ids[0]}/rows"), 404)
        store = c.app.state.report_store
        meta_path = reports_dir(c) / ids[1] / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["created_at_epoch"] -= store.ttl_seconds + 1
        meta_path.write_text(json.dumps(meta))
        assert store.purge() == 1 and [p.name for p in reports_dir(c).iterdir()] == [ids[2]]


# ---------------------------------------------------------------- downloads
@pytest.fixture(scope="module")
def injected(client, text_rows):
    frame = text_rows.iloc[[0, 1]].copy()
    frame["product_name"] = ["=cmd()", "@SUM(A1)"]            # valid text for the schema -> rows are scored
    frame.loc[frame.index[1], "order_id"] = "+HYPERLINK(1)"   # invalid id -> invalid_data, still exported
    return upload(client, to_csv(frame), "=evil.csv").json()


def test_download_csv_is_excel_friendly_and_formula_safe(client, injected):
    r = client.get(f"/reports/{injected['report_id']}/download", params={"format": "csv"})
    assert r.status_code == 200 and r.content.startswith(b"\xef\xbb\xbf")
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"] and "evil-report.csv" in r.headers["content-disposition"]
    df = download_csv(client, injected["report_id"])
    assert list(df.columns) == UID_COLUMNS + REPORT_COLUMNS
    assert list(df["product_name"]) == ["'=cmd()", "'@SUM(A1)"]
    assert df["order_id"].iloc[1] == "'+HYPERLINK(1)"
    assert not any(cell.startswith(("=", "+", "@")) for cell in df.to_numpy().ravel())


def test_download_json(client, injected):
    r = client.get(f"/reports/{injected['report_id']}/download", params={"format": "json"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["summary"] == injected["summary"] and len(body["rows"]) == 2
    assert body["rows"][0]["values"]["product_name"] == "'=cmd()"
    assert {r["status"] for r in body["rows"]} == {"invalid_data", body["rows"][0]["status"]}
    again = client.get(f"/reports/{injected['report_id']}/download", params={"format": "json"})
    assert again.content == r.content                          # cached file is reused


def test_download_html_is_self_contained(client, injected, bundle):
    meta = json.loads((reports_dir(client) / injected["report_id"] / "meta.json").read_text())
    r = client.get(f"/reports/{injected['report_id']}/download", params={"format": "html"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "attachment" in r.headers["content-disposition"]
    page = r.text
    for needle in ("&#x27;=evil.csv", meta["sha256"], bundle["model_version"], f"{bundle['threshold']:.6f}",
                   "Score distribution", "<svg", "Rows that could not be checked", "Customer country",
                   "not proof of fraud", meta["uploaded_at"]):
        assert needle in page, needle
    for forbidden in ("<script", "http://", "https://", "<link", "@import", "src="):
        assert forbidden not in page, forbidden


# ---------------------------------------------------------------- template
def test_template_csv(client):
    r = client.get("/template.csv")
    assert r.status_code == 200 and r.content.startswith(b"\xef\xbb\xbf")
    assert 'filename="order-template.csv"' in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert rows[0] == UID_COLUMNS and len(rows) == 2
    body = upload(client, r.content, "order-template.csv").json()
    assert body["summary"]["normal"] == 1


# ---------------------------------------------------------------- consistency with /analyze-order and Task 1
def test_batch_matches_analyze_order_and_task1_on_real_rows(client, raw, scored, bundle, text_rows):
    """60 real rows (first 40 + 20 suspicious): same score (==) and status as /analyze-order and score_orders."""
    picks = list(range(40)) + list(scored[scored.status == "suspicious"].row_id.iloc[:20])
    body = upload(client, to_csv(text_rows.iloc[picks])).json()
    rows = by_row(all_rows(client, body["report_id"]))
    df, problems = validate_orders(raw.iloc[picks].reset_index(drop=True))
    assert len(problems) == 0
    task1_scores, task1_status = score_orders(bundle, df)
    for n, row_id in enumerate(picks, start=1):
        single = client.post("/analyze-order", json=row_payload(raw.iloc[row_id])).json()
        batch = rows[n]
        assert batch["anomaly_score"] == single["anomaly_score"] == task1_scores[n - 1]
        assert batch["status"] == single["status"] == task1_status[n - 1]
        assert batch["reasons"] == single["reasons"]
    assert sum(r["status"] == "suspicious" for r in rows.values()) >= 20


def test_full_file_reproduces_task1(client, raw, scored, bundle):
    """All 150,000 rows of orders-sheet.csv: Task 1 counts and scores, missing/invalid reported separately."""
    with open(DATA_PATH, "rb") as f:
        r = upload(client, f.read(), "orders-sheet.csv")
    assert r.status_code == 200, r.text
    body = r.json()
    s = body["summary"]
    assert s["total_rows"] == len(raw) == 150_000
    assert s["missing_data"] == 0 and s["invalid_data"] == 0
    assert s["suspicious"] == int((scored.status == "suspicious").sum())
    assert s["normal"] == int((scored.status == "normal").sum())
    report = download_csv(client, body["report_id"])
    assert list(report["row_number"].astype(int)) == list(range(1, 150_001))
    scores = report["anomaly_score"].map(float).to_numpy()     # Python float(): exact round-trip of repr()
    df, _ = validate_orders(raw)
    task1_scores, task1_status = score_orders(bundle, df)
    assert np.array_equal(scores, task1_scores)                        # exact
    assert (report["status"].to_numpy() == task1_status).all()
    assert np.allclose(scores, scored.anomaly_score.to_numpy(), atol=1e-6, rtol=0)   # file precision
    assert body["duration_ms"] < 30_000
