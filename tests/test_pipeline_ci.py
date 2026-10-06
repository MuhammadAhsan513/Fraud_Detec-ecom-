"""CI tests for the order anomaly project.

Unit tests        -> feature engineering (features.py inside the saved model bundle)
Integration tests -> detect_anomalies.py CLI end to end, and the FastAPI service

Needs two files in the repo root: model.joblib and test_orders.csv
Run locally:  pytest tests/test_pipeline_ci.py -v
"""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "model.joblib"
SAMPLE_CSV = ROOT / "test_orders.csv"

NORMAL_IDS = ["ORD0021459", "ORD0007322", "ORD0032620", "ORD0010405"]  # real-looking orders
BULK_ANOMALY_ID = "ORD0099002"  # qty 60 at price 1500 on a product that usually costs about 8


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def bundle():
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def pipeline(bundle):
    return bundle["pipeline"] if isinstance(bundle, dict) else bundle


@pytest.fixture(scope="module")
def orders():
    return pd.read_csv(SAMPLE_CSV)


# ---------------------------------------------------------------- UNIT TESTS
def test_unit_features_are_finite_and_complete(bundle, pipeline, orders):
    """Feature engineering returns one finite row per order, in the saved column order."""
    X = pipeline[0].transform(orders)
    assert len(X) == len(orders)
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    if isinstance(bundle, dict) and "feature_order" in bundle:
        assert list(X.columns) == list(bundle["feature_order"])


def test_unit_dirty_labels_do_not_change_features(pipeline, orders):
    """'garden' / 'paypal' / 'web ' must be cleaned, so they give the same features as clean labels."""
    dirty = orders[orders.order_id == "ORD0048204"].copy()
    clean = dirty.copy()
    clean["category"] = "Garden"
    clean["payment_method"] = "PayPal"
    clean["sales_channel"] = "Web"
    featurizer = pipeline[0]
    pd.testing.assert_frame_equal(
        featurizer.transform(dirty).reset_index(drop=True),
        featurizer.transform(clean).reset_index(drop=True),
    )


def test_unit_missing_discount_is_handled(pipeline, orders):
    """A null discount_pct must not crash, and must not create a fake residual violation."""
    row = orders[orders.discount_pct.isna()]
    assert len(row) == 1
    X = pipeline[0].transform(row)
    assert np.isfinite(X.to_numpy(dtype=float)).all()
    if "log_value_ratio" in X.columns:
        assert X["log_value_ratio"].iloc[0] == 0


# ------------------------------------------------------- INTEGRATION TESTS
def test_integration_cli_scores_every_row_and_flags_bulk_order(tmp_path):
    """detect_anomalies.py: CSV in -> scored CSV out, with the planted bulk order ranked as most anomalous."""
    out = tmp_path / "scored.csv"
    result = subprocess.run(
        [sys.executable, str(ROOT / "detect_anomalies.py"),
         "--model", str(MODEL_PATH), "--input", str(SAMPLE_CSV), "--output", str(out)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr

    scored = pd.read_csv(out)
    assert {"row_id", "order_id", "anomaly_score", "status"} <= set(scored.columns)
    assert len(scored) == len(pd.read_csv(SAMPLE_CSV))          # no rows dropped
    assert set(scored["status"]) <= {"normal", "suspicious"}

    bulk = scored[scored.order_id == BULK_ANOMALY_ID].iloc[0]
    normals = scored[scored.order_id.isin(NORMAL_IDS)]
    assert bulk["status"] == "suspicious"
    assert bulk["anomaly_score"] > normals["anomaly_score"].max()


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(MODEL_PATH))
    monkeypatch.setenv("APP_ENV", "production")            # fail fast if the model cannot load
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as test_client:                   # runs startup, loads the model
        yield test_client


def test_integration_api_health(client):
    """The API starts, loads the model bundle, and reports healthy."""
    response = client.get("/health")
    assert response.status_code == 200


def test_integration_api_batch_upload(client):
    """POST /analyze-batch accepts the sample CSV and returns a result."""
    content = SAMPLE_CSV.read_bytes()
    response = None
    for field in ("file", "csv", "upload", "data"):        # upload field name differs between implementations
        response = client.post("/analyze-batch", files={field: ("test_orders.csv", content, "text/csv")})
        if response.status_code == 200:
            break
    assert response.status_code == 200, response.text
    assert response.content
