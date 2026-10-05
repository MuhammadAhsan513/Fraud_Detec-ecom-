"""Load the Task 1 model bundle once, validate it, and score single orders.

Scoring deliberately reuses Task 1 code end to end:
    features.validate_orders  ->  detect_anomalies.score_orders  ->  features.anomaly_scores
    ->  Pipeline[OrderFeatures.transform, IsolationForest]
so an order scored here gets exactly the same anomaly_score and status as in detect_anomalies.py.
"""
import math
from dataclasses import dataclass, field
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.utils.validation import check_is_fitted

from detect_anomalies import score_orders
from features import ID_DTYPES, OrderFeatures, explain, validate_orders

from .logging_config import get_logger
from .reasons import to_sentence
from .schemas import ALLOWED

log = get_logger("model")

# detect_anomalies.score_orders emits exactly these labels (score > threshold -> "suspicious").
# The bundle does not store them, so they mirror Task 1; tests/test_api.py checks they agree.
STATUS_LABELS = ["normal", "suspicious"]
REQUIRED_KEYS = ["pipeline", "threshold", "feature_order", "model_version", "versions"]


class ModelUnavailableError(RuntimeError):
    """The model is missing, corrupt or failed to load. Mapped to HTTP 503."""


class ArtifactError(RuntimeError):
    """The loaded file is not a valid Task 1 model bundle."""


class OrderRejectedError(ValueError):
    """Task 1's validate_orders rejected the order. Mapped to HTTP 422."""

    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def validate_artifact(bundle: Any) -> None:
    """Raise ArtifactError unless `bundle` looks like a fitted Task 1 model bundle."""
    if not isinstance(bundle, dict):
        raise ArtifactError(f"expected a dict bundle, got {type(bundle).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in bundle]
    if missing:
        raise ArtifactError(f"bundle is missing keys {missing}")
    threshold = bundle["threshold"]
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
        raise ArtifactError("threshold is not a finite number")
    pipe = bundle["pipeline"]
    steps = getattr(pipe, "named_steps", None)
    if steps is None or "features" not in steps or "iforest" not in steps:
        raise ArtifactError("pipeline must have 'features' and 'iforest' steps")
    if not isinstance(steps["features"], OrderFeatures):
        raise ArtifactError("pipeline 'features' step is not features.OrderFeatures")
    feature_order = list(bundle["feature_order"])
    if list(steps["features"].features) != feature_order:
        raise ArtifactError("feature_order does not match the pipeline's feature transformer")
    try:
        check_is_fitted(steps["iforest"])
        check_is_fitted(steps["features"], "prod_med_price_")
    except Exception as e:  # NotFittedError and friends
        raise ArtifactError(f"pipeline is not fitted: {e}") from None
    if getattr(steps["iforest"], "n_features_in_", None) != len(feature_order):
        raise ArtifactError("model input width does not match feature_order")


@dataclass
class ModelService:
    """Holds the loaded bundle. `loaded` is False if loading failed (the API then answers 503)."""
    model_path: str | None = None
    bundle: dict | None = None
    error: str | None = None                 # internal reason, logged, never sent to clients
    warnings: list[str] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        return self.bundle is not None

    @classmethod
    def load(cls, model_path: str) -> "ModelService":
        """Load + validate the bundle once. Never raises: failures are recorded in `error`."""
        service = cls(model_path=model_path)
        try:
            bundle = joblib.load(model_path)
            validate_artifact(bundle)
        except Exception as e:                # missing file, corrupt pickle, wrong object...
            service.error = f"{type(e).__name__}: {e}"
            return service
        trained = bundle["versions"]
        for name, runtime in [("scikit-learn", sklearn.__version__), ("pandas", pd.__version__)]:
            if trained.get(name) != runtime:
                service.warnings.append(f"{name} {runtime} at runtime, model trained with {trained.get(name)}")
        service.bundle = bundle
        return service

    def _require(self) -> dict:
        if self.bundle is None:
            raise ModelUnavailableError("model not loaded")
        return self.bundle

    def info(self) -> dict:
        """Model metadata, read from the loaded bundle."""
        b = self._require()
        versions = b["versions"]
        return {
            "model_name": f"{type(b['pipeline'].named_steps['iforest']).__name__} ({b.get('feature_set', 'custom')})",
            "model_version": b["model_version"],
            "feature_names": list(b["feature_order"]),
            "threshold": float(b["threshold"]),
            "threshold_method": b.get("threshold_method"),
            "score_definition": b.get("score_definition"),
            "status_labels": STATUS_LABELS,
            "trained_with": {"scikit-learn": versions.get("scikit-learn"), "pandas": versions.get("pandas"),
                             "numpy": versions.get("numpy"), "python": versions.get("python")},
            "n_training_rows": b.get("n_train"),
            "trained_at": b.get("trained_at"),
            "allowed_values": {col: list(values) for col, values in ALLOWED.items()},
        }

    def known_categories(self) -> set[str]:
        """Categories the feature transformer learned a median price for."""
        return set(self._require()["pipeline"].named_steps["features"].cat_med_price_)

    def _validated_frame(self, orders: list[dict]) -> tuple[pd.DataFrame, pd.Series]:
        """Schema-validated orders -> (frame, problems) via Task 1's validate_orders, same dtypes as read_orders."""
        raw = pd.DataFrame(orders)
        for col, dtype in ID_DTYPES.items():
            raw[col] = raw[col].astype(dtype)
        return validate_orders(raw)

    def score(self, order: dict) -> dict:
        """Score one already schema-validated order with Task 1's validate_orders + score_orders."""
        b = self._require()
        df, problems = self._validated_frame([order])
        if len(problems):
            raise OrderRejectedError(problems.tolist())
        scores, status = score_orders(b, df)
        return {"anomaly_score": float(scores[0]), "status": str(status[0]),
                "threshold_used": float(b["threshold"]), "model_version": b["model_version"],
                "reasons": self.plain_reasons(df)}

    def score_batch(self, orders: list[dict]) -> list[dict]:
        """Score many schema-validated orders in one model call (POST /analyze-batch).

        Same code path as score(): validate_orders -> score_orders -> explain, so every row gets exactly the
        score it would get from /analyze-order. One result per order, in order: either
        {"anomaly_score", "status", "reasons"} or {"problems": [...]} if Task 1's validate_orders rejected it.
        """
        b = self._require()
        df, problems = self._validated_frame(orders)
        results: list[dict] = [{"problems": problems[i].split("; ")} if i in problems.index else {}
                               for i in df.index]
        ok = df.drop(index=problems.index)
        if len(ok):
            scores, status = score_orders(b, ok)
            for i, score, label, reasons in zip(ok.index, scores, status, self._reason_lists(ok)):
                results[i] = {"anomaly_score": float(score), "status": str(label), "reasons": reasons}
        return results

    def _reason_lists(self, df: pd.DataFrame) -> list[list[str]]:
        """Top-3 plain reasons per row: Task 1's features.explain, worded by app.reasons."""
        b = self._require()
        if "explain_stats" not in b:                    # older bundles: score still works, no reasons
            return [[] for _ in range(len(df))]
        X = b["pipeline"].named_steps["features"].transform(df)
        return [[to_sentence(feature, z) for feature, z in row] for row in explain(X, b["explain_stats"])]

    def plain_reasons(self, df: pd.DataFrame) -> list[str]:
        """Top-3 reasons for one validated order."""
        return self._reason_lists(df)[0]
