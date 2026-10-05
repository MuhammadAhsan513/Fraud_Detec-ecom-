"""Shared feature engineering for the order anomaly detector.

Imported by train.py, detect_anomalies.py and score_order.py, so that the pickled pipeline in
model.joblib can be loaded anywhere (`joblib.load` needs this module). Holds everything training and
scoring must agree on: model configuration, feature lists, input loading/validation, the feature
transformer, the anomaly-score definition and the robust-z explanations.

Design rules (see REPORT_preprocessing.md):
  * label cleaning uses a FIXED mapping, never "most frequent spelling";
    unseen / dirty labels pass through and never crash;
  * fit() learns lookups (product median price etc.) from TRAINING rows only;
  * transform() is purely row-wise, so scoring one row == scoring it inside a batch;
  * no clipping of raw values; clipping only inside ratios to avoid log(0) / x/0.
"""
from os import PathLike

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------- model configuration
RANDOM_SEED = 42                 # IsolationForest random_state
CONTAMINATION = 0.018            # threshold = quantile(train scores, 1 - CONTAMINATION)
N_ESTIMATORS = 300               # IsolationForest trees
MAX_SAMPLES = 4096               # IsolationForest subsample size per tree
DEFAULT_FEATURE_SET = "reduced11"

# ---------------------------------------------------------------- feature lists
FULL = ["log_quantity", "log_unit_price", "discount_pct", "log_shipping", "log1p_tax_rate",
        "log_fee_rate", "price_vs_product", "log_order_value", "log_value_ratio",
        "log_acct_age", "log_ship_ratio", "log_fee_to_value"]
REDUCED_11 = [f for f in FULL if f != "log_ship_ratio"]
# Non-circular ablation set: no order_value, no residual.
NO_VALUE = ["log_quantity", "log_unit_price", "discount_pct", "log_shipping", "log1p_tax_rate",
            "log_fee_rate", "price_vs_product", "log_acct_age", "log_ship_ratio"]
FEATURE_SETS = {"full12": FULL, "reduced11": REDUCED_11, "no_value": NO_VALUE}

# Lower bounds used only inside ratios / logs, never on raw values.
MIN_SUBTOTAL = 0.01
MIN_FEE_RATE = 1e-4
MIN_FEE_TO_VALUE = 1e-6
IQR_TO_SIGMA = 1.349             # IQR of a normal distribution, in standard deviations

# ---------------------------------------------------------------- input columns
CAT_COLS = ["customer_country", "ip_country", "category", "payment_method",
            "sales_channel", "device_type", "order_status", "coupon_used"]
FIXED_MAP = {"payment_method": {"paypal": "PayPal"}, "sales_channel": {"web": "Web"}}

# Raw columns the feature transformer actually reads.
NUMERIC_INPUTS = ["quantity", "unit_price", "discount_pct", "shipping_cost", "tax_amount",
                  "platform_fee", "order_value", "account_age_days"]
NULLABLE_INPUTS = ["discount_pct", "shipping_cost"]          # imputed inside transform()
MODEL_INPUTS = NUMERIC_INPUTS + ["product_id", "category"]
REQUIRED_COLUMNS = ["order_id"] + MODEL_INPUTS
ID_DTYPES = {"order_id": "string", "product_id": "string"}   # read as text: leading zeros matter

# (column, predicate on non-null values, human-readable rule)
VALUE_RULES = [
    ("quantity", lambda s: s >= 1, ">= 1"),
    ("unit_price", lambda s: s > 0, "> 0"),
    ("discount_pct", lambda s: (s >= 0) & (s < 100), "in [0, 100)"),
    ("shipping_cost", lambda s: s >= 0, ">= 0"),
    ("tax_amount", lambda s: s >= 0, ">= 0"),
    ("platform_fee", lambda s: s >= 0, ">= 0"),
    ("order_value", lambda s: s > 0, "> 0"),
    ("account_age_days", lambda s: s >= 0, ">= 0"),
]


# ---------------------------------------------------------------- loading and validation
class InputValidationError(ValueError):
    """Raised when an order table is missing required columns."""


def read_orders(path: str | PathLike) -> pd.DataFrame:
    """Read an order CSV and prepend `row_id` (0-based position in the file)."""
    raw = pd.read_csv(path, dtype=ID_DTYPES)
    raw.insert(0, "row_id", np.arange(len(raw)))
    return raw


def validate_orders(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Check columns and values before the pipeline.

    Returns (clean_df, problems) where clean_df has numeric columns coerced to float and
    problems is a Series (index aligned to df) with a reason string for each invalid row.
    Raises InputValidationError if required columns are missing.
    Extreme-but-valid values (qty 99, discount 95 %) are deliberately allowed: they are the signal.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError(f"missing required column(s): {missing}")
    df = df.copy()
    reasons = pd.Series("", index=df.index, dtype=object)

    def add_reason(mask: pd.Series, message: str) -> None:
        reasons.loc[mask] = reasons.loc[mask] + message + "; "

    for col in NUMERIC_INPUTS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        add_reason(coerced.isna() & df[col].notna(), f"{col} not numeric")
        if col not in NULLABLE_INPUTS:
            add_reason(df[col].isna(), f"{col} missing")
        add_reason(np.isinf(coerced), f"{col} infinite")
        df[col] = coerced.astype(float)
    for col, is_valid, rule in VALUE_RULES:
        values = df[col]
        add_reason(values.notna() & np.isfinite(values) & ~is_valid(values), f"{col} must be {rule}")
    for col in ["product_id", "category"]:
        add_reason(df[col].isna() | (df[col].astype("string").str.strip() == ""), f"{col} missing")
    return df, reasons.str.rstrip("; ")[reasons != ""]


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and canonicalise categorical labels with a fixed mapping.

    Columns that are absent are skipped (a single API order may omit display-only fields).
    """
    df = df.copy()
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        labels = df[col].astype("string").str.strip()
        if col == "category":
            labels = labels.str.title()             # 'toys' -> 'Toys' (all categories are single words)
        elif col in FIXED_MAP:
            labels = labels.str.lower().map(FIXED_MAP[col]).fillna(labels)
        df[col] = labels
    return df


# ---------------------------------------------------------------- feature transformer
class OrderFeatures(BaseEstimator, TransformerMixin):
    """Raw order rows -> engineered numeric features (row-wise; lookups learned in fit)."""

    def __init__(self, features=tuple(REDUCED_11)):
        self.features = features

    def fit(self, df: pd.DataFrame, y=None) -> "OrderFeatures":
        """Learn median-price lookups and the shipping median from training rows."""
        orders = clean_labels(df)
        self.prod_med_price_ = orders.groupby("product_id").unit_price.median().to_dict()
        self.cat_med_price_ = orders.groupby("category").unit_price.median().to_dict()
        self.global_med_price_ = float(orders.unit_price.median())
        self.ship_median_ = float(orders.shipping_cost.median())
        return self

    def product_median_price(self, orders: pd.DataFrame) -> pd.Series:
        """Product median unit price, falling back to category median, then global median."""
        return (orders.product_id.map(self.prod_med_price_)
                .fillna(orders.category.map(self.cat_med_price_))
                .fillna(self.global_med_price_).astype(float))

    def _numeric_inputs(self, orders: pd.DataFrame) -> dict[str, pd.Series]:
        """Float inputs with discount/shipping imputed, plus the derived subtotal and expected value."""
        discount = orders.discount_pct.astype(float).fillna(0.0)
        shipping = orders.shipping_cost.astype(float).fillna(self.ship_median_)
        quantity = orders.quantity.astype(float)
        unit_price = orders.unit_price.astype(float)
        tax = orders.tax_amount.astype(float)
        subtotal = (quantity * unit_price * (1 - discount / 100)).clip(lower=MIN_SUBTOTAL)
        return {
            "quantity": quantity, "unit_price": unit_price, "discount": discount, "shipping": shipping,
            "tax": tax, "fee": orders.platform_fee.astype(float), "value": orders.order_value.astype(float),
            "subtotal": subtotal, "expected_value": subtotal + shipping + tax,
            # the order_value residual is trusted only if nothing was imputed
            "residual_trusted": orders.discount_pct.notna() & orders.shipping_cost.notna(),
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineered features for each row, columns in `self.features` order."""
        orders = clean_labels(df)
        v = self._numeric_inputs(orders)
        X = pd.DataFrame(index=orders.index)
        X["log_quantity"] = np.log1p(v["quantity"])
        X["log_unit_price"] = np.log(v["unit_price"])
        X["discount_pct"] = v["discount"]
        X["log_shipping"] = np.log1p(v["shipping"])
        X["log1p_tax_rate"] = np.log1p(v["tax"] / v["subtotal"])
        X["log_fee_rate"] = np.log((v["fee"] / v["subtotal"]).clip(lower=MIN_FEE_RATE))
        X["log_ship_ratio"] = np.log1p(v["shipping"] / v["subtotal"])
        X["price_vs_product"] = np.log(v["unit_price"] / self.product_median_price(orders))
        X["log_order_value"] = np.log(v["value"])
        X["log_fee_to_value"] = np.log((v["fee"] / v["value"]).clip(lower=MIN_FEE_TO_VALUE))
        X["log_value_ratio"] = np.where(v["residual_trusted"], np.log(v["value"] / v["expected_value"]), 0.0)
        X["log_acct_age"] = np.log1p(orders.account_age_days.astype(float))
        return X[list(self.features)]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.features, dtype=object)


# ---------------------------------------------------------------- scoring
def anomaly_scores(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """anomaly_score = -IsolationForest.score_samples(x): higher = more anomalous."""
    return -pipeline.score_samples(df)


# ---------------------------------------------------------------- explanations
def robust_stats(X: pd.DataFrame, floor: float = 0.01) -> dict[str, dict[str, float]]:
    """Training median and robust scale (IQR/1.349, floored) per feature, for explanations."""
    median = X.median()
    scale = np.maximum((X.quantile(0.75) - X.quantile(0.25)) / IQR_TO_SIGMA, floor)
    return {"median": median.to_dict(), "scale": scale.to_dict()}


def explain(X: pd.DataFrame, stats: dict, top: int = 3) -> list[list[tuple[str, float]]]:
    """Top-`top` features by |robust z| for each row of an engineered feature frame X.

    Returns a list (one per row) of [(feature, z), ...]. Row-wise, so usable from the API.
    """
    median = pd.Series(stats["median"])[X.columns]
    scale = pd.Series(stats["scale"])[X.columns]
    Z = (X - median) / scale
    top_cols = np.argsort(-Z.abs().to_numpy(), axis=1)[:, :top]
    names = np.asarray(X.columns)
    return [[(names[j], round(float(Z.iat[i, j]), 1)) for j in top_cols[i]] for i in range(len(X))]
