"""Request / response models (Pydantic v2).

The request mirrors one row of orders-sheet.csv. Fields the feature transformer reads
(features.REQUIRED_COLUMNS) are fed to the model; the context fields are validated but not used by it.

Categorical policy: labels are normalised the way training did (strip whitespace, case-insensitive, so
'paypal' -> 'PayPal', 'web ' -> 'Web', 'toys' -> 'Toys'), then validated against the values seen in
the training data (orders-sheet.csv). Anything else is rejected with 422.
"""
from datetime import datetime, timedelta, timezone
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

import features

# ---------------------------------------------------------------- allowed values (seen in training data)
CATEGORIES = ["Automotive", "Beauty", "Electronics", "Fashion", "Garden", "Home", "Office", "Sports", "Tools", "Toys"]
COUNTRIES = ["Australia", "Canada", "France", "Germany", "Ireland", "Italy", "Netherlands", "Spain", "UK", "US"]
PAYMENT_METHODS = ["Apple Pay", "Bank Transfer", "Card", "Google Pay", "PayPal"]
SALES_CHANNELS = ["Marketplace", "Mobile App", "Web"]
DEVICE_TYPES = ["Android", "Desktop", "Tablet", "iPhone"]
ORDER_STATUSES = ["Cancelled", "Completed", "Pending", "Refunded", "Shipped"]
COUPON_USED = ["No", "Yes"]
ALLOWED = {"category": CATEGORIES, "customer_country": COUNTRIES, "ip_country": COUNTRIES,
           "payment_method": PAYMENT_METHODS, "sales_channel": SALES_CHANNELS, "device_type": DEVICE_TYPES,
           "order_status": ORDER_STATUSES, "coupon_used": COUPON_USED}
_CANONICAL = {col: {v.lower(): v for v in values} for col, values in ALLOWED.items()}

# ---------------------------------------------------------------- bounds
DISCOUNT_MIN, DISCOUNT_MAX = 0.0, 95.0      # range of discount_pct in the training data
CUSTOMER_AGE_MIN, CUSTOMER_AGE_MAX = 18, 120
MAX_AMOUNT = 1e9                           # sanity cap on money fields (training max order_value ~2e5)
MAX_QUANTITY = 1_000_000
MAX_ACCOUNT_AGE_DAYS = 36_500              # 100 years
FUTURE_TOLERANCE = timedelta(minutes=5)    # clock skew allowed on order_timestamp
ID_PATTERN = r"^[A-Za-z0-9_-]+$"

Id = Annotated[str, Field(strict=True, min_length=1, max_length=64, pattern=ID_PATTERN)]
Label = Annotated[str, Field(strict=True, min_length=1, max_length=64)]
Text = Annotated[str, Field(strict=True, min_length=1, max_length=200)]
Money = Annotated[float, Field(strict=True, ge=0, le=MAX_AMOUNT, allow_inf_nan=False)]
PositiveMoney = Annotated[float, Field(strict=True, gt=0, le=MAX_AMOUNT, allow_inf_nan=False)]


class OrderRequest(BaseModel):
    """One order. All listed fields are required (keys must be present); `| None` fields accept null."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # --- read by the feature transformer (features.REQUIRED_COLUMNS)
    order_id: Id
    product_id: Id
    category: Label
    quantity: Annotated[int, Field(strict=True, ge=1, le=MAX_QUANTITY)]
    unit_price: PositiveMoney
    discount_pct: Annotated[float, Field(strict=True, ge=DISCOUNT_MIN, le=DISCOUNT_MAX, allow_inf_nan=False)] | None
    shipping_cost: Money | None
    tax_amount: Money
    platform_fee: Money
    order_value: PositiveMoney                       # > 0: the transformer takes log(order_value)
    account_age_days: Annotated[int, Field(strict=True, ge=0, le=MAX_ACCOUNT_AGE_DAYS)]
    # --- context: validated, not used by the model
    order_timestamp: Annotated[str, Field(strict=True, min_length=1, max_length=40)]
    customer_age: Annotated[int, Field(strict=True, ge=CUSTOMER_AGE_MIN, le=CUSTOMER_AGE_MAX)] | None
    customer_country: Label
    ip_country: Label | None
    payment_method: Label
    sales_channel: Label
    device_type: Label | None
    order_status: Label
    coupon_used: Label
    # --- optional, never logged
    customer_id: Id | None = None
    product_name: Text | None = None

    @field_validator(*ALLOWED, mode="after")
    @classmethod
    def _known_label(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        canonical = _CANONICAL[info.field_name].get(value.lower())
        if canonical is None:
            raise ValueError(f"must be one of {ALLOWED[info.field_name]}")
        return canonical

    @field_validator("order_timestamp", mode="after")
    @classmethod
    def _iso_timestamp_not_future(cls, value: str) -> str:
        try:
            ts = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError("must be an ISO 8601 datetime, e.g. 2025-09-12T02:16:00Z") from None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)         # naive timestamps are taken as UTC
        if ts > datetime.now(timezone.utc) + FUTURE_TOLERANCE:
            raise ValueError("must not be in the future")
        return value

    def model_inputs(self) -> dict:
        """Raw columns for the Task 1 pipeline (order_id + features.MODEL_INPUTS)."""
        return self.model_dump(include={"order_id", "product_id", "category", "quantity", "unit_price",
                                         "discount_pct", "shipping_cost", "tax_amount", "platform_fee",
                                         "order_value", "account_age_days"})


# ---------------------------------------------------------------- required fields (batch CSV and the UI)
# A row with a blank cell in any of these is "missing_data" and is not scored.
#   * MODEL_REQUIRED_FIELDS: what features.py truly needs: its REQUIRED_COLUMNS minus the inputs it imputes
#     (features.NULLABLE_INPUTS = discount_pct, shipping_cost).
#   * CONTEXT_REQUIRED_FIELDS: not used by the model, but OrderRequest (shared with /analyze-order) does not accept
#     them as null, so a row without them cannot pass the same schema.
MODEL_REQUIRED_FIELDS = [c for c in features.REQUIRED_COLUMNS if c not in features.NULLABLE_INPUTS]
CONTEXT_REQUIRED_FIELDS = ["order_timestamp", "customer_country", "payment_method", "sales_channel",
                           "order_status", "coupon_used"]
REQUIRED_FIELDS = MODEL_REQUIRED_FIELDS + CONTEXT_REQUIRED_FIELDS
# The CSV header must contain these. discount_pct / shipping_cost cells may be blank (the model imputes them).
REQUIRED_COLUMNS = REQUIRED_FIELDS + list(features.NULLABLE_INPUTS)
INT_FIELDS = {"quantity", "account_age_days", "customer_age"}
FLOAT_FIELDS = {"unit_price", "discount_pct", "shipping_cost", "tax_amount", "platform_fee", "order_value"}


class AnalyzeResponse(BaseModel):
    order_id: str
    anomaly_score: float
    status: str
    threshold_used: float
    model_version: str
    reasons: list[str]                               # top-3 plain-language reasons (features.explain)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    feature_names: list[str]
    threshold: float
    threshold_method: str | None
    score_definition: str | None
    status_labels: list[str]
    trained_with: dict[str, str | None]
    n_training_rows: int | None
    trained_at: str | None
    allowed_values: dict[str, list[str]]             # accepted labels per categorical field


class ErrorDetail(BaseModel):
    field: str
    reason: str


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: list[ErrorDetail]


class ErrorResponse(BaseModel):
    error: ErrorBody


# Columns that may be absent from a CSV (absent = null for every row): every other request field.
OPTIONAL_COLUMNS = [f for f in OrderRequest.model_fields if f not in REQUIRED_COLUMNS]
