"""Unit tests for features.py. Run: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features import FULL, REDUCED_11, OrderFeatures, clean_labels, explain, robust_stats  # noqa: E402


def make_orders(n=400, seed=0):
    """Small synthetic, formula-consistent order table with the real column layout."""
    rs = np.random.RandomState(seed)
    products = [f"P{i:05d}" for i in range(20)]
    cats = ["Home", "Toys", "Electronics", "Books"]
    base_price = {p: rs.uniform(5, 200) for p in products}
    pid = rs.choice(products, n)
    qty = rs.randint(1, 5, n)
    price = np.array([base_price[p] for p in pid]) * rs.uniform(0.9, 1.1, n)
    disc = rs.choice([0, 0, 10, 20], n).astype(float)
    ship = rs.uniform(0, 20, n).round(2)
    sub = qty * price * (1 - disc / 100)
    tax = (sub * rs.randint(0, 21, n) / 100).round(2)
    return pd.DataFrame({
        "order_id": [f"ORD{i:07d}" for i in range(n)],
        "customer_id": [f"C{i % 50:06d}" for i in range(n)],
        "customer_country": "France", "ip_country": "Italy",
        "account_age_days": rs.randint(30, 3000, n),
        "product_id": pid,
        "category": [cats[int(p[1:]) % 4] for p in pid],
        "quantity": qty, "unit_price": price.round(2), "discount_pct": disc,
        "shipping_cost": ship, "tax_amount": tax,
        "platform_fee": (sub * 0.115).round(2),
        "order_value": (sub + ship + tax).round(2),
        "payment_method": "Card", "sales_channel": "Web", "device_type": "Android",
        "order_status": "Shipped", "coupon_used": "No",
    })


@pytest.fixture(scope="module")
def train():
    return make_orders()


@pytest.fixture(scope="module")
def pipe(train):
    return Pipeline([("features", OrderFeatures(tuple(REDUCED_11))),
                     ("iforest", IsolationForest(n_estimators=50, random_state=0))]).fit(train)


def test_clean_labels_dirty_values():
    df = pd.DataFrame({"category": [" toys", "ELECTRONICS ", "Home"],
                       "payment_method": ["paypal", " PayPal ", "Card"],
                       "sales_channel": ["web ", "Web", "Mobile App"]})
    out = clean_labels(df)
    assert out.category.tolist() == ["Toys", "Electronics", "Home"]
    assert out.payment_method.tolist() == ["PayPal", "PayPal", "Card"]
    assert out.sales_channel.tolist() == ["Web", "Web", "Mobile App"]


def test_clean_labels_unseen_and_missing_do_not_crash():
    df = pd.DataFrame({"payment_method": ["Crypto", None], "category": ["automotive ", np.nan]})
    out = clean_labels(df)                     # other CAT_COLS absent -> skipped
    assert out.payment_method.iloc[0] == "Crypto"
    assert out.category.iloc[0] == "Automotive"
    assert out.payment_method.isna().iloc[1] and out.category.isna().iloc[1]


def test_unseen_product_falls_back_to_category_then_global(train, pipe):
    fe = pipe.named_steps["features"]
    row = train.iloc[[0]].copy()
    row["product_id"] = "P99999"
    pm = fe.product_median_price(clean_labels(row)).iloc[0]
    assert pm == pytest.approx(fe.cat_med_price_[row.category.iloc[0]])
    row["category"] = "automotive "
    pm = fe.product_median_price(clean_labels(row)).iloc[0]
    assert pm == pytest.approx(fe.global_med_price_)
    assert np.isfinite(pipe.score_samples(row)).all()


def test_nan_discount_and_shipping_imputed_and_residual_zeroed(train, pipe):
    fe = OrderFeatures(tuple(FULL)).fit(train)
    rows = train.iloc[:3].copy()
    rows.loc[rows.index[0], "discount_pct"] = np.nan
    rows.loc[rows.index[1], "shipping_cost"] = np.nan
    X = fe.transform(rows)
    assert not X.isna().any().any()
    assert X.log_value_ratio.iloc[0] == 0.0 and X.log_value_ratio.iloc[1] == 0.0
    assert X.discount_pct.iloc[0] == 0.0
    assert X.log_shipping.iloc[1] == pytest.approx(np.log1p(fe.ship_median_))
    assert np.isfinite(pipe.score_samples(rows)).all()


def test_residual_is_zero_for_consistent_and_large_for_tampered(train):
    fe = OrderFeatures(tuple(FULL)).fit(train)
    rows = train.iloc[:2].copy()
    rows.loc[rows.index[1], "order_value"] *= 50
    r = fe.transform(rows).log_value_ratio
    assert abs(r.iloc[0]) < 1e-3
    assert r.iloc[1] == pytest.approx(np.log(50), abs=1e-2)


def test_zero_fee_and_tiny_subtotal_stay_finite(train):
    fe = OrderFeatures(tuple(FULL)).fit(train)
    rows = train.iloc[:1].copy()
    rows["platform_fee"] = 0.0          # log(0) guarded by clip
    rows["discount_pct"] = 99.999       # subtotal -> clipped at 0.01
    assert np.isfinite(fe.transform(rows).to_numpy()).all()


def test_one_row_equals_batch(train, pipe):
    batch = pipe.score_samples(train)
    single = np.array([pipe.score_samples(train.iloc[[i]])[0] for i in range(len(train))])
    np.testing.assert_array_equal(single, batch)


def test_output_columns_follow_feature_order(train, pipe):
    X = pipe.named_steps["features"].transform(train)
    assert list(X.columns) == REDUCED_11
    assert len(X) == len(train)


def test_explain_top3(train):
    fe = OrderFeatures(tuple(FULL)).fit(train)
    X = fe.transform(train)
    stats = robust_stats(X)
    row = train.iloc[[0]].copy()
    row["account_age_days"] = 1          # touches exactly one feature
    (top,) = explain(fe.transform(row), stats)
    assert len(top) == 3 and top[0][0] == "log_acct_age" and top[0][1] < -3


# ---------------------------------------------------------------- order_uid
from detect_anomalies import make_order_uid  # noqa: E402


def test_order_uid_unique_with_duplicates_and_repeated_order_ids(train):
    df = pd.concat([train, train.iloc[[0, 0]]], ignore_index=True)   # 2 extra exact copies of row 0
    df.loc[5, "order_id"] = df.loc[6, "order_id"]                     # a repeated order_id, different order
    uid = make_order_uid(df)
    assert uid.is_unique
    assert uid.iloc[-2] == uid.iloc[0] + "-2" and uid.iloc[-1] == uid.iloc[0] + "-3"


def test_order_uid_stable_across_position_and_number_formatting(train):
    uid = make_order_uid(train)
    assert make_order_uid(train.iloc[::-1]).sort_index().equals(uid)  # order of rows does not matter
    row = train.iloc[[3]].astype(object)
    row["quantity"] = str(row.quantity.iloc[0]) + ".0"              # "2.0" (text) == 2 (int)
    assert make_order_uid(row).iloc[0] == uid.iloc[3]
