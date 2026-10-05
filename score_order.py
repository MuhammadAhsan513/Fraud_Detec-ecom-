"""Look up the anomaly score (and the reasons) of any order, order_id or product.

    python score_order.py --uid UCA5E4916EB5B             # one order by its unique id
    python score_order.py --row-id 20465                   # one order by position in the file
    python score_order.py --order-id ORD0020474            # all orders sharing this (non-unique) order_id
    python score_order.py --product P01005 [--top 20]      # every order of one product, summary + most anomalous
    python score_order.py --json '{"product_id": "P01005", "category": "Home", "quantity": 1, ...}'   # a new order

Scores are computed live with the saved model (same code path as detect_anomalies.py).
anomaly_score: higher = more anomalous; status = suspicious if score > the threshold stored in the model.
"""
import argparse
import json
import sys

import joblib
import numpy as np
import pandas as pd

from detect_anomalies import make_order_uid
from features import InputValidationError, clean_labels, explain, read_orders, validate_orders

SHOW = ["row_id", "order_uid", "order_id", "product_id", "category", "quantity", "unit_price", "discount_pct",
        "shipping_cost", "tax_amount", "platform_fee", "order_value", "account_age_days"]


def score_frame(bundle, raw):
    """Validate + score + explain a raw order frame. Returns raw columns plus score, status, reasons."""
    df, problems = validate_orders(raw)
    if len(problems):
        raise InputValidationError("; ".join(f"row {i}: {r}" for i, r in problems.items()))
    pipe = bundle["pipeline"]
    fe = pipe.named_steps["features"]
    X = fe.transform(df)
    out = raw.copy()
    out["product_median_price"] = fe.product_median_price(clean_labels(df)).round(2).to_numpy()
    out["anomaly_score"] = (-pipe.named_steps["iforest"].score_samples(X)).round(6)
    out["status"] = np.where(out.anomaly_score > bundle["threshold"], "suspicious", "normal")
    out["top3_reasons"] = [", ".join(f"{f} (z={z:+.1f})" for f, z in r) for r in explain(X, bundle["explain_stats"])]
    return out


def print_orders(t):
    for _, r in t.iterrows():
        print(f"\n  row_id {r.get('row_id', '-')} | order_uid {r.get('order_uid', '-')} | order_id {r.get('order_id', '-')}")
        print(f"    product {r.product_id} ({r.category}) | qty {r.quantity} @ {r.unit_price} (product median {r.product_median_price})"
              f" | discount {r.discount_pct} | shipping {r.shipping_cost} | tax {r.tax_amount} | fee {r.platform_fee}"
              f" | order_value {r.order_value} | account {r.account_age_days} d")
        print(f"    anomaly_score {r.anomaly_score:.6f} -> {r.status.upper()}   reasons: {r.top3_reasons}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--data", default="orders-sheet.csv", help="order file to search (not used with --json)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--uid", help="order_uid, e.g. UCA5E4916EB5B")
    g.add_argument("--row-id", type=int, help="0-based row position in --data")
    g.add_argument("--order-id", help="order_id (may match several orders)")
    g.add_argument("--product", help="product_id: score every order of this product")
    g.add_argument("--json", help="a new order as JSON (fields as in the CSV)")
    ap.add_argument("--top", type=int, default=10, help="--product: how many most-anomalous orders to show")
    a = ap.parse_args(argv)

    try:
        bundle = joblib.load(a.model)
        thr = bundle["threshold"]
        print(f"model {bundle['model_version']} | threshold {thr:.6f} (score above = suspicious)")

        if a.json:
            fields = json.loads(a.json)
            fields.setdefault("order_id", "NEW")
            order = pd.DataFrame([fields])
            order.insert(0, "order_uid", make_order_uid(order))
            print_orders(score_frame(bundle, order))
            return

        raw = read_orders(a.data)
        raw.insert(1, "order_uid", make_order_uid(raw))
        if a.uid:
            sel = raw[raw.order_uid == a.uid.strip().upper()]
        elif a.row_id is not None:
            sel = raw[raw.row_id == a.row_id]
        elif a.order_id:
            sel = raw[raw.order_id == a.order_id.strip()]
        else:
            sel = raw[raw.product_id == a.product.strip()]
        if sel.empty:
            sys.exit("no matching order found")

        t = score_frame(bundle, sel)
        if a.product:
            fe = bundle["pipeline"].named_steps["features"]
            print(f"\nproduct {a.product}: {len(t)} orders | median unit price (model lookup) "
                  f"{round(fe.prod_med_price_[a.product.strip()], 2) if a.product.strip() in fe.prod_med_price_ else 'unknown -> category/global fallback'}")
            print(f"  suspicious {(t.status == 'suspicious').sum()} | normal {(t.status == 'normal').sum()}"
                  f" | score min {t.anomaly_score.min():.4f} / median {t.anomaly_score.median():.4f} / max {t.anomaly_score.max():.4f}")
            print(f"\n  top {min(a.top, len(t))} most anomalous orders of this product:")
            t = t.sort_values("anomaly_score", ascending=False).head(a.top)
        print_orders(t)
    except (OSError, ValueError, KeyError) as e:      # includes InputValidationError and bad JSON
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
