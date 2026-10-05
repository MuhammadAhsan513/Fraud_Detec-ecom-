"""Score an order CSV with a saved model bundle.

    python detect_anomalies.py --model model.joblib --input orders-sheet.csv --output scored_orders.csv

Output columns: row_id, order_uid, order_id, anomaly_score, status
  * row_id        = 0-based position of the row in the input file (order_id is NOT unique)
  * order_uid     = unique, content-based order id (see make_order_uid); stable across files
  * anomaly_score = -IsolationForest.score_samples(x): HIGHER = MORE anomalous (range ~0.3-0.8)
  * status        = "suspicious" if anomaly_score > bundle["threshold"] else "normal"
                    ("invalid" only with --on-invalid flag, for rows failing validation)
Every input row is scored (duplicates included), so output rows == input rows.
"""
import argparse
import hashlib
import sys

import joblib
import numpy as np
import pandas as pd

# validate_orders / InputValidationError are also imported from here by the notebooks and MANUAL.md.
from features import InputValidationError, anomaly_scores, read_orders, validate_orders

OUTPUT_FLOAT_FORMAT = "%.6f"
MAX_INVALID_ROWS_SHOWN = 10

# The 22 original columns, in a fixed order, that define an order's identity.
UID_COLUMNS = ["order_id", "order_timestamp", "customer_id", "customer_country", "customer_age",
               "account_age_days", "product_id", "product_name", "category", "quantity", "unit_price",
               "discount_pct", "shipping_cost", "tax_amount", "platform_fee", "order_value",
               "payment_method", "sales_channel", "device_type", "ip_country", "order_status", "coupon_used"]


def make_order_uid(df: pd.DataFrame) -> pd.Series:
    """Unique, deterministic id per order: 'U' + first 12 hex chars of SHA-1 over the 22 raw fields.

    Numbers are written as %.4f (so 1, 1.0 and "1" hash the same), text is stripped, missing = "".
    The same order content always gets the same uid, whatever its file or position. Exact duplicate
    rows (identical on all 22 fields) get a suffix by order of appearance: U..., U...-2, U...-3.
    """
    parts = []
    for col in UID_COLUMNS:
        values = df[col] if col in df.columns else pd.Series(pd.NA, index=df.index, dtype="object")
        as_number = pd.to_numeric(values, errors="coerce")
        as_text = values.astype("string").str.strip().fillna("")
        parts.append(as_text.where(as_number.isna(), as_number.map(lambda v: f"{v:.4f}")).astype(str))
    canonical = parts[0].str.cat(parts[1:], sep="|")
    base = canonical.map(lambda t: "U" + hashlib.sha1(t.encode()).hexdigest()[:12].upper())
    occurrence = base.groupby(base).cumcount()
    uid = base.where(occurrence == 0, base + "-" + (occurrence + 1).astype(str))
    if uid.duplicated().any():                      # 48-bit hash: collisions are ~impossible, but check
        raise RuntimeError("order_uid collision between different orders")
    return uid


def score_orders(bundle: dict, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """anomaly_score (higher = more anomalous) and status for already-validated rows."""
    scores = anomaly_scores(bundle["pipeline"], df)
    return scores, np.where(scores > bundle["threshold"], "suspicious", "normal")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="scored_orders.csv")
    ap.add_argument("--on-invalid", choices=["error", "flag"], default="error",
                    help="error: abort if any row fails validation (default); "
                         "flag: keep the row with status='invalid' and an empty score")
    return ap.parse_args(argv)


def build_output(bundle: dict, raw: pd.DataFrame, df: pd.DataFrame, problems: pd.Series) -> pd.DataFrame:
    """One output row per input row; rows that failed validation keep status 'invalid' and no score."""
    is_valid = ~df.index.isin(problems.index)
    out = pd.DataFrame({"row_id": raw.row_id, "order_uid": make_order_uid(raw), "order_id": raw.order_id,
                        "anomaly_score": np.nan, "status": "invalid"})
    if is_valid.any():
        scores, status = score_orders(bundle, df[is_valid])
        out.loc[is_valid, "anomaly_score"] = scores
        out.loc[is_valid, "status"] = status
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        bundle = joblib.load(args.model)
        raw = read_orders(args.input)
        df, problems = validate_orders(raw)
    except (OSError, InputValidationError, pd.errors.ParserError) as e:
        sys.exit(f"ERROR: {e}")
    if len(problems) and args.on_invalid == "error":
        sample = "\n".join(f"  row_id {raw.row_id[i]}: {r}" for i, r in problems.head(MAX_INVALID_ROWS_SHOWN).items())
        sys.exit(f"ERROR: {len(problems)} invalid row(s) (use --on-invalid flag to keep them):\n{sample}")

    out = build_output(bundle, raw, df, problems)
    out.to_csv(args.output, index=False, float_format=OUTPUT_FLOAT_FORMAT)

    counts = out.status.value_counts().to_dict()
    print(f"model {bundle['model_version']} | threshold {bundle['threshold']:.6f} | rows in {len(raw)} -> out {len(out)}")
    print(f"suspicious {counts.get('suspicious', 0)} | normal {counts.get('normal', 0)} | invalid {counts.get('invalid', 0)}"
          f" | wrote {args.output}")


if __name__ == "__main__":
    main()
