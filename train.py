"""Train the order anomaly detector and save the model bundle.

    python train.py --data orders-sheet.csv --out model.joblib

Steps: read CSV -> validate -> drop exact duplicates (training only; row_id excluded from the
comparison) -> fit Pipeline[OrderFeatures, IsolationForest] -> threshold = quantile of the
TRAINING anomaly scores at 1 - contamination -> save bundle with metadata.
"""
import argparse
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from features import (CONTAMINATION, DEFAULT_FEATURE_SET, FEATURE_SETS, MAX_SAMPLES, N_ESTIMATORS,
                      RANDOM_SEED, InputValidationError, OrderFeatures, anomaly_scores, read_orders,
                      robust_stats, validate_orders)

REPORT_QUANTILES = [0.5, 0.9, 0.95, 0.98, 0.99, 0.999]     # printed only


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="orders-sheet.csv")
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--features", choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURE_SET)
    ap.add_argument("--contamination", type=float, default=CONTAMINATION)
    ap.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--model-version", default=None, help="default: iforest-<features>-<UTC date>")
    return ap.parse_args(argv)


def load_training_rows(path: str) -> tuple[pd.DataFrame, int]:
    """Valid, de-duplicated training rows, plus the number of exact duplicates dropped."""
    raw = read_orders(path)
    try:
        df, problems = validate_orders(raw)
    except InputValidationError as e:
        sys.exit(f"ERROR: {e}")
    if len(problems):
        print(f"WARNING: excluding {len(problems)} invalid rows from training, e.g. {problems.head(3).to_dict()}")
        df = df.drop(index=problems.index)

    is_duplicate = raw.loc[df.index].drop(columns="row_id").duplicated(keep="first").to_numpy()
    train_df = df[~is_duplicate]
    n_duplicates = int(is_duplicate.sum())
    print(f"rows {len(raw)} | invalid {len(problems)} | exact duplicates dropped {n_duplicates} | n_train {len(train_df)}")
    return train_df, n_duplicates


def fit_pipeline(train_df: pd.DataFrame, features: list[str], n_estimators: int, max_samples: int,
                 seed: int) -> Pipeline:
    """Fit OrderFeatures + IsolationForest on the training rows."""
    return Pipeline([
        ("features", OrderFeatures(tuple(features))),
        ("iforest", IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                    random_state=seed, n_jobs=-1)),
    ]).fit(train_df)


def histogram_valley(scores: np.ndarray, lo_q: float = 0.95, width: float = 0.01) -> tuple[float, dict]:
    """Least-populated score bin (3-bin smoothed) above the `lo_q` quantile: a cross-check only."""
    edges = np.arange(np.quantile(scores, lo_q), scores.max() + width, width)
    counts, _ = np.histogram(scores, bins=edges)
    smoothed = np.convolve(counts, np.ones(3) / 3, mode="same")
    i = int(np.argmin(smoothed[1:-1])) + 1    # ignore the edge bins
    return float((edges[i] + edges[i + 1]) / 2), dict(zip(edges[:-1].round(3).tolist(), counts.tolist()))


def build_bundle(pipe: Pipeline, train_df: pd.DataFrame, train_scores: np.ndarray, valley: float,
                 n_duplicates: int, args: argparse.Namespace) -> dict:
    """Model bundle: fitted pipeline, threshold and the metadata needed to reproduce it."""
    threshold = float(np.quantile(train_scores, 1 - args.contamination))
    X_train = pipe.named_steps["features"].transform(train_df)
    now = datetime.now(timezone.utc)
    return {
        "pipeline": pipe,
        "feature_order": list(FEATURE_SETS[args.features]),
        "feature_set": args.features,
        "threshold": threshold,
        "threshold_method": f"quantile(train anomaly_score, 1 - {args.contamination})",
        "threshold_valley": valley,
        "contamination": args.contamination,
        "score_definition": "anomaly_score = -IsolationForest.score_samples(x); higher = more anomalous",
        "hyperparameters": {"n_estimators": args.n_estimators, "max_samples": args.max_samples,
                            "random_state": args.seed},
        "explain_stats": robust_stats(X_train),
        "n_train": int(len(train_df)),
        "n_duplicates_dropped": n_duplicates,
        "train_data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "trained_at": now.isoformat(timespec="seconds"),
        "model_version": args.model_version or f"iforest-{args.features}-{now:%Y%m%d}",
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
    }


def print_report(bundle: dict, train_scores: np.ndarray, hist: dict, out_path: str) -> None:
    """Training-score distribution, threshold vs histogram valley, and where the bundle was saved."""
    features, threshold, valley = bundle["feature_order"], bundle["threshold"], bundle["threshold_valley"]
    quantiles = np.quantile(train_scores, REPORT_QUANTILES)
    print(f"features ({len(features)}): {features}")
    print("train score quantiles 50/90/95/98/99/99.9%:", np.round(quantiles, 4).tolist(),
          "| max", round(train_scores.max(), 4))
    print(f"threshold (q{1 - bundle['contamination']:.3f}) = {threshold:.4f} -> flags {(train_scores > threshold).sum()} train rows"
          f" | histogram valley = {valley:.4f} -> would flag {(train_scores > valley).sum()}")
    print("upper-tail histogram (bin start: count):", hist)
    print(f"saved {out_path} | model_version {bundle['model_version']}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_df, n_duplicates = load_training_rows(args.data)
    pipe = fit_pipeline(train_df, FEATURE_SETS[args.features], args.n_estimators, args.max_samples, args.seed)
    train_scores = anomaly_scores(pipe, train_df)
    valley, hist = histogram_valley(train_scores)
    bundle = build_bundle(pipe, train_df, train_scores, valley, n_duplicates, args)
    joblib.dump(bundle, args.out, compress=3)
    print_report(bundle, train_scores, hist, args.out)


if __name__ == "__main__":
    main()
