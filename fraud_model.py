"""
fraud_model.py
--------------
Supervised fraud classifier trained on the `is_fraud` ground-truth labels
from the Supabase `orders` table.

Workflow
--------
1. One-time training (run from the command line):
       python fraud_model.py --train

2. Fast repeated inference (import and call in your pipeline):
       from fraud_model import predict_order
       result = predict_order(order_dict)   # dict or pd.Series

The fitted model and feature-column list are saved to a single .joblib file
so training never needs to run again in production.  The model is loaded
from disk exactly once per Python process and cached in a module-level
variable — subsequent calls are pure in-memory inference.

Model choice: HistGradientBoostingClassifier
  - Histogram-based gradient boosting (same algorithm as LightGBM)
  - Tiny serialised size, sub-millisecond inference per row
  - Handles class imbalance natively via class_weight='balanced'
  - No extra dependencies beyond scikit-learn

Requirements:
    pip install supabase pandas scikit-learn joblib
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Config — mirrors fraud_check.py so both files stay in sync
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nxyitzzdnzfbdvkfbgcs.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_aOBNA_g5uQvnqygPsaYxqA_MxraaT0S")

MODEL_PATH = Path("fraud_classifier.joblib")   # serialised model + column list

HISTORICAL_CSV = Path(
    r"C:\Users\lmans\Programs\realtimeFrauddetection"
    r"\databricks-masterclass-main-modified-for-fraud-detection"
    r"\databricks-masterclass-main\projects\databricks-e2e-project"
    r"\00_synthetic_data\data\historical_orders.csv"
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Preprocessing — identical to fraud_check.py so features match at inference
# ---------------------------------------------------------------------------
PII_COLUMNS = [
    "items", "first_name", "last_name", "street", "zip_code",
    "phone_number", "mobile", "email_address", "card_number", "cvv",
    "is_fraud",   # label — extracted before preprocessing, not used as a feature
]

ONE_HOT_COLUMNS = [
    "order_type", "payment_method", "order_status",
    "title", "suffix", "city", "country", "state",
]


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline.  Does NOT touch the is_fraud column
    (it must be extracted by the caller before passing df here)."""
    df = df.copy()

    if "order_id" in df:
        df["order_id"] = (
            df["order_id"].astype(str)
            .str.replace("-", "", regex=False)
            .str.replace("ORD", "", regex=False)
        )
        df["order_id"] = pd.to_numeric(df["order_id"], errors="coerce")
        df["order_id"] = _minmax(df["order_id"])

    if "restaurant_id" in df:
        df["restaurant_id"] = (
            df["restaurant_id"].astype(str)
            .str.replace("-", "", regex=False)
            .str.replace(r"[a-zA-Z]", "", regex=True)
        )
        df["restaurant_id"] = pd.to_numeric(df["restaurant_id"], errors="coerce")

    if "customer_id" in df:
        df["customer_id"] = (
            df["customer_id"].astype(str)
            .str.replace("-", "", regex=False)
            .str.replace(r"[a-zA-Z]", "", regex=True)
        )
        df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce")

    for ts_col in ("timestamp", "created_at"):
        if ts_col in df:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            df[ts_col] = _minmax(ts.astype("int64"))

    if "total_amount" in df:
        df["total_amount"] = _minmax(pd.to_numeric(df["total_amount"], errors="coerce"))

    existing_cats = [c for c in ONE_HOT_COLUMNS if c in df.columns]
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, dtype=int)

    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns], errors="ignore")
    df = df.select_dtypes(include="number").fillna(0)
    return df


# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------
def _get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _fetch_all_orders(client: Client) -> pd.DataFrame:
    PAGE_SIZE = 1000
    rows, offset = [], 0
    while True:
        batch = client.table("orders").select("*").range(offset, offset + PAGE_SIZE - 1).execute().data
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.DataFrame(rows)


def _load_historical() -> pd.DataFrame:
    """Load historical_orders.csv and label every row as not-fraud (0).
    Returns an empty DataFrame if the file doesn't exist."""
    if not HISTORICAL_CSV.exists():
        print(f"  [warn] Historical CSV not found at {HISTORICAL_CSV} — skipping.")
        return pd.DataFrame()
    df = pd.read_csv(HISTORICAL_CSV)
    df["is_fraud"] = 0
    return df


# ---------------------------------------------------------------------------
# Module-level model cache — loaded from disk exactly once per process
# ---------------------------------------------------------------------------
_cache: dict | None = None   # {"model": clf, "columns": [...]}


def _load() -> dict:
    global _cache
    if _cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at '{MODEL_PATH}'. "
                "Run `python fraud_model.py --train` first."
            )
        _cache = joblib.load(MODEL_PATH)
    return _cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_and_save(eval: bool = True) -> None:
    """Fetch all labeled orders, train the classifier, and persist it.

    Parameters
    ----------
    eval : bool
        If True, prints a classification report on a held-out 20 % test set.
    """
    # --- Supabase orders only ---
    print("Fetching orders from Supabase…")
    client = _get_client()
    df_supabase = _fetch_all_orders(client)
    if df_supabase.empty:
        raise ValueError("No orders found in the Supabase `orders` table.")

    if "is_fraud" in df_supabase.columns:
        valid_labels = {0, 1}
        df_supabase = df_supabase.dropna(subset=["is_fraud"])
        df_supabase = df_supabase[
            pd.to_numeric(df_supabase["is_fraud"], errors="coerce").isin(valid_labels)
        ].reset_index(drop=True)
        print(f"  {len(df_supabase)} valid orders  |  fraud rate: {df_supabase['is_fraud'].astype(int).mean():.4%}")
    else:
        df_supabase["is_fraud"] = 0
        print(f"  No `is_fraud` column found — labeling all {len(df_supabase)} rows as 0.")

    df_raw = df_supabase
    y = df_raw["is_fraud"].astype(int)

    print(f"  Total training rows: {len(df_raw)}  |  overall fraud rate: {y.mean():.2%}")

    # Preprocess features (is_fraud is listed in PII_COLUMNS so it's dropped)
    X = _preprocess(df_raw)
    columns = list(X.columns)

    if eval:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, y_train = X, y

    # HistGradientBoostingClassifier: small, fast, handles imbalance
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    print("Training HistGradientBoostingClassifier…")
    clf.fit(X_train, y_train)

    # Compute a decision threshold calibrated so ~0.1 % of training orders
    # are flagged as fraud (99.9th percentile of fraud probability).
    import numpy as _np
    train_probs = clf.predict_proba(X_train)[:, 1]
    threshold = float(_np.percentile(train_probs, 99.9))
    print(f"Calibrated threshold (99.9th pct of training fraud probs): {threshold:.6f}")

    if eval:
        y_pred = (clf.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        print("\nEvaluation on held-out 20 % test set (threshold={:.6f}):".format(threshold))
        print(classification_report(y_test, y_pred, target_names=["NOT FRAUD", "FRAUD"]))

    payload = {
        "model": clf,
        "columns": columns,
        "threshold": threshold,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, MODEL_PATH, compress=3)
    print(f"Model saved → {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")

    # Invalidate the in-process cache so the next predict_order call re-loads
    global _cache
    _cache = None


def predict_order(order: Union[dict, "pd.Series"]) -> dict:
    """Predict fraud for a single incoming order.

    Parameters
    ----------
    order : dict or pd.Series
        A raw (unprocessed) order row exactly as returned by Supabase.

    Returns
    -------
    dict with keys:
        is_fraud         – bool
        label            – "FRAUD" | "NOT FRAUD"
        fraud_probability – float in [0, 1]  (confidence in the FRAUD class)
    """
    payload = _load()
    clf: HistGradientBoostingClassifier = payload["model"]
    columns: list[str] = payload["columns"]

    # Wrap in a single-row DataFrame for preprocessing
    if isinstance(order, pd.Series):
        df = order.to_frame().T.reset_index(drop=True)
    else:
        df = pd.DataFrame([order])

    features = _preprocess(df)

    # Align columns to exactly what the model was trained on:
    #   - missing one-hot columns → 0  (category unseen at training)
    #   - extra columns → dropped
    features = features.reindex(columns=columns, fill_value=0)

    prob = float(clf.predict_proba(features)[0][1])   # P(fraud)
    threshold = float(payload.get("threshold", 0.5))
    pred = prob >= threshold

    return {
        "is_fraud": bool(pred),
        "label": "FRAUD" if pred else "NOT FRAUD",
        "fraud_probability": prob,
        "threshold": threshold,
    }


def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of raw orders in one vectorised call.

    Returns the input DataFrame with three new columns appended:
        is_fraud_pred, label_pred, fraud_probability
    """
    payload = _load()
    clf: HistGradientBoostingClassifier = payload["model"]
    columns: list[str] = payload["columns"]

    threshold = float(payload.get("threshold", 0.5))
    features = _preprocess(df_raw).reindex(columns=columns, fill_value=0)
    probs = clf.predict_proba(features)[:, 1]
    preds = probs >= threshold

    out = df_raw.copy()
    out["is_fraud_pred"] = preds
    out["label_pred"] = ["FRAUD" if p else "NOT FRAUD" for p in preds]
    out["fraud_probability"] = probs
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--train" in sys.argv:
        train_and_save(eval=True)
    elif "--score" in sys.argv:
        # Quick smoke-test: score the first unprocessed order
        client = _get_client()
        df = _fetch_all_orders(client)
        if df.empty:
            print("No orders found.")
        else:
            sample = df.iloc[0].to_dict()
            result = predict_order(sample)
            print(f"order_id={sample.get('order_id')}  →  {result}")
    else:
        print("Usage:")
        print("  python fraud_model.py --train    # one-time training")
        print("  python fraud_model.py --score    # smoke-test on first order")
