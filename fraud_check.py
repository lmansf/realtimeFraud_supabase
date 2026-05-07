"""
fraud_check.py
--------------
Pulls the most recent unprocessed row from the Supabase `orders` table,
runs it through an Isolation Forest fitted on historical data
(mirroring the preprocessing pipeline from Fraud_Detection_in_Realtime.ipynb),
and reports whether the row is FRAUD or NOT FRAUD.

Notification: writes a structured event to stdout + a local JSONL log.
Swap `notify()` for email/SMS/Slack/Supabase-update as needed (see notes
at the bottom of the file).

Requirements:
    pip install supabase pandas scikit-learn
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from supabase import Client, create_client

# Supervised model — imported lazily so fraud_check.py works standalone
try:
    from fraud_model import predict_order as _supervised_predict, MODEL_PATH as _MODEL_PATH
    _SUPERVISED_AVAILABLE = True
except ImportError:
    _supervised_predict = None   # type: ignore[assignment]
    _MODEL_PATH = Path("fraud_classifier.joblib")
    _SUPERVISED_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUPABASE_URL          = os.getenv("SUPABASE_URL", "https://nxyitzzdnzfbdvkfbgcs.supabase.co")
SUPABASE_KEY          = os.getenv("SUPABASE_KEY", "sb_publishable_aOBNA_g5uQvnqygPsaYxqA_MxraaT0S")
# Service-role key bypasses RLS — required for backend writes.
# Get it from: Supabase dashboard → Project Settings → API → service_role secret
SUPABASE_SERVICE_KEY  = os.getenv("SUPABASE_SERVICE_KEY", "")

CONTAMINATION = 0.05          # matches the notebook
RANDOM_STATE  = 42
PROCESSED_LOG = Path("processed_orders.txt")   # tracks which order_ids we've already scored
EVENT_LOG     = Path("fraud_events.jsonl")     # one JSON event per scored row


# ---------------------------------------------------------------------------
# Preprocessing — keep this aligned with the notebook
# ---------------------------------------------------------------------------
PII_COLUMNS = [
    "items", "first_name", "last_name", "street", "zip_code",
    "phone_number", "mobile", "email_address", "card_number", "cvv",
    "is_fraud",  # label column — must never be used as a feature
]

ONE_HOT_COLUMNS = [
    "order_type", "payment_method", "order_status",
    "title", "suffix", "city", "country", "state",
]


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1]; returns 0.0 if the column is constant."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.0, index=series.index)
    return (series - lo) / (hi - lo)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the notebook's feature engineering, end-to-end."""
    df = df.copy()

    # Strip non-numeric chars from id columns, then scale order_id
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

    # Scale timestamps
    for ts_col in ["timestamp", "created_at"]:
        if ts_col in df:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            df[ts_col] = _minmax(ts.astype("int64"))

    # Scale total_amount
    if "total_amount" in df:
        df["total_amount"] = _minmax(pd.to_numeric(df["total_amount"], errors="coerce"))

    # One-hot encode categoricals
    existing = [c for c in ONE_HOT_COLUMNS if c in df.columns]
    if existing:
        df = pd.get_dummies(df, columns=existing, dtype=int)

    # Drop PII / unstructured columns
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns], errors="ignore")

    # Anything that's still non-numeric (e.g. stray object cols) gets dropped —
    # IsolationForest only accepts numeric input.
    df = df.select_dtypes(include="number").fillna(0)

    return df


# ---------------------------------------------------------------------------
# Supabase + state helpers
# ---------------------------------------------------------------------------
def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_service_client() -> Client:
    """Returns a client that uses the service-role key (bypasses RLS)."""
    key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
    return create_client(SUPABASE_URL, key)


def load_processed_ids() -> set[str]:
    if not PROCESSED_LOG.exists():
        return set()
    return {line.strip() for line in PROCESSED_LOG.read_text().splitlines() if line.strip()}


def mark_processed(order_id: str) -> None:
    with PROCESSED_LOG.open("a") as f:
        f.write(f"{order_id}\n")


def fetch_orders(client: Client) -> pd.DataFrame:
    """Fetch all rows from `orders`, paginating past Supabase's 1000-row default limit."""
    PAGE_SIZE = 1000
    all_rows = []
    offset = 0
    while True:
        resp = client.table("orders").select("*").range(offset, offset + PAGE_SIZE - 1).execute()
        batch = resp.data
        all_rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_latest_unprocessed():
    client = get_client()
    df_raw = fetch_orders(client)
    if df_raw.empty:
        print("No rows in `orders` table.")
        return None

    processed_ids = load_processed_ids()

    # Sort by created_at (fall back to timestamp) descending and pick the
    # newest order_id that hasn't been scored yet.
    sort_col = "created_at" if "created_at" in df_raw.columns else "timestamp"
    df_raw = df_raw.sort_values(by=sort_col, ascending=False)

    target_row = next(
        (row for _, row in df_raw.iterrows() if row["order_id"] not in processed_ids),
        None,
    )
    if target_row is None:
        print("All orders have already been processed.")
        return None

    target_order_id = target_row["order_id"]
    print(f"Scoring order_id={target_order_id} (created_at={target_row.get(sort_col)})")

    # Preprocess the FULL dataset together so that scaling + one-hot columns
    # are consistent for the target row. Isolation Forest is unsupervised and
    # needs the whole population to know what 'normal' looks like.
    df_features = preprocess(df_raw)

    # Locate the target row's positional index in the processed frame
    target_pos = df_raw.index.get_loc(target_row.name)

    # Fit on the rest of the data, score just the target row
    train = df_features.drop(df_features.index[target_pos])
    target_features = df_features.iloc[[target_pos]]

    # backup: isolation forest
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    iso.fit(train)

    prediction = int(iso.predict(target_features)[0])     #  1 = normal, -1 = anomaly
    score      = float(iso.score_samples(target_features)[0])  # lower = more anomalous
    is_fraud   = prediction == -1

    result = {
        "order_id":   target_order_id,
        "customer_id": target_row.get("customer_id"),
        "total_amount": float(target_row["total_amount"]) if pd.notnull(target_row.get("total_amount")) else None,
        "timestamp":  str(target_row.get("timestamp")),
        "is_fraud":   is_fraud,
        "label":      "FRAUD" if is_fraud else "NOT FRAUD",
        "anomaly_score": score,
        "scored_at":  datetime.now(timezone.utc).isoformat(),
    }

    notify(result, client)
    mark_processed(target_order_id)
    return result


# ---------------------------------------------------------------------------
# Supabase writer
# ---------------------------------------------------------------------------
def write_fraud_result(result: dict) -> None:
    """Upsert the fraud result into the `fraud_results` Supabase table.
    Uses the service-role client so RLS is bypassed for this backend write.
    """
    service_client = get_service_client()
    row = {
        "order_id":      result["order_id"],
        "customer_id":   str(result["customer_id"]) if result["customer_id"] is not None else None,
        "scored_at":     result["scored_at"],
        "is_fraud":      result["is_fraud"],
        "label":         result["label"],
        "anomaly_score": result["anomaly_score"],
    }
    service_client.table("fraud_results").upsert(row, on_conflict="order_id").execute()


# ---------------------------------------------------------------------------
# Notification — see "How to notify the user" section at the bottom
# ---------------------------------------------------------------------------
def notify(result: dict, client: Client = None) -> None:  # client kept for signature compat
    """Prints a banner, appends to a JSONL audit log, and writes to Supabase."""
    banner = "🚨 FRAUD DETECTED" if result["is_fraud"] else "✅ Not fraud"
    print("-" * 60)
    print(f"{banner} — order {result['order_id']}")
    print(f"  customer:   {result['customer_id']}")
    print(f"  amount:     {result['total_amount']}")
    print(f"  score:      {result['anomaly_score']:.4f}  (lower = more anomalous)")
    print("-" * 60)

    with EVENT_LOG.open("a") as f:
        f.write(json.dumps(result) + "\n")

    write_fraud_result(result)


# ---------------------------------------------------------------------------
# Batch classification — one-time run over all existing transactions
# ---------------------------------------------------------------------------
def batch_classify_all():
    """Classify every unprocessed order in the `orders` table in one pass.

    Fits a single IsolationForest on the full dataset, then scores every row
    that hasn't been written to processed_orders.txt yet.
    """
    client = get_client()
    df_raw = fetch_orders(client)
    if df_raw.empty:
        print("No rows in `orders` table.")
        return

    processed_ids = load_processed_ids()
    unprocessed_mask = ~df_raw["order_id"].isin(processed_ids)
    df_unprocessed = df_raw[unprocessed_mask]

    if df_unprocessed.empty:
        print("All orders have already been processed.")
        return

    total = len(df_unprocessed)
    print(f"Batch classifying {total} unprocessed order(s) (out of {len(df_raw)} total)...")

    # Fit on the full population so scaling and one-hot columns are consistent
    df_features = preprocess(df_raw)
    iso = IsolationForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    iso.fit(df_features)

    # Align unprocessed rows to the preprocessed frame by positional index
    unprocessed_positions = df_raw.index.get_indexer(df_unprocessed.index)
    target_features = df_features.iloc[unprocessed_positions]

    predictions = iso.predict(target_features)          #  1 = normal, -1 = anomaly
    scores      = iso.score_samples(target_features)    # lower = more anomalous

    fraud_count = 0
    for i, (_, raw_row) in enumerate(df_unprocessed.iterrows()):
        is_fraud = int(predictions[i]) == -1
        result = {
            "order_id":      raw_row["order_id"],
            "customer_id":   raw_row.get("customer_id"),
            "total_amount":  float(raw_row["total_amount"]) if pd.notnull(raw_row.get("total_amount")) else None,
            "timestamp":     str(raw_row.get("timestamp")),
            "is_fraud":      is_fraud,
            "label":         "FRAUD" if is_fraud else "NOT FRAUD",
            "anomaly_score": float(scores[i]),
            "scored_at":     datetime.now(timezone.utc).isoformat(),
        }
        notify(result, client)
        mark_processed(raw_row["order_id"])
        if is_fraud:
            fraud_count += 1

    print(f"\nBatch complete: {total} orders scored, {fraud_count} flagged as FRAUD.")


# ---------------------------------------------------------------------------
# Supervised scoring — uses fraud_model.py when a trained model exists
# ---------------------------------------------------------------------------
def score_latest_supervised():
    """Score the newest unprocessed order using the supervised classifier.

    Falls back to score_latest_unprocessed() (Isolation Forest) if the model
    file hasn't been created yet.
    """
    if not _SUPERVISED_AVAILABLE or _supervised_predict is None or not _MODEL_PATH.exists():
        print("Supervised model not found — falling back to Isolation Forest.")
        print("Run `python fraud_model.py --train` to enable supervised scoring.")
        return score_latest_unprocessed()

    client = get_client()
    df_raw = fetch_orders(client)
    if df_raw.empty:
        print("No rows in `orders` table.")
        return None

    processed_ids = load_processed_ids()
    sort_col = "created_at" if "created_at" in df_raw.columns else "timestamp"
    df_raw = df_raw.sort_values(by=sort_col, ascending=False)

    target_row = next(
        (row for _, row in df_raw.iterrows() if row["order_id"] not in processed_ids),
        None,
    )
    if target_row is None:
        print("All orders have already been processed.")
        return None

    target_order_id = target_row["order_id"]
    print(f"[supervised] Scoring order_id={target_order_id}")

    prediction = _supervised_predict(target_row.to_dict())

    result = {
        "order_id":        target_order_id,
        "customer_id":     target_row.get("customer_id"),
        "total_amount":    float(target_row["total_amount"]) if pd.notnull(target_row.get("total_amount")) else None,
        "timestamp":       str(target_row.get("timestamp")),
        "is_fraud":        prediction["is_fraud"],
        "label":           prediction["label"],
        "anomaly_score":   prediction["fraud_probability"],  # probability used as score
        "scored_at":       datetime.now(timezone.utc).isoformat(),
    }

    notify(result, client)
    mark_processed(target_order_id)
    return result


def batch_supervised():
    """Score every unprocessed order in one vectorised pass using the
    supervised classifier.  Falls back to batch_classify_all() if the model
    file doesn't exist.
    """
    if not _SUPERVISED_AVAILABLE or _supervised_predict is None or not _MODEL_PATH.exists():
        print("Supervised model not found — falling back to Isolation Forest batch.")
        print("Run `python fraud_model.py --train` to enable supervised scoring.")
        return batch_classify_all()

    from fraud_model import predict_batch as _predict_batch

    client = get_client()
    df_raw = fetch_orders(client)
    if df_raw.empty:
        print("No rows in `orders` table.")
        return

    processed_ids = load_processed_ids()
    df_unprocessed = df_raw[~df_raw["order_id"].isin(processed_ids)]

    if df_unprocessed.empty:
        print("All orders have already been processed.")
        return

    total = len(df_unprocessed)
    print(f"[supervised] Batch scoring {total} unprocessed order(s)…")

    scored = _predict_batch(df_unprocessed)

    fraud_count = 0
    for _, row in scored.iterrows():
        result = {
            "order_id":      row["order_id"],
            "customer_id":   row.get("customer_id"),
            "total_amount":  float(row["total_amount"]) if pd.notnull(row.get("total_amount")) else None,
            "timestamp":     str(row.get("timestamp")),
            "is_fraud":      bool(row["is_fraud_pred"]),
            "label":         row["label_pred"],
            "anomaly_score": float(row["fraud_probability"]),
            "scored_at":     datetime.now(timezone.utc).isoformat(),
        }
        notify(result, client)
        mark_processed(row["order_id"])
        if result["is_fraud"]:
            fraud_count += 1

    print(f"\nBatch complete: {total} orders scored, {fraud_count} flagged as FRAUD.")


if __name__ == "__main__":
    import sys
    if "--single" in sys.argv:
        score_latest_unprocessed()
    elif "--supervised" in sys.argv:
        score_latest_supervised()
    elif "--supervised-batch" in sys.argv:
        batch_supervised()
    else:
        batch_classify_all()
