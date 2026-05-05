import os
import json
import random
import time
from datetime import datetime
from supabase import create_client, Client
import pandas as pd
import numpy as np
rng = np.random.default_rng()

from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=_env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "orders")

# ============================================
# JSON helper
# ============================================
def json_safe(obj, **kwargs):
    """
    JSON-dump that can handle NumPy types (int64, float64, bool_, etc.).
    Extra kwargs (like indent=4) are passed through.
    """
    return json.dumps(
        obj,
        default=lambda x: x.item() if hasattr(x, "item") else x,
        **kwargs
    )

# ============================================
# LOAD MASTER DATA
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r"C:\Users\lmans\Programs\realtimeFrauddetection\databricks-masterclass-main-modified-for-fraud-detection\databricks-masterclass-main\projects\databricks-e2e-project\00_synthetic_data\data"
df_restaurants = pd.read_csv(os.path.join(DATA_DIR, "restaurants.csv"))
df_customers = pd.read_csv(os.path.join(DATA_DIR, "customers2.csv"))
df_menu_items = pd.read_csv(os.path.join(DATA_DIR, "menu_items.csv"))

RESTAURANTS = df_restaurants['restaurant_id'].tolist()
CUSTOMERS = df_customers['customer_id'].tolist()

# Silence FutureWarning by avoiding grouping column in apply
MENU_BY_RESTAURANT = (
    df_menu_items
    .groupby('restaurant_id', group_keys=False)
    .apply(lambda x: x.to_dict('records'))
    .to_dict()
)

ORDER_TYPES = ["dine_in", "takeaway", "delivery"]
PAYMENT_METHODS = ["cash", "card", "wallet"]
ORDER_STATUSES = ["delivered", "completed"]
TITLE_NAME = ["Mr", "Mrs", "Miss", "Ms", "Dr"]
SUFFIX_NAME = ["Sr", "Jr", "III", "Esq", "MD", "PHD", "DDS"]
CARD_ACCEPTED = [0, 1]
CARD_ACCEPTED_PROBABILITIES = [0.001, 0.999]


def generate_order():
    order_date = datetime.utcnow()
    restaurant_id = random.choice(RESTAURANTS)
    customer_id = random.choice(CUSTOMERS)

    # Pull customer row ONCE
    cust = df_customers.loc[df_customers['customer_id'] == customer_id]

    if cust.empty:
        raise ValueError(f"Customer ID {customer_id} not found in df_customers")

    cust = cust.iloc[0]

    # Convert NumPy types → Python native types for customer row
    cust = cust.apply(lambda x: x.item() if hasattr(x, "item") else x)

    # Build items
    menu_items = MENU_BY_RESTAURANT[restaurant_id]
    num_items = random.randint(1, min(5, len(menu_items)))
    selected_items = random.sample(menu_items, num_items)

    items = []
    total_amount = 0.0

    for item in selected_items:
        quantity = random.randint(1, 3)
        subtotal = float(item["price"]) * quantity  # ensure Python float
        total_amount += subtotal

        items.append({
            "item_id": int(item["item_id"]) if hasattr(item["item_id"], "item") else item["item_id"],
            "name": item["name"],
            "category": item["category"],
            "quantity": int(quantity),
            "unit_price": float(item["price"]),
            "subtotal": round(float(subtotal), 2)
        })

    order_id = f"ORD-{order_date.strftime('%Y%m%d')}-{random.randint(100000, 999999)}"

    order = {
        "order_id": order_id,
        "timestamp": order_date.isoformat() + "Z",
        "restaurant_id": restaurant_id,
        "customer_id": customer_id,
        "order_type": random.choice(ORDER_TYPES),
        "items": items,
        "total_amount": round(float(total_amount), 2),
        "payment_method": random.choice(PAYMENT_METHODS),
        "order_status": random.choice(ORDER_STATUSES),
        "created_at": order_date.isoformat() + "Z",

        # Customer details
        "title": random.choice(TITLE_NAME),
        "first_name": cust.first_name,
        "last_name": cust.last_name,
        "suffix": random.choice(SUFFIX_NAME),
        "street": cust.street,
        "city": cust.city,
        "country": cust.country,
        "state": cust.state,
        "zip_code": cust.zip_code,
        "phone_number": cust.phone,
        "mobile": cust.mobile,
        "email_address": cust.email_address,
        "card_number": cust.card_number,
        "credit_card_exp_year": cust.credit_card_exp_year,
        "credit_card_exp_month": cust.credit_card_exp_month,
        "cvv": cust.cvv,
        "card_match": rng.choice(CARD_ACCEPTED, p=CARD_ACCEPTED_PROBABILITIES),
        "is_fraud": 0
    }

    return order


def stream_to_supabase(interval_seconds=3, max_orders=None, batch_size=10):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print(f"\n\nStreaming to Supabase table: {SUPABASE_TABLE} (batch_size={batch_size})", flush=True)
    order_count = 0

    try:
        while True:
            batch = []
            for _ in range(batch_size):
                order = generate_order()
                batch.append(json.loads(json_safe(order)))
                order_count += 1
                print(f"\n[{order_count}] {order['order_id']} | {order['restaurant_id']} | AED {order['total_amount']}", flush=True)
                print(json_safe(order, indent=4), flush=True)

                if max_orders and order_count >= max_orders:
                    break

            # Batch insert
            supabase.table(SUPABASE_TABLE).insert(batch).execute()
            print(f"\n-- Inserted batch of {len(batch)} orders --", flush=True)

            if max_orders and order_count >= max_orders:
                break

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nStopped", flush=True)


if __name__ == "__main__":
    stream_to_supabase(interval_seconds=3)
