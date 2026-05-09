import os
import json
import random
import time
from datetime import datetime
from azure.eventhub import EventHubProducerClient, EventData
import pandas as pd
import numpy as np
rng = np.random.default_rng()

from dotenv import load_dotenv
load_dotenv()   

EVENTHUB_CONNECTION_STRING = os.getenv("EVENTHUB_CONNECTION_STRING_FRAUD")
EVENTHUB_NAME = os.getenv("EVENTHUB_NAME")

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
df_restaurants = pd.read_csv(os.path.join(script_dir, "data", "restaurants.csv"))
df_customers = pd.read_csv(os.path.join(script_dir, "data", "customers2.csv"))
df_menu_items = pd.read_csv(os.path.join(script_dir, "data", "menu_items.csv"))

RESTAURANTS = df_restaurants['restaurant_id'].tolist()
CUSTOMERS = df_customers['customer_id'].tolist()

# Silence FutureWarning by avoiding grouping column in apply
MENU_BY_RESTAURANT = (
    df_menu_items
    .groupby('restaurant_id', group_keys=False)
    .apply(lambda x: x.to_dict('records'))
    .to_dict()
)

ORDER_TYPES = [ "delivery"]
PAYMENT_METHODS = ["card"]
ORDER_STATUSES = ["delivered", "completed"]
TITLE_NAME = [""]
SUFFIX_NAME = [""]
CARD_ACCEPTED = [0, 1]
CARD_ACCEPTED_PROBABILITIES = [0.98, 0.02]


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
        "card_match": rng.choice(CARD_ACCEPTED, p=CARD_ACCEPTED_PROBABILITIES)
    }

    return order


def stream_to_eventhub(interval_seconds=3, max_orders=None):
    producer = EventHubProducerClient.from_connection_string(
        conn_str=EVENTHUB_CONNECTION_STRING,
        eventhub_name=EVENTHUB_NAME
    )

    print(f"\n\nStreaming to Event Hub: {EVENTHUB_NAME}")
    order_count = 3

    try:
        while True:
            order = generate_order()

            # JSON-safe serialization for Event Hub
            order_json = json_safe(order)

            event_data_batch = producer.create_batch()
            event_data_batch.add(EventData(order_json))
            producer.send_batch(event_data_batch)

            order_count += 1
            print(f"\n[{order_count}] {order['order_id']} | {order['restaurant_id']} | AED {order['total_amount']}")
            # JSON-safe pretty print
            print(json_safe(order, indent=4))

            if max_orders and order_count >= max_orders:
                break

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        producer.close()


if __name__ == "__main__":
    stream_to_eventhub(interval_seconds=3)
