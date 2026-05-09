import pandas as pd
import random
from datetime import datetime, timedelta
import json
import os
import numpy as np
rng = np.random.default_rng()
# ============================================
# LOAD MASTER DATA
# ============================================
script_dir = os.path.dirname(os.path.abspath(__file__))
df_restaurants = pd.read_csv(os.path.join(script_dir, "data", "restaurants.csv"))
df_customers = pd.read_csv(os.path.join(script_dir, "data", "customers2.csv"))
df_menu_items = pd.read_csv(os.path.join(script_dir, "data", "menu_items.csv"))

RESTAURANTS = df_restaurants['restaurant_id'].tolist()
CUSTOMERS = df_customers['customer_id'].tolist()
# from df_customers, return customer_id, first_name, last_name, title, suffix, street, city, country, state, zip_code, phone_number, mobile, email_address, card_number, credit_card_exp_year, credit_card_exp_month, cvv, card_match
customer_table = df_customers['customer_id', 'first_name', 'last_name', 'title', 'suffix', 'street', 'city', 'country', 'state', 'zip_code', 'phone_number', 'mobile', 'email_address', 'card_number', 'credit_card_exp_year', 'credit_card_exp_month', 'cvv', 'card_match']

MENU_BY_RESTAURANT = df_menu_items.groupby('restaurant_id').apply(
    lambda x: x.to_dict('records')
).to_dict()

ORDER_TYPES = ["dine_in", "takeaway", "delivery"]
PAYMENT_METHODS = ["cash", "card", "wallet"]
ORDER_STATUSES = ["delivered", "completed"]  # Only completed orders for historical data
TITLE_NAME = ["Mr", "Mrs","Miss","Ms","Dr"]
SUFFIX_NAME = ["Sr","Jr","III","Esq","MD","PHD","DDS"]
CARD_ACCEPTED = [0,1]
CARD_ACCEPTED_PROBABILITIES = [0.001, 0.999]
# ============================================
# GENERATE HISTORICAL ORDER
# ============================================
def generate_historical_order(order_date):
    """Generate single historical order"""
    restaurant_id = random.choice(RESTAURANTS)
    customer_id = random.choice(CUSTOMERS)
    
    menu_items = MENU_BY_RESTAURANT[restaurant_id]
    num_items = random.randint(1, min(5, len(menu_items)))
    selected_items = random.sample(menu_items, num_items)
    
    items = []
    total_amount = 0.0
    
    for item in selected_items:
        quantity = random.randint(1, 3)
        subtotal = item["price"] * quantity
        total_amount += subtotal
        
        items.append({
            "item_id": item["item_id"],
            "name": item["name"],
            "category": item["category"],
            "quantity": quantity,
            "unit_price": item["price"],
            "subtotal": round(subtotal, 2)
        })
    
    order_id = f"ORD-{order_date.strftime('%Y%m%d')}-{random.randint(100000, 999999)}"
    
    return {
        "order_id": order_id,
        "timestamp": order_date.isoformat(),
        "restaurant_id": restaurant_id,
        "customer_id": customer_id,
        "order_type": random.choice(ORDER_TYPES),
        "items": json.dumps(items),  # Store as JSON string for CSV
        "total_amount": round(total_amount, 2),
        "payment_method": random.choice(PAYMENT_METHODS),
        "order_status": random.choice(ORDER_STATUSES),
        "created_at": order_date.isoformat()
        #### ADDED
        ,"title": random.choice(TITLE_NAME),
        "first_name": df_customers.loc[df_customers['customer_id'] == customer_id, 'first_name'],
        "last_name": df_customers.loc[df_customers['customer_id'] == customer_id, 'last_name'],
        "suffix": random.choice(SUFFIX_NAME),
        "street": df_customers.loc[df_customers['customer_id'] == customer_id, 'street'],
        "city": df_customers.loc[df_customers['customer_id'] == customer_id, 'city'],
        "country": df_customers.loc[df_customers['customer_id'] == customer_id, 'country'],
        "state": df_customers.loc[df_customers['customer_id'] == customer_id, 'state'],
        "zip_code": df_customers.loc[df_customers['customer_id'] == customer_id, 'zip_code'],
        "phone_number": df_customers.loc[df_customers['customer_id'] == customer_id, 'phone'],
        "mobile": df_customers.loc[df_customers['customer_id'] == customer_id, 'mobile'],
        "email_address": df_customers.loc[df_customers['customer_id'] == customer_id, 'email_address'],
        "card_number": df_customers.loc[df_customers['customer_id'] == customer_id, 'card_number'],
        "credit_card_exp_year": df_customers.loc[df_customers['customer_id'] == customer_id, 'credit_card_exp_year'],
        "credit_card_exp_month": df_customers.loc[df_customers['customer_id'] == customer_id, 'credit_card_exp_month'],
        "cvv": df_customers.loc[df_customers['customer_id'] == customer_id, 'cvv'],
        "card_match": rng.choice(CARD_ACCEPTED, p=CARD_ACCEPTED_PROBABILITIES)
    }

# ============================================
# GENERATE BATCH HISTORICAL ORDERS
# ============================================
def generate_historical_orders(num_orders=8000, months_back=6):
    """Generate historical orders over past X months"""
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months_back * 30)
    
    orders = []
    
    print(f"Generating {num_orders} orders from {start_date.date()} to {end_date.date()}")
    
    for i in range(num_orders):
        # Random date within range
        days_offset = random.randint(0, (end_date - start_date).days)
        order_date = start_date + timedelta(days=days_offset)
        
        # Add random hours/minutes
        order_date = order_date.replace(
            hour=random.randint(10, 22),
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )
        
        order = generate_historical_order(order_date)
        orders.append(order)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1} orders...")
    
    df_orders = pd.DataFrame(orders)
    
    # Sort by timestamp
    df_orders = df_orders.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    df_orders.to_csv(os.path.join(script_dir, "data", "historical_orders.csv"), index=False)
    
    print(f"\nGenerated {len(df_orders)} historical orders")
    print(f"Saved to: historical_orders.csv")
    print(f"Date range: {df_orders['timestamp'].min()} to {df_orders['timestamp'].max()}")
    print(f"Total revenue: AED {df_orders['total_amount'].sum():,.2f}")

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    generate_historical_orders(num_orders=8000, months_back=6)