import pandas as pd
import random
from datetime import datetime, timedelta

# Generate 100 example orders
orders = []
products = ["Laptop Model X", "Monitor 24\"", "Keyboard", "Mouse", "Printer", "Headphones", "Tablet", "Webcam"]
statuses = ["Processing", "Shipped", "Delivered", "Cancelled"]

for i in range(100):
    order_number = 1000 + i
    phone_number = f"+44{7911000000 + i}"  # UK phone numbers
    status = random.choice(statuses)
    delivery_date = (datetime.today() + timedelta(days=random.randint(-10, 10))).strftime("%Y-%m-%d")
    product = random.choice(products)
    quantity = random.randint(1, 5)
    notes = random.choice([
        "Left at front door",
        "Customer requested gift wrap",
        "Signed by neighbor",
        "Call before delivery",
        "-",
        "Customer requested refund"
    ])
    
    orders.append({
        "OrderNumber": order_number,
        "CustomerPhone": phone_number,
        "Status": status,
        "DeliveryDate": delivery_date,
        "Product": product,
        "Quantity": quantity,
        "Notes": notes
    })

# Create DataFrame and save as Excel
df = pd.DataFrame(orders)
df.to_excel("demo_sap.xlsx", index=False)
print("demo_sap.xlsx created with 100 sample orders!")
