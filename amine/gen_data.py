import csv
import random
from datetime import datetime, timedelta

# Configuration
num_items = 545
base_date = datetime.today()
categories = ["vegetables", "dairy", "canned", "bakery", "meat", "dry"]
max_priority = 10

with open("raw_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["item_id", "type", "expiry_date", "quantity", "priority"])
    
    for item_id in range(1, num_items + 1):
        food_type = random.choices(categories, weights=[30,20,15,10,5,10])[0]
        days_to_expiry = random.randint(1, 30)
        expiry = (base_date + timedelta(days=days_to_expiry)).strftime("%Y-%m-%d")
        quantity = random.randint(1, 20)
        # Higher urgency if closer to expiry
        priority = max(1, max_priority - days_to_expiry)
        
        writer.writerow([item_id, food_type, expiry, quantity, priority])
