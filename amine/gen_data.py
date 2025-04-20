import csv
import random
from datetime import datetime, timedelta

# Configuration
num_items = 545
unique_date_count = 90
# make sure this range ≥ unique_date_count
max_days_range = 365  

base_date = datetime.today()
categories = ["vegetables", "dairy", "canned", "bakery", "meat", "dry"]
max_priority = 10

# 1) Pick 90 different day‑offsets
unique_offsets = random.sample(range(1, max_days_range + 1), unique_date_count)
# 2) Turn them into formatted dates
unique_dates = [
    (base_date + timedelta(days=d)).strftime("%Y-%m-%d")
    for d in unique_offsets
]

# 3) Build your full list of 545 expiry dates:
#    — start with the 90 unique ones (so each appears at least once)
#    — then sample the remaining 455 from those 90 (with replacement)
expiry_dates = unique_dates.copy()
expiry_dates += random.choices(unique_dates, k=num_items - unique_date_count)

# 4) Shuffle so the “singles” and “duplicates” are randomly interspersed
random.shuffle(expiry_dates)

# 5) Now generate the CSV, pulling expiry_dates[i] for each row
with open("raw_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["item_id", "type", "expiry_date", "quantity", "priority"])
    
    for item_id in range(1, num_items + 1):
        food_type = random.choices(categories, weights=[30,20,15,10,5,10])[0]
        expiry = expiry_dates[item_id - 1]
        quantity = random.randint(1, 20)
        # priority inversely proportional to days until expiry:
        days_to_expiry = (datetime.strptime(expiry, "%Y-%m-%d") - base_date).days
        priority = max(1, max_priority - days_to_expiry)
        
        writer.writerow([item_id, food_type, expiry, quantity, priority])
