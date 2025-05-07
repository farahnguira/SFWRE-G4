import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Configuration
num_items = 1000  # Reduced from 10000 to 1000 for more manageable simulation
base_date = datetime.today()
categories = ["vegetables", "dairy", "canned", "bakery", "meat", "dry goods"]
category_codes = {cat: idx + 1 for idx, cat in enumerate(categories)}

# Define realistic ranges per food type
food_types = {
    "vegetables": {"temp": (2, 10), "humid": (80, 95), "shelf_base": (7, 60)},
    "dairy": {"temp": (0, 4), "humid": (70, 90), "shelf_base": (7, 30)},
    "meat": {"temp": (0, 4), "humid": (75, 95), "shelf_base": (3, 20)},
    "bakery": {"temp": (15, 20), "humid": (40, 60), "shelf_base": (3, 10)},
    "canned": {"temp": (15, 20), "humid": (40, 60), "shelf_base": (180, 365)},
    "dry goods": {"temp": (15, 25), "humid": (30, 50), "shelf_base": (100, 365)}
}

# Generate data
data = []
for _ in range(num_items):
    # More balanced weights: vegetables, dairy, canned, bakery, meat, dry goods
    food_type = random.choices(categories, weights=[20, 15, 25, 15, 10, 15])[0] 
    params = food_types[food_type]
    
    # Temperature and humidity
    temp = np.random.uniform(params["temp"][0], params["temp"][1])
    humid = np.random.uniform(params["humid"][0], params["humid"][1])
    
    # Shelf life: inverse relationship with temperature for perishables
    if food_type in ["meat", "dairy", "vegetables", "bakery"]:
        temp_range = params["temp"][1] - params["temp"][0]
        shelf_range = params["shelf_base"][1] - params["shelf_base"][0]
        temp_factor = (temp - params["temp"][0]) / temp_range if temp_range > 0 else 0
        shelf_life = params["shelf_base"][1] - temp_factor * shelf_range
        shelf_life += np.random.uniform(-2, 2)  # Small noise
    else:
        shelf_life = np.random.uniform(params["shelf_base"][0], params["shelf_base"][1])
    
    shelf_life = max(1, int(shelf_life))  # Ensure positive and integer
    expiry_date = base_date + timedelta(days=shelf_life)
    
    data.append([
        food_type, category_codes[food_type], expiry_date, shelf_life, 
        np.random.randint(1, 20), max(1, 10 - shelf_life // 30), temp, humid
    ])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "type", "food_type_code", "expiry_date", "shelf_life_days", 
    "quantity", "priority", "temperature", "humidity"
])
df["item_id"] = range(1, len(df) + 1)

# Save to CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "processed_data.csv")
df.to_csv(output_path, index=False)
print(f" Generated {num_items} realistic records to: {output_path}")
print("Now run clean_data.py to clean the data and create cleaned_data.csv")