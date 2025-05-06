import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# Parameters
steps = 1000
recipients = 8  # 4 in Region_A (urban), 4 in Region_B (rural)
food_types = ["vegetables", "dairy", "canned", "bakery", "meat", "dry goods"]
base_demand = {
    "Region_A": 45.0,  # Urban demand
    "Region_B": 21.71  # Rural demand
}
food_type_factors = {
    "vegetables": 1.2, "dairy": 1.0, "canned": 0.8,
    "bakery": 1.3, "meat": 1.4, "dry goods": 0.9
}
weekend_spike = 1.5
holiday_spike = 1.6  # Reduced for Region_A
shortage_prob = 0.2
shortage_factor = 1.8  # Reduced for smoother spikes
holiday_steps = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]  # More frequent holidays
promotion_steps = [150, 350, 550, 750]  # Simulated promotions

# Generate data
data = []
base_date = datetime(2023, 1, 1)
for step in range(steps):
    current_date = base_date + timedelta(days=step)
    day_of_week = current_date.weekday()
    month = current_date.month
    is_weekend = 1 if step in [250, 500, 750] else 0
    is_holiday = 1 if step in holiday_steps else 0
    is_promotion = 1 if step in promotion_steps else 0
    shortage = int(random.random() < shortage_prob)
    for recip_id in range(recipients):
        region = "Region_A" if recip_id < 4 else "Region_B"
        base = base_demand[region]
        demand_factor = shortage_factor if (region == "Region_A" and shortage) else 1
        if is_weekend:
            base *= weekend_spike
        if is_holiday and region == "Region_A":
            base *= holiday_spike
        if is_promotion and region == "Region_A":
            base *= 1.3  # Promotion effect
        disaster_flag = shortage if region == "Region_B" else 0
        holiday_flag = is_holiday if region == "Region_A" else 0
        promotion_flag = is_promotion if region == "Region_A" else 0
        for food_type in food_types:
            food_factor = food_type_factors.get(food_type, 1.0)
            demand = base * demand_factor * food_factor * (0.9 + 0.2 * random.random())
            data.append([step, recip_id, region, food_type, demand, is_weekend, disaster_flag, holiday_flag, promotion_flag, day_of_week, month])

# Create DataFrame
columns = ["timestamp", "recipient_id", "region", "food_type", "demand_kg", "is_weekend", "disaster_flag", "holiday_flag", "promotion_flag", "day_of_week", "month"]
df = pd.DataFrame(data, columns=columns)
df["demand_kg"] = df["demand_kg"].apply(lambda x: max(1.0, x))

# Save to CSV
os.makedirs("data/demand_prediction", exist_ok=True)
df.to_csv("data/demand_prediction/demand_data.csv", index=False)

print("Generated raw data with holidays, promotions, and refined shortages for food type categories.")
print(f"Rows: {len(df)}")
print("Sample data:")
print(df.head())