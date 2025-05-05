import pandas as pd
import numpy as np
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
steps = 1000
recipients = 8  # 4 in Region_A (urban), 4 in Region_B (rural)
food_types = ["vegetables", "dairy", "canned", "bakery", "meat", "dry goods"]
base_demand = {
    "Region_A": 35.31,  # Urban, higher demand (Scenario 1)
    "Region_B": 21.71   # Rural, lower demand
}
# Add food type specific demand factors
food_type_factors = {
    "vegetables": 1.1, "dairy": 1.0, "canned": 0.8,
    "bakery": 1.2, "meat": 1.3, "dry goods": 0.9
}
weekend_spike = 1.5  # 50% increase on weekends
shortage_prob = 0.3  # 30% chance of shortage in Region_B (Changed from 0.5)
shortage_factor = 2  # Double demand in Region_A during Region_B shortages

# Generate data
data = []
for step in range(steps):
    is_weekend = 1 if step in [250, 500, 750] else 0  # Weekends at steps 250, 500, 750
    # Determine shortage once per timestamp for Region_B
    shortage = int(random.random() < shortage_prob)
    for recip_id in range(recipients):
        region = "Region_A" if recip_id < 4 else "Region_B"
        base = base_demand[region]
        # Apply shortage effect: double demand in Region_A if shortage occurs
        demand_factor = shortage_factor if (region == "Region_A" and shortage) else 1
        # Apply weekend spike
        if is_weekend:
            base *= weekend_spike
        # Set disaster_flag: 1 for Region_B during shortage, 0 otherwise
        disaster_flag = shortage if region == "Region_B" else 0
        for food_type in food_types:
            # Apply food type factor
            food_factor = food_type_factors.get(food_type, 1.0)
            demand = base * demand_factor * food_factor * (0.8 + 0.4 * random.random())  # Random variation ±20%
            data.append([step, recip_id, region, food_type, demand, is_weekend, disaster_flag])

# Create DataFrame
columns = ["timestamp", "recipient_id", "region", "food_type", "demand_kg", "is_weekend", "disaster_flag"]
df = pd.DataFrame(data, columns=columns)

# Ensure no negative or zero demand
df["demand_kg"] = df["demand_kg"].apply(lambda x: max(1.0, x))

# Save to CSV
os.makedirs("data/demand_prediction", exist_ok=True)
df.to_csv("data/demand_prediction/demand_data.csv", index=False)

print("Generated raw data with 6 food types, food-specific factors, and random shortages.")
print(f"Rows: {len(df)}")
print("Sample data:")
print(df.head())