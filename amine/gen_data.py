import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
food_types = ["vegetables", "dairy", "canned", "bakery", "meat", "dry goods"]
locations = ["Tunis Center", "Region C"]  # Urban, rural/disaster-affected
start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 12, 31)
num_days = (end_date - start_date).days + 1
holidays = [datetime(2024, 12, 25), datetime(2025, 1, 1), datetime(2025, 12, 25)]
disaster_start = datetime(2025, 7, 1)  # Scenario 3: Disaster in Region C
disaster_end = disaster_start + timedelta(days=200)

# Quantity parameters (mean, std in kg, zero probability by location)
quantity_params = {
    "vegetables": {
        "Tunis Center": {"mean": 60, "std": 25, "zero_prob": 0.25, "weekend_zero_prob": 0.15},
        "Region C": {"mean": 40, "std": 15, "zero_prob": 0.30, "weekend_zero_prob": 0.20}
    },
    "dairy": {
        "Tunis Center": {"mean": 35, "std": 15, "zero_prob": 0.30, "weekend_zero_prob": 0.20},
        "Region C": {"mean": 25, "std": 10, "zero_prob": 0.35, "weekend_zero_prob": 0.25}
    },
    "canned": {
        "Tunis Center": {"mean": 25, "std": 10, "zero_prob": 0.35, "weekend_zero_prob": 0.30},
        "Region C": {"mean": 15, "std": 8, "zero_prob": 0.40, "weekend_zero_prob": 0.35}
    },
    "bakery": {
        "Tunis Center": {"mean": 50, "std": 30, "zero_prob": 0.25, "weekend_zero_prob": 0.15},
        "Region C": {"mean": 30, "std": 20, "zero_prob": 0.30, "weekend_zero_prob": 0.20}
    },
    "meat": {
        "Tunis Center": {"mean": 18, "std": 10, "zero_prob": 0.40, "weekend_zero_prob": 0.25},
        "Region C": {"mean": 12, "std": 6, "zero_prob": 0.45, "weekend_zero_prob": 0.30}
    },
    "dry goods": {
        "Tunis Center": {"mean": 30, "std": 12, "zero_prob": 0.30, "weekend_zero_prob": 0.25},
        "Region C": {"mean": 20, "std": 10, "zero_prob": 0.35, "weekend_zero_prob": 0.30}
    }
}

# Days until expiry and priority parameters
expiry_priority = {
    "vegetables": {"days_until_expiry": 7.0, "priority": 1.0, "disaster_priority": 1.5},
    "dairy": {"days_until_expiry": 5.0, "priority": 1.2, "disaster_priority": 1.7},
    "canned": {"days_until_expiry": 365.0, "priority": 0.5, "disaster_priority": 1.0},
    "bakery": {"days_until_expiry": 2.0, "priority": 1.3, "disaster_priority": 1.8},
    "meat": {"days_until_expiry": 3.0, "priority": 1.5, "disaster_priority": 2.0},
    "dry goods": {"days_until_expiry": 365.0, "priority": 0.7, "disaster_priority": 1.2}
}

# Generate data
data = []
for day in range(num_days):
    donation_date = start_date + timedelta(days=day)
    is_holiday = donation_date in holidays
    is_weekend = donation_date.weekday() >= 5
    is_disaster = (disaster_start <= donation_date <= disaster_end)
    
    for location in locations:
        for food_type in food_types:
            # Handle disaster scenario: zero quantities in Region C during disaster
            if is_disaster and location == "Region C":
                quantity = 0.0
            else:
                # Generate quantity
                params = quantity_params[food_type][location]
                zero_prob = params["weekend_zero_prob"] if is_weekend else params["zero_prob"]
                quantity = np.random.normal(params["mean"], params["std"])
                
                # Apply zero probability
                if np.random.random() < zero_prob:
                    quantity = 0.0
                else:
                    # Adjust for seasonality and holidays
                    if is_weekend and food_type in ["bakery", "vegetables", "meat"]:
                        quantity *= 2.0  # 2x for Scenario 2
                    if is_holiday:
                        quantity *= 1.5  # 50% increase
                    # Add monthly seasonality
                    month_factor = 1 + 0.2 * np.sin(2 * np.pi * donation_date.month / 12)
                    quantity *= month_factor
                    quantity = max(0, round(quantity, 2))
            
            # Generate expiry date
            days_until_expiry = expiry_priority[food_type]["days_until_expiry"]
            days_until_expiry += np.random.normal(0, 0.5)
            days_until_expiry = max(1, round(days_until_expiry))
            expiry_date = donation_date + timedelta(days=days_until_expiry)
            
            # Generate priority
            priority_key = "disaster_priority" if is_disaster and location == "Region C" else "priority"
            priority = expiry_priority[food_type][priority_key]
            priority += np.random.normal(0, 0.1)
            priority = max(0, round(priority, 2))
            
            data.append({
                "donation_date": donation_date.strftime("%Y-%m-%d"),
                "expiry_date": expiry_date.strftime("%Y-%m-%d"),
                "type": food_type,
                "quantity": quantity,
                "priority": priority,
                "location": location
            })

# Create DataFrame
df = pd.DataFrame(data)

# Adjust to ensure ~30% zero quantities overall (excluding disaster period)
non_disaster_mask = ~((df["donation_date"] >= disaster_start.strftime("%Y-%m-%d")) & 
                      (df["donation_date"] <= disaster_end.strftime("%Y-%m-%d")) & 
                      (df["location"] == "Region C"))
zero_prop = (df.loc[non_disaster_mask, "quantity"] == 0).mean()
if zero_prop < 0.28 or zero_prop > 0.32:
    zero_mask = (df["quantity"] == 0) & non_disaster_mask
    non_zero_mask = (df["quantity"] > 0) & non_disaster_mask
    if zero_prop < 0.28:
        reduce_count = int((0.30 - zero_prop) * non_disaster_mask.sum() / (1 - 0.30))
        non_zero_indices = df[non_zero_mask].sample(n=reduce_count, random_state=42).index
        df.loc[non_zero_indices, "quantity"] = 0
    elif zero_prop > 0.32:
        increase_count = int((zero_prop - 0.30) * non_disaster_mask.sum() / 0.30)
        zero_indices = df[zero_mask].sample(n=increase_count, random_state=42).index
        for idx in zero_indices:
            food_type = df.loc[idx, "type"]
            location = df.loc[idx, "location"]
            df.loc[idx, "quantity"] = np.random.normal(
                quantity_params[food_type][location]["mean"],
                quantity_params[food_type][location]["std"]
            )
    df["quantity"] = df["quantity"].clip(lower=0).round(2)

# Save to CSV
df.to_csv("amine/data/raw_data.csv", index=False)
print(f"Generated raw_data.csv with {len(df)} rows, {len(df['donation_date'].unique())} days, "
      f"{(df['quantity'] == 0).mean():.2%} zero quantities overall, "
      f"{(df.loc[non_disaster_mask, 'quantity'] == 0).mean():.2%} zero quantities outside disaster")