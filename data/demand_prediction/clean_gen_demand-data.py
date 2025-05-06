import pandas as pd
import os

# Load raw data
input_file = "data/demand_prediction/demand_data.csv"
output_file = "data/demand_prediction/cleaned_demand_data.csv"

os.makedirs("data/demand_prediction", exist_ok=True)
df = pd.read_csv(input_file)

# Check for missing values
print("Missing values before cleaning:")
print(df.isnull().sum())
df = df.dropna()

# Validate and clean data
df = df[df["demand_kg"] > 0]
df = df.drop_duplicates()
df["timestamp"] = df["timestamp"].astype(int)
df["recipient_id"] = df["recipient_id"].astype(int)
df["demand_kg"] = df["demand_kg"].astype(float)
df["is_weekend"] = df["is_weekend"].astype(int)
df["disaster_flag"] = df["disaster_flag"].astype(int)
df["holiday_flag"] = df["holiday_flag"].astype(int)
df["promotion_flag"] = df["promotion_flag"].astype(int)
df["day_of_week"] = df["day_of_week"].astype(int)
df["month"] = df["month"].astype(int)

# Outlier detection and capping (Region_A only)
region_a = df[df["region"] == "Region_A"]
q95 = region_a["demand_kg"].quantile(0.95)
df.loc[(df["region"] == "Region_A") & (df["demand_kg"] > q95), "demand_kg"] = q95
print(f"Capped Region_A demand at 95th percentile: {q95:.2f}")

# Optional: Normalize demand for Region_A to reduce variability (uncomment to enable)
# max_demand = region_a["demand_kg"].max()
# df.loc[df["region"] == "Region_A", "demand_kg"] = df["demand_kg"] / max_demand * 100

# Validate scenario patterns
weekend_steps = df[df["is_weekend"] == 1]["timestamp"].unique()
expected_weekends = [250, 500, 750]
if not all(ws in weekend_steps for ws in expected_weekends):
    print(f"Warning: Expected weekend steps {expected_weekends} not fully present. Found {weekend_steps}")

holiday_steps = df[df["holiday_flag"] == 1]["timestamp"].unique()
expected_holidays = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900]
if not all(hs in holiday_steps for hs in expected_holidays):
    print(f"Warning: Expected holiday steps {expected_holidays} not fully present. Found {holiday_steps}")

promotion_steps = df[df["promotion_flag"] == 1]["timestamp"].unique()
expected_promotions = [150, 350, 550, 750]
if not all(ps in promotion_steps for ps in expected_promotions):
    print(f"Warning: Expected promotion steps {expected_promotions} not fully present. Found {promotion_steps}")

shortage_count = df[df["disaster_flag"] == 1]["timestamp"].nunique()
expected_shortage_range = int(1000 * 0.2)
if not (expected_shortage_range * 0.8 <= shortage_count <= expected_shortage_range * 1.2):
    print(f"Warning: Expected ~{expected_shortage_range} shortages, found {shortage_count}")

# Validate day_of_week and month
if df["day_of_week"].max() > 6 or df["day_of_week"].min() < 0:
    print(f"Warning: Invalid day_of_week values found: {df['day_of_week'].unique()}")
if df["month"].max() > 12 or df["month"].min() < 1:
    print(f"Warning: Invalid month values found: {df['month'].unique()}")

# Save cleaned data
df.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")
print(f"Rows: {len(df)}")
print("Sample data:")
print(df.head())