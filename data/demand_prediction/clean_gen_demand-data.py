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
df = df.dropna() # Drop rows with any missing values

# Validate and clean data
df = df[df["demand_kg"] > 0]
df = df.drop_duplicates()
df["timestamp"] = df["timestamp"].astype(int)
df["recipient_id"] = df["recipient_id"].astype(int)
df["demand_kg"] = df["demand_kg"].astype(float)
df["is_weekend"] = df["is_weekend"].astype(int)
df["disaster_flag"] = df["disaster_flag"].astype(int)

# Validate scenario patterns
weekend_steps = df[df["is_weekend"] == 1]["timestamp"].unique()
expected_weekends = [250, 500, 750]
if not all(ws in weekend_steps for ws in expected_weekends):
    print(f"Warning: Expected weekend steps {expected_weekends} not fully present. Found {weekend_steps}")

shortage_count = df[df["disaster_flag"] == 1]["timestamp"].nunique()
expected_shortage_range = int(1000 * 0.5)  # ~500 random shortages
if not (expected_shortage_range * 0.8 <= shortage_count <= expected_shortage_range * 1.2):
    print(f"Warning: Expected ~{expected_shortage_range} shortages, found {shortage_count}")

# Save cleaned data
df.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")
print(f"Rows: {len(df)}")
print("Sample data:")
print(df.head())