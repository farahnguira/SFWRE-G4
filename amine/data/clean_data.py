import os
import pandas as pd
import random
from datetime import datetime

# 1. Paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
# Use processed_data.csv that's already in the same directory
RAW_PATH     = os.path.join(BASE_DIR, "processed_data.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")

# 2. Load raw data, parsing dates
df = pd.read_csv(RAW_PATH, parse_dates=["expiry_date"])

# 3. Initial data validation
for col in ["item_id", "type", "food_type_code", "expiry_date", "shelf_life_days", "quantity", "priority", "temperature", "humidity"]:
    assert df[col].notnull().all(), f"Nulls found in '{col}'"

assert pd.api.types.is_datetime64_any_dtype(df["expiry_date"]), \
        "expiry_date column is not datetime"

today = pd.Timestamp.today().normalize()
assert (df["expiry_date"] >= today).all(), "Some expiry_date values are before today"

print("✅ Raw data loaded and validated.")

# 4. Type normalization map
synonym_map = {
    "veg":        "vegetables",
    "vegetable":  "vegetables",
    "vegetables": "vegetables",
    "dairy":      "dairy",
    "dairy food": "dairy",
    "canned":     "canned",
    "canned food":"canned",
    "bakery":     "bakery",
    "bakery food":"bakery",
    "meat":       "meat",
    "meat food":  "meat",
    "dry":        "dry goods",
    "dry goods":  "dry goods"
}

df["type"] = (
    df["type"]
      .str.lower()
      .str.strip()
      .map(synonym_map)
      .fillna(df["type"])
)

# 5. Drop duplicates
df = df.drop_duplicates()

# 6. Re-validate after cleaning
# Note: We keep all columns now since processed_data.csv already has them
for col in ["item_id", "type", "food_type_code", "expiry_date", "shelf_life_days", "quantity", "priority", "temperature", "humidity"]:
    assert df[col].notnull().all(), f"Nulls found in '{col}'"
assert pd.api.types.is_datetime64_any_dtype(df["expiry_date"])
assert (df["expiry_date"] >= today).all(), "Some expiry_date values are before today"

print("✅ Data cleaned and re-validated.")

# Since processed_data.csv already has all the required fields, we can skip the following section
# and just ensure the columns are in the desired order

# Define the desired column order
desired_order = [
    "item_id", "type", "food_type_code", "expiry_date", "shelf_life_days",
    "quantity", "priority", "temperature", "humidity"
]
df = df[desired_order]

# 8. Save cleaned + enriched dataset
df.to_csv(CLEANED_PATH, index=False)
print(f"✅ Cleaned data saved to: {CLEANED_PATH}")
print("✅ Data processing complete.")