import os
import pandas as pd

# 1. Paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RAW_PATH     = os.path.join(BASE_DIR, "raw_data.csv")
CLEANED_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")

# 2. Load raw data, parsing dates
df = pd.read_csv(RAW_PATH, parse_dates=["expiry_date"])

# 1. No nulls in any critical column
for col in ["item_id", "type", "expiry_date", "quantity", "priority"]:
    assert df[col].notnull().all(), f"Nulls found in '{col}'"

# 2. expiry_date really is datetime
assert pd.api.types.is_datetime64_any_dtype(df["expiry_date"]), \
        "expiry_date column is not datetime"

# 3. (Optional) No expiry dates in the past
today = pd.Timestamp.today().normalize()
assert (df["expiry_date"] >= today).all(), "Some expiry_date values are before today"

print("All data checks passed.")
# 3. Define your type normalization map
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

# 4. Normalize the 'type' column
df["type"] = (
    df["type"]
      .str.lower()
      .str.strip()
      .map(synonym_map)
      .fillna(df["type"])    # leave anything unexpected unchanged
)

# 5. Drop exact duplicates
df = df.drop_duplicates()

# 6. Data quality checks
assert df["item_id"].notnull().all(),      "Null values in item_id"
assert df["type"].notnull().all(),         "Null values in type"
assert df["expiry_date"].notnull().all(),  "Null values in expiry_date"
assert pd.api.types.is_datetime64_any_dtype(df["expiry_date"]), "expiry_date not datetime"
today = pd.Timestamp.today().normalize()
assert (df["expiry_date"] >= today).all(), "Some expiry_date values are before today"
assert df["quantity"].notnull().all(),     "Null values in quantity"
assert df["priority"].notnull().all(),     "Null values in priority"

# 7. Save the cleaned CSV
df.to_csv(CLEANED_PATH, index=False)
print(f"Cleaned data saved to {CLEANED_PATH}")
