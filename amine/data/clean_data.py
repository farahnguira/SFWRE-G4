import pandas as pd
import random
from datetime import datetime
import argparse

def clean_data(input_path, output_path, large_output_path=None):
    # 1. Load raw data, parsing dates
    df = pd.read_csv(input_path, parse_dates=["expiry_date"])
    
    # 2. Initial data validation
    for col in ["item_id", "type", "food_type_code", "expiry_date", "shelf_life_days", "quantity", "priority", "temperature", "humidity"]:
        assert df[col].notnull().all(), f"Nulls found in '{col}'"

    assert pd.api.types.is_datetime64_any_dtype(df["expiry_date"]), \
            "expiry_date column is not datetime"

    today = pd.Timestamp.today().normalize()
    assert (df["expiry_date"] >= today).all(), "Some expiry_date values are before today"

    print("✅ Raw data loaded and validated.")
    
    # 3. Type normalization
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
    
    # 4. Drop duplicates
    df = df.drop_duplicates()
    
    # 5. Re-validate after cleaning
    for col in ["item_id", "type", "food_type_code", "expiry_date", "shelf_life_days", "quantity", "priority", "temperature", "humidity"]:
        assert df[col].notnull().all(), f"Nulls found in '{col}'"
    assert pd.api.types.is_datetime64_any_dtype(df["expiry_date"])
    assert (df["expiry_date"] >= today).all(), "Some expiry_date values are before today"

    print("✅ Data cleaned and re-validated.")
    
    # 6. Ensure columns are in desired order
    desired_order = [
        "item_id", "type", "food_type_code", "expiry_date", "shelf_life_days",
        "quantity", "priority", "temperature", "humidity"
    ]
    df = df[desired_order]
    
    # 7. Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")
    
    # 8. Optionally save a copy as cleaned_data_large.csv if requested
    if large_output_path:
        df.to_csv(large_output_path, index=False)
        print(f"✅ Larger cleaned dataset saved to: {large_output_path}")
    
    print("✅ Data processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean food data')
    parser.add_argument('--create-large', action='store_true', help='Also create cleaned_data_large.csv')
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(BASE_DIR, "processed_data.csv")
    CLEANED_PATH = os.path.join(BASE_DIR, "cleaned_data.csv")
    LARGE_PATH = os.path.join(BASE_DIR, "cleaned_data_large.csv") if args.create_large else None
    
    clean_data(RAW_PATH, CLEANED_PATH, LARGE_PATH)
