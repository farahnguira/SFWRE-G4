import pandas as pd
import os

def clean_demand_data(input_path, output_path):
    df = pd.read_csv(input_path)
    # Ensure no nulls
    assert df.isnull().sum().sum() == 0, "Null values detected"
    # Validate data types
    df["timestamp"] = df["timestamp"].astype(int)
    df["demand_kg"] = df["demand_kg"].astype(float)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["disaster_flag"] = df["disaster_flag"].astype(int)
    # Ensure positive demand
    assert (df["demand_kg"] >= 0).all(), "Negative demand detected"
    # Save cleaned data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    df = clean_demand_data("amine/demand_data.csv", "amine/cleaned_demand_data.csv")
    print("Cleaned Dataset Info:")
    print(df.info())
    print("\nSample Rows:")
    print(df.head())