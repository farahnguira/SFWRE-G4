# dash_app/use_data.py

import pandas as pd

def load_cleaned_data(path="amine/data/cleaned_data.csv"):
    """Load the cleaned donation/demand CSV."""
    df = pd.read_csv(path, parse_dates=["expiry_date"])
    return df

def aggregate_daily(df):
    """Group by expiry_date, summing up any numeric columns."""
    daily = df.groupby("expiry_date").sum().reset_index()
    return daily

def make_sequences(daily, window=7):
    """
    Build input/output sequences for an RNN:
    - X: sliding windows of length=window on the target column (e.g. 'quantity')
    - y: the next-day target
    """
    values = daily["quantity"].values
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i : i + window])
        y.append(values[i + window])
    X = pd.DataFrame(X)
    y = pd.Series(y, name="target")
    return X, y

if __name__ == "__main__":
    df = load_cleaned_data()
    daily = aggregate_daily(df)
    X, y = make_sequences(daily)
    print("Daily shape:", daily.shape)
    print("X shape:", X.shape, "y shape:", y.shape)