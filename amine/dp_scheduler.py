#!/usr/bin/env python3
import argparse
import os
import pandas as pd

def dp_knapsack(weights, values, capacity):
    n = len(weights)
    # Build DP table: (n+1) rows × (capacity+1) cols
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    # Fill table
    for i in range(1, n + 1):
        wi, vi = weights[i-1], values[i-1]
        for w in range(capacity + 1):
            if wi > w:
                dp[i][w] = dp[i-1][w]
            else:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-wi] + vi)

    # Backtrack to find which items were taken
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            selected.append(i-1)       # store zero-based index
            w -= weights[i-1]
    selected.reverse()
    return dp, selected

def main():
    parser = argparse.ArgumentParser(description="DP-based knapsack scheduler")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join("data", "cleaned_data.csv"),
        help="Path to cleaned CSV with columns: item_id, type, expiry_date, quantity, priority"
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=50,
        help="Maximum total quantity (knapsack capacity)"
    )
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)
    weights = df["quantity"].astype(int).tolist()
    values  = df["priority"].astype(int).tolist()
    ids     = df["item_id"].tolist()

    # Solve
    _, selected_idx = dp_knapsack(weights, values, args.capacity)
    selected_ids = [ids[i] for i in selected_idx]
    total_value  = sum(values[i] for i in selected_idx)

    print("Selected item IDs:", selected_ids)
    print("Total priority  :", total_value)

if __name__ == "__main__":
    main()
