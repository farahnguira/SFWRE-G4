import csv
from dp_scheduler import dp_knapsack  # or whatever your function is named

import os, csv, sys

def load_first_n_items(filename, n=5):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f" script_dir = {script_dir}")
    print(" contains:", os.listdir(script_dir))
    csv_path = os.path.join(script_dir, filename)
    print(f" attempting to open {csv_path}")
    items = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            items.append({
                "weight": int(row["quantity"]),
                "value": int(row["priority"])
            })
    return items

if __name__ == "__main__":
    items = load_first_n_items("cleaned_data.csv", n=50)
    capacity = 600  # Increased capacity to 600
    print("Items loaded:")

    weights = [item["weight"] for item in items]
    values  = [item["value"]  for item in items]

    #unpack the return values
    _, selected_indices = dp_knapsack(weights, values, capacity)
    total_value = sum(values[i] for i in selected_indices)
    
    print("Selected item IDs:", selected_indices)
    print("Total priority:    ", total_value)
    print("Total weight:      ", sum(weights[i] for i in selected_indices))
    print("Total items:      ", len(selected_indices))