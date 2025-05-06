#!/usr/bin/env python3
import argparse
import os
import pandas as pd
from datetime import datetime

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

def save_to_csv(df, selected_idx, output_dir="outputs", scenario="standard", capacity=50):
    """
    Save the selected items and summary metrics to CSV files.
    
    Args:
        df: The DataFrame containing all items
        selected_idx: Indices of selected items
        output_dir: Directory to save output CSV files
        scenario: Scenario name (e.g., 'urban', 'spike', 'disaster')
        capacity: Capacity used for this allocation
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames - use a more file-system friendly format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save selected items
    if selected_idx:
        selected_df = df.iloc[selected_idx].copy()
        selected_df['selection_date'] = datetime.now().strftime("%Y-%m-%d")
        selected_df['scenario'] = scenario
        selected_df['capacity'] = capacity
        selected_path = os.path.join(output_dir, f"selected_items_{scenario}_cap{capacity}_{timestamp}.csv")
        selected_df.to_csv(selected_path, index=False)
        print(f"✅ Selected items saved to: {selected_path}")
        
        # Calculate and save summary metrics
        total_items = len(selected_idx)
        total_quantity = selected_df["quantity"].sum()
        total_priority = selected_df["priority"].sum()
        
        # Calculate wasted items (if expiry_date column exists)
        wasted_items = 0
        if 'expiry_date' in df.columns:
            today = pd.Timestamp.today().normalize()
            wasted_mask = pd.to_datetime(df['expiry_date']) < today
            wasted_items = df[wasted_mask]["quantity"].sum()
        
        # Food type distribution analysis (if type column exists)
        type_distribution = {}
        if 'type' in selected_df.columns:
            # Group by type and sum quantities
            type_distribution = selected_df.groupby('type')['quantity'].sum().to_dict()
            
        # Expiry analysis (if expiry_date exists)
        avg_shelf_life = None
        urgent_items = 0
        if 'expiry_date' in selected_df.columns:
            today = pd.Timestamp.today().normalize()
            expiry_dates = pd.to_datetime(selected_df['expiry_date'])
            days_until_expiry = (expiry_dates - today).dt.days
            
            # Average remaining shelf life in days
            avg_shelf_life = days_until_expiry.mean()
            
            # Count items expiring within 3 days (urgent delivery needed)
            urgent_items = sum(selected_df.loc[days_until_expiry <= 3, 'quantity'])
        
        # Create enhanced summary DataFrame
        summary_data = {
            'date': [datetime.now().strftime("%Y-%m-%d")],
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'scenario': [scenario],  # Use the scenario parameter
            'algorithm': ['dp_knapsack'],
            'capacity': [capacity],  # Add capacity to summary
            'total_items_selected': [total_items],
            'total_quantity_delivered': [total_quantity],
            'total_priority_score': [total_priority],
            'food_wasted': [wasted_items],
        }
        
        # Add average shelf life if available
        if avg_shelf_life is not None:
            summary_data['avg_remaining_shelf_life'] = [round(avg_shelf_life, 1)]
            summary_data['urgent_delivery_items'] = [urgent_items]
        
        # Add type distribution if available
        for food_type, quantity in type_distribution.items():
            summary_data[f'quantity_{food_type}'] = [quantity]
        
        summary_df = pd.DataFrame(summary_data)
        
        summary_path = os.path.join(output_dir, f"summary_{scenario}_cap{capacity}_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Enhanced summary metrics saved to: {summary_path}")
        
        # Return the summary for potential use by other components
        return summary_df
    
    return None

def main():
    # Get this script's directory for path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="DP-based knapsack scheduler")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(script_dir, "cleaned_data.csv"),
        help="Path to cleaned CSV with columns: item_id, type, expiry_date, quantity, priority"
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=600,
        help="Maximum total quantity (knapsack capacity)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(script_dir, "outputs"),
        help="Directory to save output CSV files"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="standard",
        choices=["standard", "urban", "spike", "disaster"],
        help="Simulation scenario type"
    )
    parser.add_argument(
        "--no-export-csv",
        action="store_true",
        help="Disable exporting results to CSV files"
    )
    args = parser.parse_args()

    # Load data with better error handling
    try:
        df = pd.read_csv(args.input)
        print(f"✅ Successfully loaded data from {args.input}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find the CSV file at {args.input}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        print(f"Try running the script with --input=PATH_TO_CLEANED_DATA_CSV")
        return
    
    # Add validation for required columns
    required_cols = ["item_id", "quantity", "priority"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns in CSV: {missing_cols}")
        return
    
    # Validate numeric data
    try:
        weights = df["quantity"].astype(int).tolist()
        values = df["priority"].astype(int).tolist()
        ids = df["item_id"].tolist()
    except ValueError:
        print("❌ Error: 'quantity' and 'priority' columns must contain numeric values")
        return
    
    # Handle empty data
    if len(weights) == 0:
        print("⚠️ Warning: No items found in the data")
        return
        
    # Solve
    _, selected_idx = dp_knapsack(weights, values, args.capacity)
    selected_ids = [ids[i] for i in selected_idx]
    total_value = sum(values[i] for i in selected_idx)
    total_weight = sum(weights[i] for i in selected_idx)
    
    print("Selected item IDs:", selected_ids)
    print("Total priority:", total_value)
    print("Total weight:", total_weight)
    print("Total items selected:", len(selected_ids))
    
    # Optional: Show selected items details
    if selected_idx:
        selected_df = df.iloc[selected_idx]
        print("\nSelected items:")
        print(selected_df[["item_id", "type", "quantity", "priority"]].to_string(index=False))
    
    # Export to CSV by default, unless explicitly disabled
    if not args.no_export_csv:
        save_to_csv(df, selected_idx, args.output_dir, args.scenario, args.capacity)

if __name__ == "__main__":
    main()
