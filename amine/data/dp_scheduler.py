import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

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
            selected.append(i-1)  # store zero-based index
            w -= weights[i-1]
    selected.reverse()
    return dp, selected

def save_to_csv(df, selected_idx, output_dir="outputs", scenario="standard", capacity=50, day=1):
    """
    Save the selected items and summary metrics to CSV files.
    
    Args:
        df: The DataFrame containing all items
        selected_idx: Indices of selected items
        output_dir: Directory to save output CSV files
        scenario: Scenario name (e.g., 'urban', 'spike', 'disaster')
        capacity: Capacity used for this allocation
        day: The simulation day (default 1)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save selected items
    if selected_idx:
        selected_df = df.iloc[selected_idx].copy()
        selected_df['selection_date'] = datetime.now().strftime("%Y-%m-%d")
        selected_df['scenario'] = scenario
        selected_df['capacity'] = capacity
        selected_path = os.path.join(output_dir, f"selected_items_{scenario}_cap{capacity}_day{day}_{timestamp}.csv")
        selected_df.to_csv(selected_path, index=False)
        print(f"Selected items saved to: {selected_path}")
        
        # Calculate and save summary metrics
        total_items = len(selected_idx)
        total_quantity = selected_df["quantity"].sum()
        total_priority = selected_df["priority"].sum()
        
        # Calculate wasted items (if expiry_date column exists)
        wasted_items = 0
        if 'remaining_shelf_life' in df.columns:
            today = pd.Timestamp("2025-05-06") + pd.to_timedelta(day - 1, unit="d")
            wasted_mask = df['remaining_shelf_life'] <= 0
            wasted_items = df[wasted_mask]["quantity"].sum()
        
        # Food type distribution analysis
        type_distribution = selected_df.groupby('type')['quantity'].sum().to_dict()
            
        # Expiry analysis
        avg_shelf_life = None
        urgent_items = 0
        if 'remaining_shelf_life' in df.columns:
            today = pd.Timestamp("2025-05-06") + pd.to_timedelta(day - 1, unit="d")
            days_until_expiry = df.iloc[selected_idx]['remaining_shelf_life']
            avg_shelf_life = days_until_expiry.mean()
            urgent_items = sum(selected_df.loc[days_until_expiry <= 3, 'quantity'])
        
        # Create enhanced summary DataFrame
        summary_data = {
            'date': [(pd.to_datetime("2025-05-06") + pd.to_timedelta(day - 1, unit="d")).strftime("%Y-%m-%d")],
            'timestamp': [(pd.to_datetime("2025-05-06") + pd.to_timedelta(day - 1, unit="d")).strftime("%Y-%m-%d %H:%M:%S")],
            'scenario': [scenario],
            'algorithm': ['dp_knapsack'],
            'capacity': [capacity],
            'total_items_selected': [total_items],
            'total_quantity_delivered': [total_quantity],
            'total_priority_score': [total_priority],
            'food_wasted': [wasted_items],
        }
        
        # Add average shelf life if available
        if avg_shelf_life is not None:
            summary_data['avg_remaining_shelf_life'] = [round(avg_shelf_life, 1)]
            summary_data['urgent_delivery_items'] = [urgent_items]
        
        # Add type distribution
        for food_type, quantity in type_distribution.items():
            summary_data[f'quantity_{food_type}'] = [quantity]
        
        summary_df = pd.DataFrame(summary_data)
        
        summary_path = os.path.join(output_dir, f"summary_{scenario}_cap{capacity}_day{day}_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Enhanced summary metrics saved to: {summary_path}")
        
        return summary_df
    
    return None

def run_simulation(data_path, timesteps=30, capacity=600, scenario="urban", output_dir="data/tarji"):
    """
    Simulate food redistribution using DP scheduler for allocation.
    
    Args:
        data_path (str): Path to cleaned_data.csv.
        timesteps (int): Number of days to simulate.
        capacity (int): Maximum total quantity to process per timestep.
        scenario (str): 'urban', 'spike', or 'disaster'.
        output_dir (str): Directory for output CSVs.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f" Successfully loaded data from {data_path}")
    except Exception as e:
        print(f" Error loading data: {e}")
        exit()
    
    # Validate required columns
    required_cols = ["item_id", "type", "shelf_life_days", "quantity", "priority"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f" Error: Missing required columns in CSV: {missing_cols}")
        exit()
    
    # Initialize inventory
    inventory = df.copy()
    inventory['remaining_shelf_life'] = inventory['shelf_life_days']
    original_df = df.copy()  # For restocking
    
    print(f"\nStarting {scenario} simulation for {timesteps} days (capacity={capacity})...")
    
    # Simulate over timesteps
    for day in range(timesteps):
        print(f"\n--- Day {day + 1} ---")
        if len(inventory) == 0:
            print("Inventory empty. Stopping simulation.")
            break
        
        # Update remaining shelf life
        inventory['remaining_shelf_life'] -= 1
        
        # Check for expired items
        expired = inventory[inventory['remaining_shelf_life'] <= 0]
        if not expired.empty:
            print(f"Expired items: {len(expired)} (Total waste: {expired['quantity'].sum()} units)")
            print(f"Waste by type:\n{expired['type'].value_counts().to_string()}")
            inventory = inventory[inventory['remaining_shelf_life'] > 0]
        
        # Restock every 7 days
        restock_volume = 100
        if day % 7 == 0 and day > 0:
            restock_weights = original_df['shelf_life_days'] / original_df['shelf_life_days'].sum()
            restock = original_df[original_df['type'].isin(['canned', 'dry goods', 'vegetables', 'dairy'])].sample(
                n=min(restock_volume, len(original_df)), random_state=day, replace=True, weights=restock_weights
            )
            restock['remaining_shelf_life'] = restock['shelf_life_days']
            inventory = pd.concat([inventory, restock]).reset_index(drop=True)
            print(f"Restocked {len(restock)} items: {restock['type'].value_counts().to_string()}")
        
        # Simulate scenarios
        if scenario == "spike" and day in [6, 7, 13, 14]:  # Weekend spike
            restock_weights = original_df['shelf_life_days'] / original_df['shelf_life_days'].sum()
            spike = original_df.sample(
                n=restock_volume * 2, random_state=day, replace=True, weights=restock_weights
            )
            spike['remaining_shelf_life'] = spike['shelf_life_days']
            inventory = pd.concat([inventory, spike]).reset_index(drop=True)
            print(f"Weekend spike: Added {len(spike)} items: {spike['type'].value_counts().to_string()}")
        
        if scenario == "disaster" and 19 <= day <= 24:  # Disaster increased demand
            print("Disaster scenario: Increased demand simulated (no inventory impact).")
        
        # Use DP scheduler
        weights = inventory['quantity'].astype(int).tolist()
        values = inventory['priority'].astype(float).tolist()
        _, selected_idx = dp_knapsack(weights, values, capacity)
        
        # Process selected items and save to CSV
        if selected_idx:
            # Call save_to_csv
            summary_df = save_to_csv(
                inventory,
                selected_idx,
                output_dir=output_dir,
                scenario=scenario,
                capacity=capacity,
                day=day + 1
            )
            
            # Remove processed items
            mask = np.ones(len(inventory), dtype=bool)
            mask[selected_idx] = False
            inventory = inventory.loc[mask].reset_index(drop=True)
        else:
            print("No items selected for processing.")

def main():
    # Get this script's directory for path resolution
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description="DP-based knapsack scheduler")
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(script_dir,"cleaned_data.csv"),
        help="Path to cleaned CSV with columns: item_id, type, shelf_life_days, quantity, priority"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(script_dir, "data", "tarji"),
        help="Directory to save output CSV files"
    )
    args = parser.parse_args()

    # Define scenarios and capacities
    scenarios = ["urban", "spike", "disaster"]
    capacities = [50, 100, 200, 400, 600]
    timesteps = 30

    # Run simulations for all scenarios and capacities
    for scenario in scenarios:
        for capacity in capacities:
            print(f"\n=== Running {scenario} scenario with capacity {capacity} ===")
            run_simulation(
                data_path=args.input,
                timesteps=timesteps,
                capacity=capacity,
                scenario=scenario,
                output_dir=args.output_dir
            )

if __name__ == "__main__":
    main()