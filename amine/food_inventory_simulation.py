import pandas as pd
import numpy as np
import os
import joblib
import sys
from datetime import datetime, timedelta

# Add the data directory to the Python path to import dp_scheduler
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
sys.path.append(data_dir)

# Import dp_knapsack from dp_scheduler instead of redefining it
from dp_scheduler import dp_knapsack

# Simulation
def run_simulation(data_path, model_path, timesteps=30, capacity=50):
    """
    Simulate food inventory management, calling dp_knapsack at each timestep with priorities
    calculated using the Random Forest model.
    
    Args:
        data_path (str): Path to cleaned_data.csv.
        model_path (str): Path to expiry_predictor_model.joblib.
        timesteps (int): Number of days to simulate.
        capacity (int): Maximum total quantity to process per timestep.
    """
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Successfully loaded data from {data_path}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        exit()
    
    # Validate required columns
    required_cols = ["item_id", "type", "shelf_life_days", "quantity", "priority", "temperature", "humidity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns in CSV: {missing_cols}")
        exit()
    
    # Load Random Forest model
    try:
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Using raw priority as fallback.")
        model = None
    
    # Minimum shelf life constraints
    min_shelf_life = {
        'vegetables': 7, 'dairy': 7, 'canned': 180, 'bakery': 3, 'meat': 5, 'dry goods': 100
    }
    
    # Prepare feature columns for model
    feature_cols = pd.get_dummies(df['type'], prefix='food_type').columns.tolist()
    feature_cols.extend(['temperature', 'humidity'])
    
    # Initialize inventory
    inventory = df.copy()
    inventory['remaining_shelf_life'] = inventory['shelf_life_days']
    total_waste = 0
    total_sold = 0
    original_df = df.copy()  # For restocking
    
    print(f"\nStarting simulation for {timesteps} days...")
    
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
            total_waste += expired['quantity'].sum()
            print(f"Expired items: {len(expired)} (Total waste: {total_waste} units)")
            print(f"Waste by type:\n{expired['type'].value_counts().to_string()}")
            inventory = inventory[inventory['remaining_shelf_life'] > 0]
        
        # Restock every 7 days
        if day % 7 == 0 and day > 0:
            restock = original_df.sample(n=min(100, len(original_df)), random_state=day)
            restock['remaining_shelf_life'] = restock['shelf_life_days']
            inventory = pd.concat([inventory, restock]).reset_index(drop=True)
            print(f"Restocked {len(restock)} items")
        
        # Calculate adjusted priorities using model (if available)
        if model is not None:
            X = pd.get_dummies(inventory['type'], prefix='food_type')
            for col in feature_cols:
                if col not in X.columns and col.startswith('food_type'):
                    X[col] = 0
            X[['temperature', 'humidity']] = inventory[['temperature', 'humidity']]
            X = X[feature_cols]
            predicted_shelf_life = np.maximum(model.predict(X), 1)
            # Adjust priority
            inventory['adjusted_priority'] = inventory.apply(
                lambda row: (
                    row['priority'] * 0.1 if row['type'] in ['canned', 'dry goods']
                    else row['priority'] * (1 / max(1, row['remaining_shelf_life'])) * (
                        1 / max(min_shelf_life.get(row['type'], 1), predicted_shelf_life[inventory.index.get_loc(row.name)])
                    )
                ),
                axis=1
            )
        else:
            inventory['adjusted_priority'] = inventory['priority']
        
        # Prepare inputs for dp_knapsack
        weights = inventory['quantity'].astype(int).tolist()
        values = inventory['adjusted_priority'].astype(float).tolist()
        
        # Call DP scheduler
        _, selected_idx = dp_knapsack(weights, values, capacity)
        
        if selected_idx:
            selected_items = inventory.iloc[selected_idx]
            total_sold += selected_items['quantity'].sum()
            print(f"Processed {len(selected_items)} items: {selected_items['type'].tolist()}")
            print(f"Details:\n{selected_items[['item_id', 'type', 'quantity', 'adjusted_priority', 'remaining_shelf_life']].to_string(index=False)}")
            print(f"Total sold: {total_sold} units")
            
            # Remove processed items - use boolean mask approach to avoid index errors
            mask = np.ones(len(inventory), dtype=bool)  # Create a boolean mask of True values
            mask[selected_idx] = False  # Set positions in selected_idx to False
            inventory = inventory.loc[mask].reset_index(drop=True)
        else:
            print("No items selected for processing.")
        
        # Simulate environmental changes
        if len(inventory) > 0:
            inventory['temperature'] += np.random.uniform(-0.5, 0.5, len(inventory))
            inventory['temperature'] = inventory['temperature'].clip(0, 25)
    
    print(f"\nSimulation complete.")
    print(f"Final inventory: {len(inventory)} items")
    print(f"Total sold: {total_sold} units")
    print(f"Total waste: {total_waste} units")

# Main execution
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "cleaned_data.csv")
    model_path = os.path.join(script_dir, "data", "expiry_predictor_model.joblib")
    
    run_simulation(data_path, model_path, timesteps=30, capacity=50)
