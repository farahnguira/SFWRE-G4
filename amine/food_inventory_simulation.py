import pandas as pd
import numpy as np
import os
import joblib
import sys
from datetime import datetime, timedelta

# Try to import matplotlib; provide instructions if not installed
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("Error: matplotlib package is not installed.")
    print("Please install it using the following command:")
    print("pip install matplotlib")
    print("\nAfter installation, run this script again.")
    sys.exit(1)

# Add the data directory to the Python path to import dp_scheduler
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
sys.path.append(data_dir)

# Import dp_knapsack from dp_scheduler
from dp_scheduler import dp_knapsack

# Simulation
def run_simulation(data_path, model_path, timesteps=30, capacity=100, scenario="urban", output_dir="data/outputs"):
    """
    Simulate food redistribution using DP scheduler for allocation.
    
    Args:
        data_path (str): Path to cleaned_data.csv.
        model_path (str): Path to expiry_predictor_model.joblib.
        timesteps (int): Number of days to simulate.
        capacity (int): Maximum total quantity to process per timestep.
        scenario (str): 'urban', 'spike', or 'disaster'.
        output_dir (str): Directory for output CSVs and plots.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Print dataset stats
    print("Food type distribution:\n", df['type'].value_counts().to_string())
    print("Shelf life stats by type:\n", df.groupby('type').agg({'shelf_life_days': ['mean', 'min', 'max']}).to_string())
    print("Quantity stats:\n", df['quantity'].describe().to_string())
    print("Priority stats by type:\n", df.groupby('type').agg({'priority': ['mean', 'min', 'max']}).to_string())
    
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
    total_priority = 0
    original_df = df.copy()  # For restocking
    
    # Track metrics
    waste_history = []
    sold_history = []
    priority_history = []
    allocated_items = []
    actual_days_simulated = 0  # Track actual number of days simulated
    
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
            total_waste += expired['quantity'].sum()
            print(f"Expired items: {len(expired)} (Total waste: {total_waste} units)")
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
        
        # Calculate adjusted priorities
        if model is not None:
            X = pd.get_dummies(inventory['type'], prefix='food_type')
            for col in feature_cols:
                if col not in X.columns and col.startswith('food_type'):
                    X[col] = 0
            X[['temperature', 'humidity']] = inventory[['temperature', 'humidity']]
            X = X[feature_cols]
            raw_predicted_shelf_life = model.predict(X)
            
            # Cap predictions to prevent severe underestimation (at least 50% of actual)
            predicted_shelf_life = np.maximum(raw_predicted_shelf_life, inventory['shelf_life_days'].values * 0.5)
            
            # Blend predictions with actuals (60% predicted, 40% actual)
            predicted_shelf_life = 0.6 * predicted_shelf_life + 0.4 * inventory['shelf_life_days'].values
            
            # Ensure minimum shelf life constraints
            for i, food_type in enumerate(inventory['type']):
                predicted_shelf_life[i] = max(min_shelf_life.get(food_type, 1), predicted_shelf_life[i])
            
            # Create DataFrame separately to avoid f-string nesting issues
            sample_df = pd.DataFrame({
                'type': inventory['type'][:5],
                'predicted': predicted_shelf_life[:5].round(2),
                'actual': inventory['shelf_life_days'][:5]
            })
            print(f"Sample predicted vs. actual shelf life:\n{sample_df.to_string(index=False)}")
            
            # Enhanced priority adjustment with boost for bakery and meat
            inventory['adjusted_priority'] = inventory.apply(
                lambda row: (
                    row['priority'] * 0.5 if row['type'] in ['canned', 'dry goods']
                    else row['priority'] * (1 / max(1, row['remaining_shelf_life']) ** 1.5) * (
                        1 / max(min_shelf_life.get(row['type'], 1), predicted_shelf_life[inventory.index.get_loc(row.name)]) ** 1.5
                    ) * (2.0 if row['type'] in ['bakery', 'meat'] else 1.0)  # Boost bakery/meat
                ),
                axis=1
            )
        else:
            inventory['adjusted_priority'] = inventory['priority']
        
        # Use DP scheduler
        weights = inventory['quantity'].astype(int).tolist()
        values = inventory['adjusted_priority'].astype(float).tolist()
        _, selected_idx = dp_knapsack(weights, values, capacity)
        
        # Process selected items
        if selected_idx:
            selected_items = inventory.iloc[selected_idx].copy()  # Create an explicit copy to avoid SettingWithCopyWarning
            day_sold = selected_items['quantity'].sum()
            day_priority = selected_items['adjusted_priority'].sum()
            total_sold += day_sold
            total_priority += day_priority
            print(f"Processed {len(selected_items)} items: {selected_items['type'].tolist()}")
            print(f"Details:\n{selected_items[['item_id', 'type', 'quantity', 'adjusted_priority', 'remaining_shelf_life']].to_string(index=False)}")
            print(f"Day priority: {day_priority:.2f} (Total: {total_priority:.2f})")
            
            # Log allocated items - add day column to the copied DataFrame
            selected_items['day'] = day + 1
            allocated_items.append(selected_items[['day', 'item_id', 'type', 'quantity', 'adjusted_priority']])
            
            # Remove processed items - use boolean mask approach to avoid index errors
            mask = np.ones(len(inventory), dtype=bool)
            mask[selected_idx] = False
            inventory = inventory.loc[mask].reset_index(drop=True)
        else:
            print("No items selected for processing.")
        
        # Update metrics
        waste_history.append(total_waste)
        sold_history.append(total_sold)
        priority_history.append(total_priority)
        actual_days_simulated = day + 1  # Update the actual number of days simulated
        
        # Simulate environmental changes
        if len(inventory) > 0:
            inventory['temperature'] += np.random.uniform(-0.5, 0.5, len(inventory))
            inventory['temperature'] = inventory['temperature'].clip(0, 25)
    
    # Calculate waste reduction percentage
    initial_units = df['quantity'].sum() + restock_volume * 4  # Initial + restocks
    waste_percentage = (total_waste / initial_units) * 100
    waste_reduction = 100 - waste_percentage
    
    print(f"\nSimulation complete.")
    print(f"Final inventory: {len(inventory)} items")
    print(f"Total sold: {total_sold} units")
    print(f"Total waste: {total_waste} units")
    print(f"Waste percentage: {waste_percentage:.2f}%")
    print(f"Waste reduction: {waste_reduction:.2f}%")
    print(f"Total priority: {total_priority:.2f}")
    
    # Save allocated items CSV
    if allocated_items:
        allocated_df = pd.concat(allocated_items, ignore_index=True)
        allocated_df.to_csv(os.path.join(output_dir, f"allocated_items_{scenario}_cap{capacity}.csv"), index=False)
        print(f"Allocated items saved to {output_dir}/allocated_items_{scenario}_cap{capacity}.csv")
    
    # If the simulation ended early, pad the metrics arrays
    if len(waste_history) < timesteps:
        # Pad with the last value to maintain trend
        last_waste = waste_history[-1] if waste_history else 0
        last_sold = sold_history[-1] if sold_history else 0
        last_priority = priority_history[-1] if priority_history else 0
        
        # Add padding
        waste_history.extend([last_waste] * (timesteps - len(waste_history)))
        sold_history.extend([last_sold] * (timesteps - len(sold_history)))
        priority_history.extend([last_priority] * (timesteps - len(priority_history)))
    
    # Save priority history CSV using timesteps
    priority_df = pd.DataFrame({
        'day': range(1, timesteps + 1),
        'total_priority': priority_history
    })
    priority_df.to_csv(os.path.join(output_dir, f"priority_history_{scenario}_cap{capacity}.csv"), index=False)
    print(f"Priority history saved to {output_dir}/priority_history_{scenario}_cap{capacity}.csv")
    
    # Visualization using timesteps
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(range(1, timesteps + 1), waste_history, label='Total Waste')
    plt.title(f'Waste Over Time ({scenario}, Capacity={capacity})')
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(1, timesteps + 1), sold_history, label='Total Sold', color='green')
    plt.title(f'Sales Over Time ({scenario}, Capacity={capacity})')
    plt.xlabel('Day')
    plt.ylabel('Units')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"metrics_{scenario}_cap{capacity}.png"))
    plt.close()
    print(f"Metrics plot saved to {output_dir}/metrics_{scenario}_cap{capacity}.png")
    
    return total_sold, total_waste, total_priority, waste_reduction

# Main execution
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "cleaned_data.csv")
    model_path = os.path.join(script_dir, "data", "expiry_predictor_model.joblib")
    output_dir = os.path.join(script_dir, "data", "outputs")
    
    # Run simulations for all capacities and scenarios
    capacities = [50, 100, 200, 400, 600]  # Added higher capacity values (400, 600)
    scenarios = ["urban", "spike", "disaster"]
    results = []
    
    for scenario in scenarios:
        for capacity in capacities:
            print(f"\n=== Running {scenario} scenario with capacity {capacity} ===")
            total_sold, total_waste, total_priority, waste_reduction = run_simulation(
                data_path, model_path, timesteps=30, capacity=capacity, scenario=scenario, output_dir=output_dir
            )
            results.append({
                'scenario': scenario,
                'capacity': capacity,
                'total_sold': total_sold,
                'total_waste': total_waste,
                'total_priority': total_priority,
                'waste_reduction': waste_reduction  
            })
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "simulation_results.csv"), index=False)
    print(f"Results summary saved to {output_dir}/simulation_results.csv")
