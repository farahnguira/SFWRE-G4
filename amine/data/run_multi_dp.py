#!/usr/bin/env python3
"""
Run the DP scheduler for multiple scenarios and capacities.
This script generates all the summary CSVs needed for the dashboard.
"""
import os
import subprocess
import pandas as pd
from datetime import datetime

def run_dp_scheduler_for_all_configs():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "cleaned_data.csv")
    output_dir = os.path.join(script_dir, "outputs")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define scenarios and capacities
    scenarios = ["urban", "spike", "disaster"]
    capacities = [50, 100, 200]
    
    results = []
    
    for scenario in scenarios:
        for capacity in capacities:
            print(f"\n=== Running DP scheduler for {scenario} scenario with capacity {capacity} ===")
            
            # Run the DP scheduler as a subprocess
            cmd = [
                "python", 
                os.path.join(script_dir, "dp_scheduler.py"),
                f"--input={data_path}",
                f"--capacity={capacity}",
                f"--output-dir={output_dir}",
                f"--scenario={scenario}"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                # Parse output for metrics
                total_priority = None
                total_weight = None
                total_items = None
                
                for line in result.stdout.split('\n'):
                    if "Total priority:" in line:
                        total_priority = float(line.split(":")[-1].strip())
                    elif "Total weight:" in line:
                        total_weight = int(line.split(":")[-1].strip())
                    elif "Total items selected:" in line:
                        total_items = int(line.split(":")[-1].strip())
                
                # Add to results
                results.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'scenario': scenario,
                    'capacity': capacity,
                    'total_items': total_items,
                    'total_quantity': total_weight,
                    'total_priority': total_priority
                })
                
            except subprocess.CalledProcessError as e:
                print(f"Error running DP scheduler: {e}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
    
    # Create a summary CSV with all results
    if results:
        results_df = pd.DataFrame(results)
        summary_path = os.path.join(output_dir, f"combined_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\n✅ Combined results saved to: {summary_path}")

if __name__ == "__main__":
    run_dp_scheduler_for_all_configs()
