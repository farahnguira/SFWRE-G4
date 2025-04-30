import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
num_items = 500  # Reduced number of items to increase likelihood of zero days
start_date = datetime(2025, 10, 1)
end_date = datetime(2026, 4, 30)
num_days = (end_date - start_date).days + 1
food_types = ["vegetables", "dairy", "canned", "bakery", "meat", "dry goods"]

# Generate dates
dates = [start_date + timedelta(days=x) for x in range(num_days)]

# Generate data
data = []
for _ in range(num_items):
    # Randomly select an expiry date
    expiry_date = np.random.choice(dates)
    
    # Donation date should be before expiry date (between 1 and 30 days prior)
    days_before_expiry = np.random.randint(1, 31)
    donation_date = expiry_date - timedelta(days=days_before_expiry)
    
    # Ensure donation date is not before the start date
    if donation_date < start_date:
        donation_date = start_date
    
    food_type = np.random.choice(food_types)
    # Increase chance of zero quantity
    quantity = 0 if np.random.random() < 0.3 else np.random.randint(1, 21)  # 30% chance of quantity 0
    priority = np.random.randint(1, 6)  # Priority between 1 and 5
    
    data.append({
        "donation_date": donation_date,
        "expiry_date": expiry_date,
        "type": food_type,
        "quantity": quantity,
        "priority": priority
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("amine/data/raw_data.csv", index=False)
print("Generated raw_data.csv with donation_date column and increased zero quantities.")