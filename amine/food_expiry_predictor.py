import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os
import joblib

# Construct path to the data file
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "cleaned_data.csv")

# Load data
try:
    df = pd.read_csv(data_path)
    print(f"✅ Successfully loaded data from {data_path}")
except FileNotFoundError:
    print(f"❌ Error: Could not find the data file at {data_path}")
    print("Please ensure 'cleaned_data.csv' exists in the 'data' subdirectory.")
    exit()
except Exception as e:
    print(f"❌ An error occurred while loading the data: {e}")
    exit()

# Features and target
feature_cols = ['temperature', 'humidity', 'type']
target_col = 'shelf_life_days'

if not all(col in df.columns for col in feature_cols):
    print(f"❌ Error: One or more feature columns {feature_cols} not found in the CSV.")
    exit()
if target_col not in df.columns:
    print(f"❌ Error: Target column '{target_col}' not found in the CSV.")
    exit()

# Prepare features: one-hot encode 'type'
X = pd.get_dummies(df['type'], prefix='food_type')
X[['temperature', 'humidity']] = df[['temperature', 'humidity']]
y = df[target_col]

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df['type'])

# Train Random Forest model with tuned parameters
model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# Predictions (ensure non-negative)
y_pred = np.maximum(model.predict(X_test), 0)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Detailed evaluation
print("\n--- Detailed Model Evaluation ---")
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})
print("\n--- Feature Importance ---")
print(importance.sort_values(by='Importance', ascending=False))

# Save the model
model_path = os.path.join(script_dir, "data", "expiry_predictor_model.joblib")
joblib.dump(model, model_path)
print(f"\n✅ Model saved to {model_path}")

# Sample predictions
print("\n--- Sample Predictions ---")
samples = pd.DataFrame({
    'temperature': [20, 5, 3, 18, 2],
    'humidity': [70, 90, 85, 60, 85],
    'type': ['vegetables', 'vegetables', 'dairy', 'bakery', 'meat']
})

# Minimum shelf life constraints
min_shelf_life = {
    'vegetables': 7, 'dairy': 7, 'canned': 180, 'bakery': 1, 'meat': 1, 'dry goods': 100
}

# Prepare samples for prediction
X_samples = pd.get_dummies(samples['type'], prefix='food_type')
for col in X.columns:
    if col not in X_samples.columns and col.startswith('food_type'):
        X_samples[col] = 0
X_samples[['temperature', 'humidity']] = samples[['temperature', 'humidity']]
X_samples = X_samples[X.columns]  # Ensure same column order

food_types = {1: 'vegetables', 2: 'dairy', 3: 'canned', 4: 'bakery', 5: 'meat', 6: 'dry goods'}

for i, row in samples.iterrows():
    pred_days = model.predict(X_samples.iloc[[i]])[0]
    food_type = row['type']
    # Apply minimum shelf life constraint
    pred_days = max(min_shelf_life.get(food_type, 1), pred_days)
    pred_date = pd.Timestamp.today() + pd.Timedelta(days=pred_days)
    print(f"Food type: {food_type}, Temp: {row['temperature']}°C, Humidity: {row['humidity']}%")
    print(f"  → Predicted shelf life: {pred_days:.1f} days")
    print(f"  → Expected expiry date: {pred_date.strftime('%Y-%m-%d')}")
    