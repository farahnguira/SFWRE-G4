import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import functions from use_data.py
from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data, add_temporal_features

def add_priority_feature(daily, original_df):
    """Add the priority feature to the daily DataFrame for classification."""
    try:
        priority_by_date = original_df.groupby(original_df['expiry_date'].dt.date)['priority'].mean().reset_index()
        daily = daily.merge(
            priority_by_date,
            left_on=daily['expiry_date'].dt.date,
            right_on='expiry_date',
            how='left'
        )
        daily['priority'] = daily['priority'].fillna(daily['priority'].mean())
        logger.info("Priority feature added successfully.")
        return daily
    except Exception as e:
        logger.error(f"Failed to add priority feature: {e}")
        raise

# Load & prepare the data
logger.info("Loading and preparing data...")
df = load_cleaned_data(path="amine/data/cleaned_data.csv")
daily = aggregate_daily(df, by_type=False)
daily_by_type = aggregate_daily(df, by_type=True)

# Add priority feature for classification
logger.info("Adding priority feature...")
daily = add_priority_feature(daily, df)

# Load the classifier
logger.info("Loading classifier...")
with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

# Prepare classification data
logger.info("Preparing classification data...")
window_clf = 7
X_clf, y_clf = prepare_classification_data(daily, window=window_clf, original_df=df)

# Get dates for classification validation set
clf_dates = daily['expiry_date'].iloc[window_clf:].reset_index(drop=True)

# Log shapes to debug
logger.info(f"X_clf shape: {X_clf.shape}")
logger.info(f"y_clf shape: {y_clf.shape}")
logger.info(f"clf_dates length: {len(clf_dates)}")

# Ensure lengths match
if not (len(X_clf) == len(y_clf) == len(clf_dates)):
    logger.error(f"Length mismatch: X_clf ({len(X_clf)}), y_clf ({len(y_clf)}), clf_dates ({len(clf_dates)})")
    raise ValueError("Input arrays for train_test_split must have the same length")

# Split the data
logger.info("Splitting data for classifier...")
split_result = train_test_split(X_clf, y_clf, clf_dates, test_size=0.2, shuffle=False)
logger.info(f"train_test_split returned {len(split_result)} values: {[type(x) for x in split_result]}")

if len(split_result) != 6:
    logger.error("train_test_split did not return expected number of values (6)")
    raise ValueError("train_test_split must return 6 values for X, y, and dates")

_, X_val_clf, _, y_val_clf, _, clf_val_dates = split_result

logger.info("Making classification predictions...")
X_val_clf_reshaped = X_val_clf.reshape(X_val_clf.shape[0], -1)
# Get probabilities and handle single-class case
proba = classifier.predict_proba(X_val_clf_reshaped)
logger.info(f"Classifier predict_proba shape: {proba.shape}")

if proba.shape[1] == 1:
    logger.warning("Classifier only predicted one class. Assuming all predictions are zero.")
    zero_pred = np.zeros(proba.shape[0], dtype=int)
else:
    zero_pred_prob = proba[:, 1]  # Probability of class 1 (non-zero)
    threshold = 0.3  # Adjust this value to balance precision and recall
    zero_pred = (zero_pred_prob >= threshold).astype(int)

# Create DataFrame for classifier predictions with dates
clf_pred_df = pd.DataFrame({'date': clf_val_dates, 'is_nonzero_pred': zero_pred})

# Prepare regression data per type
logger.info("Preparing regression data per type...")
window_lstm = 7
X_dict, y_dict, scaler_dict = prepare_scaled_data(daily_by_type, window=window_lstm, by_type=True, original_df=df)
types = list(X_dict.keys())

# Predict per type
logger.info("Making LSTM predictions per type...")
predictions = []
for t in types:
    if t not in X_dict or X_dict[t] is None or X_dict[t].shape[0] == 0:
        logger.warning(f"Skipping type {t} due to insufficient data for LSTM.")
        continue

    X_scaled, y_scaled = X_dict[t], y_dict[t]
    scaler = scaler_dict[t]

    if scaler is None:
        logger.warning(f"Skipping type {t} due to missing scaler.")
        continue

    # Split
    X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = train_test_split(
        X_scaled, y_scaled, test_size=0.2, shuffle=False
    )

    # Get dates for LSTM validation set for this type
    daily_type_filtered = daily_by_type[daily_by_type['type'] == t].reset_index(drop=True)
    lstm_dates_type = daily_type_filtered['expiry_date'].iloc[window_lstm:].reset_index(drop=True)
    _, _, _, _, _, lstm_val_dates_type = train_test_split(
        X_scaled, y_scaled, lstm_dates_type, test_size=0.2, shuffle=False
    )

    # Load the model
    try:
        model = load_model(f"models/demand_model_{t}.h5", compile=False)
    except OSError:
        logger.error(f"Could not load model models/demand_model_{t}.h5. Skipping type {t}.")
        continue

    # Predict
    logger.info(f"Predicting for type: {t}")
    y_pred_scaled = model.predict(X_val_scaled).flatten()
    y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_val_original = scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

    # Ensure lengths match using the validation dates length
    min_len = len(lstm_val_dates_type)
    if len(y_val_original) != min_len or len(y_pred_original) != min_len:
        logger.warning(f"Length mismatch for type {t}. Actual: {len(y_val_original)}, Pred: {len(y_pred_original)}, Dates: {min_len}. Adjusting.")
        y_val_original = y_val_original[:min_len]
        y_pred_original = y_pred_original[:min_len]

    # Store predictions with correct dates
    pred_df_type = pd.DataFrame({
        "date": lstm_val_dates_type,
        "type": t,
        "actual": y_val_original,
        "predicted": y_pred_original
    })
    predictions.append(pred_df_type)

# Combine predictions
logger.info("Combining and aggregating predictions...")
if not predictions:
    logger.error("No predictions were generated. Exiting.")
    exit()

pred_combined = pd.concat(predictions)

# Aggregate LSTM predictions by date
pred_agg = pred_combined.groupby('date').agg({
    'actual': 'sum',
    'predicted': 'sum'
}).reset_index()

# Merge aggregated LSTM predictions with classifier predictions
pred_final = pd.merge(pred_agg, clf_pred_df, on='date', how='left')

# Fill missing classifier predictions
pred_final['is_nonzero_pred'] = pred_final['is_nonzero_pred'].fillna(1)
pred_final['predicted'] = pred_final['predicted'] * pred_final['is_nonzero_pred']

# Select final columns and save
final_output = pred_final[['date', 'actual', 'predicted']]
final_output.to_csv("ui/dash_app/predictions.csv", index=False)
logger.info("predictions.csv written to ui/dash_app/predictions.csv with combined predictions")