import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data, add_temporal_features

def add_priority_feature(daily, original_df):
    """Add the priority feature to the daily DataFrame for classification."""
    try:
        # Ensure donation_date is datetime in original_df
        original_df['donation_date'] = pd.to_datetime(original_df['donation_date'], errors='coerce')
        
        # Group by donation_date and location
        priority_by_date = original_df.groupby([original_df['donation_date'].dt.date, 
                                              'location'])['priority'].mean().reset_index()
        priority_by_date['donation_date'] = pd.to_datetime(priority_by_date['donation_date'])
        
        # Log column types and shapes
        logger.info(f"daily dtypes: {daily.dtypes}")
        logger.info(f"priority_by_date dtypes: {priority_by_date.dtypes}")
        logger.info(f"daily shape before merge: {daily.shape}")
        logger.info(f"priority_by_date shape: {priority_by_date.shape}")
        
        # Merge on donation_date and location
        daily = daily.merge(
            priority_by_date,
            left_on=['donation_date', 'location'],
            right_on=['donation_date', 'location'],
            how='left'
        )
        
        # Log shape after merge
        logger.info(f"daily shape after merge: {daily.shape}")
        
        # Check if priority column exists
        if 'priority' not in daily.columns:
            logger.warning("Priority column not added after merge. Creating default priority column.")
            daily['priority'] = 1.0  # Default priority value
        else:
            # Fill NaNs in priority
            daily['priority'] = daily['priority'].fillna(daily['priority'].mean())
            if daily['priority'].isnull().all():
                logger.warning("All priority values are NaN. Setting default priority to 1.0.")
                daily['priority'] = 1.0
        
        logger.info("Priority feature added successfully.")
        logger.info(f"daily columns after merge: {daily.columns.tolist()}")
        return daily
    except Exception as e:
        logger.error(f"Failed to add priority feature: {e}")
        raise

logger.info("Loading and preparing data...")
df = load_cleaned_data(path="amine/data/cleaned_data.csv")
daily = aggregate_daily(df, by_type=False)
daily_by_type = aggregate_daily(df, by_type=True)

logger.info("Adding priority feature...")
daily = add_priority_feature(daily, df)

logger.info("Loading classifier...")
with open('models/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

logger.info("Preparing classification data...")
window_clf = 7
X_clf, y_clf, clf_dates = prepare_classification_data(daily, window=window_clf, original_df=df)

logger.info(f"X_clf shape: {X_clf.shape}")
logger.info(f"y_clf shape: {y_clf.shape}")
logger.info(f"clf_dates length: {len(clf_dates)}")

if X_clf.size == 0 or y_clf.size == 0 or clf_dates.empty:
    logger.error("prepare_classification_data returned empty arrays/Series.")
    exit()

if not (len(X_clf) == len(y_clf) == len(clf_dates)):
    logger.error(f"Length mismatch: X_clf ({len(X_clf)}), y_clf ({len(y_clf)}), clf_dates ({len(clf_dates)})")
    raise ValueError("Input arrays for train_test_split must have the same length")

logger.info("Splitting data for classifier...")
X_train_clf, X_val_clf, y_train_clf, y_val_clf, clf_train_dates, clf_val_dates = train_test_split(
    X_clf, y_clf, clf_dates, test_size=0.2, shuffle=False
)

logger.info("Making classification predictions...")
X_val_clf_reshaped = X_val_clf.reshape(X_val_clf.shape[0], -1)
proba = classifier.predict_proba(X_val_clf_reshaped)
logger.info(f"Classifier predict_proba shape: {proba.shape}")

if proba.shape[1] == 1:
    logger.warning("Classifier only predicted one class. Assuming all predictions are zero.")
    zero_pred = np.zeros(proba.shape[0], dtype=int)
else:
    zero_pred_prob = proba[:, 1]
    threshold = 0.4
    zero_pred = (zero_pred_prob >= threshold).astype(int)

clf_pred_df = pd.DataFrame({'date': clf_val_dates, 'is_nonzero_pred': zero_pred})

logger.info("Preparing regression data per type and location...")
window_lstm = 7
X_dict, y_dict, scaler_dict, dates_dict = prepare_scaled_data(daily_by_type, window=window_lstm, 
                                                            by_type=True, original_df=df)
keys = list(X_dict.keys())
logger.info(f"Available keys: {keys}")

# Check for model files
missing_models = []
for key in keys:
    model_path = f"models/demand_model_{key}.keras"
    if not os.path.exists(model_path):
        missing_models.append(model_path)
if missing_models:
    logger.warning(f"Missing model files: {missing_models}")

predictions = []
for key in keys:
    if key not in X_dict or key not in y_dict or key not in scaler_dict or key not in dates_dict:
        logger.warning(f"Skipping {key} due to missing data.")
        continue
    if X_dict[key] is None or X_dict[key].shape[0] == 0:
        logger.warning(f"Skipping {key} due to insufficient data.")
        continue
    
    X_scaled, y_scaled = X_dict[key], y_dict[key]
    scaler = scaler_dict[key]
    lstm_dates = dates_dict[key]
    
    logger.info(f"Processing {key}: X_scaled shape {X_scaled.shape}, y_scaled shape {y_scaled.shape}, lstm_dates length {len(lstm_dates)}")
    
    if scaler is None:
        logger.warning(f"Skipping {key} due to missing scaler.")
        continue
    
    if len(X_scaled) < 5:
        logger.warning(f"Skipping {key}: Insufficient samples ({len(X_scaled)}).")
        continue
    
    if not (len(X_scaled) == len(y_scaled) == len(lstm_dates)):
        logger.error(f"Length mismatch for {key}: X({len(X_scaled)}), y({len(y_scaled)}), dates({len(lstm_dates)})")
        continue
    
    # Split data for validation
    X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled, lstm_train_dates, lstm_val_dates = train_test_split(
        X_scaled, y_scaled, lstm_dates, test_size=0.2, shuffle=False
    )
    
    model_path = f"models/demand_model_{key}.keras"
    logger.info(f"Loading model for {key} from {model_path}")
    try:
        model = load_model(model_path, compile=False)
    except Exception as e:
        logger.error(f"Could not load model {model_path} for {key}: {e}")
        continue
    
    logger.info(f"Predicting for {key}")
    try:
        y_pred_scaled = model.predict(X_val_scaled, verbose=0).flatten()
        y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_val_original = scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
        
        min_len = min(len(y_val_original), len(y_pred_original), len(lstm_val_dates))
        if len(y_val_original) != min_len or len(y_pred_original) != min_len:
            logger.warning(f"Length mismatch for {key}. Actual: {len(y_val_original)}, Pred: {len(y_pred_original)}, Dates: {len(lstm_val_dates)}.")
            y_val_original = y_val_original[:min_len]
            y_pred_original = y_pred_original[:min_len]
            lstm_val_dates = lstm_val_dates[:min_len]
        
        food_type, location = key.split('_', 1)
        pred_df = pd.DataFrame({
            "date": lstm_val_dates,
            "type": food_type,
            "location": location,
            "actual": y_val_original,
            "predicted": y_pred_original
        })
        predictions.append(pred_df)
        logger.info(f"Predictions generated for {key}: {len(pred_df)} rows")
    except Exception as e:
        logger.error(f"Prediction failed for {key}: {e}")
        continue

logger.info(f"Total predictions generated: {len(predictions)} DataFrames")
if not predictions:
    logger.error("No predictions were generated. Check model files and data.")
    exit()

logger.info("Combining and aggregating predictions...")
pred_combined = pd.concat(predictions, ignore_index=True)
logger.info(f"Combined predictions shape: {pred_combined.shape}")

pred_agg = pred_combined.groupby(['date', 'location']).agg({
    'actual': 'sum',
    'predicted': 'sum',
    'type': lambda x: list(x)
}).reset_index()

pred_final = pd.merge(pred_agg, clf_pred_df, on='date', how='left')

pred_final['is_nonzero_pred'] = pred_final['is_nonzero_pred'].fillna(1)
pred_final['predicted'] = pred_final['predicted'] * pred_final['is_nonzero_pred']

final_output = pred_final[['date', 'location', 'type', 'actual', 'predicted']]

# Ensure output directory exists
os.makedirs("ui/dash_app", exist_ok=True)
final_output.to_csv("ui/dash_app/predictions.csv", index=False)
logger.info("predictions.csv written to ui/dash_app/predictions.csv")