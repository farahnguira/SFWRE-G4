import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle
import logging
import os
import json # Import json to load threshold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Import functions from use_data (updated) ---
try:
    # Import updated functions from use_data
    from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data # Removed add_priority_feature, add_temporal_features
except ImportError as e:
    logger.error(f"Failed to import from ui.dash_app.use_data: {e}")
    logger.error("Ensure the script is run from the project root directory (SFWRE-G4) or adjust Python path.")
    exit()
# --- End Imports ---


# --- Removed add_priority_feature definition from here ---


def main():
    """Main function to load models and generate predictions."""
    try:
        # --- Setup ---
        window_size = 14
        logger.info(f"Using window size: {window_size}")
        data_path = "data/demand_prediction/cleaned_demand_data.csv" # Updated data path
        models_dir = "models"
        output_dir = "ui/dash_app"
        output_filename = "predictions.csv"
        output_path = os.path.join(output_dir, output_filename)
        classifier_threshold_path = os.path.join(models_dir, "classifier_threshold.json") # Path to threshold file
        # --- End Setup ---

        logger.info("Loading and preparing data...")
        try:
            df = load_cleaned_data(path=data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found at {data_path}. Please ensure the file exists.")
            return
        except Exception as load_e:
            logger.error(f"Error loading data: {load_e}")
            return

        # Aggregate for classifier (by timestamp and region)
        daily_clf_agg = aggregate_daily(df, by_type=False, by_location=True)
        # Aggregate for LSTM (by timestamp, type and region)
        daily_lstm_agg = aggregate_daily(df, by_type=True, by_location=True)

        # --- Removed priority feature addition ---

        # --- Load Classifier and Threshold ---
        logger.info("Loading classifier and threshold...")
        classifier_path = os.path.join(models_dir, 'classifier.pkl')
        if not os.path.exists(classifier_path):
            logger.error(f"Classifier model '{classifier_path}' not found. Please run rnn_train.py first.")
            return
        if not os.path.exists(classifier_threshold_path):
            logger.error(f"Classifier threshold file '{classifier_threshold_path}' not found. Please run rnn_train.py first.")
            return

        try:
            with open(classifier_path, 'rb') as f:
                classifier = pickle.load(f)
            with open(classifier_threshold_path, 'r') as f:
                threshold_data = json.load(f)
                threshold = threshold_data.get('optimal_threshold', 0.5) # Default to 0.5 if not found
                logger.info(f"Loaded optimal classifier threshold: {threshold:.2f}")
        except Exception as e:
            logger.error(f"Error loading classifier or threshold: {e}")
            return

        # --- Prepare Data and Predict with Classifier ---
        logger.info("Preparing classification data...")
        # Call prepare_classification_data without original_df
        X_clf, y_clf, clf_timestamps = prepare_classification_data(daily_clf_agg, window=window_size)

        if X_clf.size == 0 or y_clf.size == 0 or clf_timestamps.empty:
            logger.error("prepare_classification_data returned empty arrays/Series. Cannot proceed.")
            return

        logger.info("Splitting data for classifier validation...")
        if len(X_clf) < 5:
             logger.error(f"Not enough classification samples ({len(X_clf)}) for train/test split.")
             return
        _, X_val_clf, _, y_val_clf, _, clf_val_timestamps = train_test_split(
            X_clf, y_clf, clf_timestamps, test_size=0.2, shuffle=False
        )

        logger.info("Making classification predictions...")
        n_samples_val, window_size_clf, n_features_clf = X_val_clf.shape
        X_val_clf_reshaped = X_val_clf.reshape((n_samples_val, window_size_clf * n_features_clf))

        try:
            expected_features = classifier.n_features_in_
            if X_val_clf_reshaped.shape[1] != expected_features:
                logger.error(f"Feature mismatch for classifier: Input has {X_val_clf_reshaped.shape[1]}, model expects {expected_features}.")
                logger.error("Retrain required.")
                return
        except AttributeError:
             logger.warning("Could not check classifier's n_features_in_. Proceeding.")
        except Exception as feat_check_e:
             logger.error(f"Error checking classifier features: {feat_check_e}")
             return

        try:
            proba = classifier.predict_proba(X_val_clf_reshaped)
            logger.info(f"Classifier predict_proba shape: {proba.shape}")
        except Exception as clf_pred_e:
            logger.error(f"Error during classifier prediction: {clf_pred_e}")
            return

        if proba.shape[1] == 1:
            logger.warning("Classifier only predicted one class. Assuming non-zero (class 1).")
            zero_pred = np.ones(proba.shape[0], dtype=int)
        else:
            zero_pred_prob = proba[:, 1]
            # Use the loaded optimal threshold
            zero_pred = (zero_pred_prob >= threshold).astype(int)

        # Use timestamp instead of date
        clf_pred_df = pd.DataFrame({'timestamp': clf_val_timestamps, 'is_nonzero_pred': zero_pred})
        # No need to convert timestamp to date

        # --- Prepare Data and Predict with LSTM Models ---
        logger.info("Preparing regression data per type and region...")
        # Call prepare_scaled_data without original_df
        X_dict, y_dict, scaler_dict, timestamps_dict = prepare_scaled_data(
            daily_lstm_agg, window=window_size, by_type=True
        )
        group_keys = list(X_dict.keys())

        if not group_keys:
            logger.error("No data prepared for LSTM models (X_dict is empty). Cannot proceed.")
            return

        lstm_predictions = []
        for key in group_keys:
            # Key is now potentially (food_type, region)
            if isinstance(key, tuple) and len(key) == 2:
                 food_type, region = key # Use region
            else:
                 logger.warning(f"Skipping unexpected key format: {key}")
                 continue

            logger.info(f"--- Processing LSTM for group: {key} ---")

            if key not in X_dict or key not in y_dict or key not in scaler_dict or key not in timestamps_dict:
                logger.warning(f"Skipping {key} due to missing data in dictionaries.")
                continue
            if X_dict[key] is None or X_dict[key].shape[0] == 0:
                logger.warning(f"Skipping {key} due to insufficient data in X_dict.")
                continue

            X_scaled, y_scaled = X_dict[key], y_dict[key]
            scaler_ref = scaler_dict[key] # Scaler for features+target used during prep
            lstm_timestamps = timestamps_dict[key] # Use timestamps

            if scaler_ref is None:
                 logger.warning(f"Skipping {key} due to missing scaler reference.")
                 continue

            if len(X_scaled) < 5:
                logger.warning(f"Skipping {key}: Insufficient samples ({len(X_scaled)}) for LSTM split.")
                continue

            if not (len(X_scaled) == len(y_scaled) == len(lstm_timestamps)):
                logger.error(f"Length mismatch for {key} BEFORE split: X({len(X_scaled)}), y({len(y_scaled)}), timestamps({len(lstm_timestamps)})")
                continue

            logger.info(f"Splitting LSTM data for group: {key}")
            _, X_val_scaled, _, y_val_scaled, _, lstm_val_timestamps = train_test_split(
                X_scaled, y_scaled, lstm_timestamps, test_size=0.2, shuffle=False
            )

            if X_val_scaled.shape[0] == 0:
                 logger.warning(f"Skipping {key}: Validation set empty after split.")
                 continue

            # Ensure key components are strings for filename
            filename_key = "_".join(map(str, key)).replace(" ", "_").replace("'", "").replace(",", "")
            model_path = os.path.join(models_dir, f"demand_model_{filename_key}.keras")
            scaler_path = os.path.join(models_dir, f"scaler_{filename_key}.pkl") # Target scaler path

            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}. Skipping {key}.")
                continue
            if not os.path.exists(scaler_path):
                 logger.error(f"Scaler file not found: {scaler_path}. Skipping {key}.")
                 continue

            try:
                model = load_model(model_path, compile=False)
                with open(scaler_path, 'rb') as f:
                    scaler_lstm_target = pickle.load(f) # Load the target scaler
            except Exception as e:
                logger.error(f"Error loading model or scaler for {key}: {e}")
                continue

            try:
                y_pred_scaled = model.predict(X_val_scaled).flatten()
            except Exception as lstm_pred_e:
                 logger.error(f"Error during LSTM prediction for {key}: {lstm_pred_e}")
                 continue

            try:
                # Use the loaded target scaler for inverse transform
                y_pred_original = scaler_lstm_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
                y_val_original = scaler_lstm_target.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
            except Exception as scale_e:
                 logger.error(f"Error during inverse transform for {key}: {scale_e}")
                 continue

            min_len = len(lstm_val_timestamps)
            if len(y_val_original) != min_len or len(y_pred_original) != min_len:
                logger.warning(f"Length mismatch for {key} after prediction. Adjusting.")
                y_val_original = y_val_original[:min_len]
                y_pred_original = y_pred_original[:min_len]

            y_pred_original = np.maximum(0, y_pred_original)

            pred_df_type = pd.DataFrame({
                "timestamp": lstm_val_timestamps, # Use timestamp
                "type": food_type,
                "region": region, # Use region
                "actual": y_val_original,
                "predicted_lstm": y_pred_original
            })
            lstm_predictions.append(pred_df_type)
            logger.info(f"--- Finished LSTM prediction for group: {key} ---")

        # --- Combine and Save Results ---
        if not lstm_predictions:
            logger.warning("No LSTM predictions were generated. Exiting.")
            return

        final_predictions_df = pd.concat(lstm_predictions, ignore_index=True)
        # No need to convert timestamp to date

        # Merge on timestamp and region (classifier predictions are per region)
        # Need to ensure clf_pred_df has region if classifier was trained on region-specific data
        # If classifier is global (trained on aggregate without region), merge only on timestamp
        # Assuming classifier was trained on data aggregated by region (as per train script)
        # We need to add region to clf_pred_df. This requires knowing which region each clf prediction corresponds to.
        # This is complex because prepare_classification_data currently aggregates globally.
        # --- Simplification: Apply the *same* classifier prediction to *all* regions for a given timestamp ---
        # This is a limitation if the zero-demand pattern varies significantly by region.
        # A better approach would be to train region-specific classifiers or include region in the global classifier features.
        # For now, merging only on timestamp.
        logger.warning("Applying global classifier prediction to all regions for each timestamp.")
        final_df = pd.merge(final_predictions_df, clf_pred_df, on='timestamp', how='left')

        final_df['is_nonzero_pred'] = final_df['is_nonzero_pred'].fillna(1).astype(int) # Default to non-zero if no match
        final_df['predicted'] = final_df['predicted_lstm'] * final_df['is_nonzero_pred']
        # Update columns to reflect timestamp and region
        final_df = final_df[['timestamp', 'type', 'region', 'actual', 'predicted', 'predicted_lstm', 'is_nonzero_pred']]

        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(output_path, index=False, float_format='%.2f')
        logger.info(f"Predictions saved successfully to {output_path}")

        # --- Optional: Calculate and Log Overall Metrics ---
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            actual_clean = final_df['actual'].fillna(0)
            predicted_clean = final_df['predicted'].fillna(0)
            mae = mean_absolute_error(actual_clean, predicted_clean)
            rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
            try:
                r2 = r2_score(actual_clean, predicted_clean)
            except ValueError:
                r2 = np.nan
            logger.info(f"Overall Validation Metrics:")
            logger.info(f"  MAE:  {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²:   {r2:.4f}")
        except ImportError:
             logger.warning("Could not import sklearn.metrics. Skipping metrics calculation.")
        except Exception as metrics_e:
            logger.warning(f"Could not calculate overall metrics: {metrics_e}")

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()