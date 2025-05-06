import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import pickle
import logging
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data
except ImportError as e:
    logger.error(f"Failed to import from ui.dash_app.use_data: {e}")
    logger.error("Ensure the script is run from the project root directory (SFWRE-G4) or adjust Python path.")
    exit()

def main():
    """Main function to load models and generate predictions."""
    try:
        window_size = 14
        logger.info(f"Using window size: {window_size}")
        data_path = "data/demand_prediction/cleaned_demand_data.csv"
        models_dir = "models"
        output_dir = "ui/dash_app"
        output_filename = "predictions.csv"
        output_path = os.path.join(output_dir, output_filename)

        logger.info("Loading and preparing data...")
        try:
            df = load_cleaned_data(path=data_path)
        except FileNotFoundError:
            logger.error(f"Data file not found at {data_path}. Please ensure the file exists.")
            return
        except Exception as load_e:
            logger.error(f"Error loading data: {load_e}")
            return

        daily_lstm_agg = aggregate_daily(df, by_type=True, by_location=True)

        logger.info("Preparing regression data per type and region...")
        X_dict, y_dict, scaler_dict, timestamps_dict = prepare_scaled_data(
            daily_lstm_agg, window=window_size, by_type=True
        )
        group_keys = list(X_dict.keys())

        if not group_keys:
            logger.error("No data prepared for LSTM models (X_dict is empty). Cannot proceed.")
            return

        lstm_predictions = []
        for key in group_keys:
            if isinstance(key, tuple) and len(key) == 2:
                food_type, region = key
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
            scaler_ref = scaler_dict[key]
            lstm_timestamps = timestamps_dict[key]

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

            filename_key = "_".join(map(str, key)).replace(" ", "_").replace("'", "").replace(",", "")
            model_path = os.path.join(models_dir, f"demand_model_{filename_key}.keras")
            scaler_path = os.path.join(models_dir, f"scaler_{filename_key}.pkl")

            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}. Skipping {key}.")
                continue
            if not os.path.exists(scaler_path):
                logger.error(f"Scaler file not found: {scaler_path}. Skipping {key}.")
                continue

            try:
                model = load_model(model_path, compile=False)
                with open(scaler_path, 'rb') as f:
                    scaler_lstm_target = pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading model or scaler for {key}: {e}")
                continue

            try:
                y_pred_scaled = model.predict(X_val_scaled).flatten()
            except Exception as lstm_pred_e:
                logger.error(f"Error during LSTM prediction for {key}: {lstm_pred_e}")
                continue

            try:
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
                "timestamp": lstm_val_timestamps,
                "type": food_type,
                "region": region,
                "actual": y_val_original,
                "predicted_lstm": y_pred_original
            })
            lstm_predictions.append(pred_df_type)
            logger.info(f"--- Finished LSTM prediction for group: {key} ---")

        if not lstm_predictions:
            logger.warning("No LSTM predictions were generated. Exiting.")
            return

        final_predictions_df = pd.concat(lstm_predictions, ignore_index=True)
        logger.info("Bypassing classifier (all demands non-zero). Setting is_nonzero_pred = 1.")
        final_df = final_predictions_df.copy()
        final_df['is_nonzero_pred'] = 1
        final_df['predicted'] = final_df['predicted_lstm']
        final_df = final_df[['timestamp', 'type', 'region', 'actual', 'predicted', 'predicted_lstm', 'is_nonzero_pred']]

        os.makedirs(output_dir, exist_ok=True)
        final_df.to_csv(output_path, index=False, float_format='%.2f')
        logger.info(f"Predictions saved successfully to {output_path}")

        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            def calculate_metrics(actual, predicted):
                mae = mean_absolute_error(actual, predicted)
                rmse = np.sqrt(mean_squared_error(actual, predicted))
                mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
                return mae, rmse, mape

            # Overall metrics
            actual_clean = final_df['actual'].fillna(0)
            predicted_clean = final_df['predicted'].fillna(0)
            mae, rmse, mape = calculate_metrics(actual_clean, predicted_clean)
            logger.info(f"Overall Validation Metrics:")
            logger.info(f"  MAE:  {mae:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAPE: {mape:.2f}%")

            # Group-wise metrics for bakery and vegetables
            for (food_type, region), group in final_df.groupby(['type', 'region']):
                if food_type in ['bakery', 'vegetables']:
                    mae, rmse, mape = calculate_metrics(group['actual'], group['predicted'])
                    logger.info(f"Metrics for {food_type}, {region}:")
                    logger.info(f"  MAE:  {mae:.4f}")
                    logger.info(f"  RMSE: {rmse:.4f}")
                    logger.info(f"  MAPE: {mape:.2f}%")
        except ImportError:
            logger.warning("Could not import sklearn.metrics. Skipping metrics calculation.")
        except Exception as metrics_e:
            logger.warning(f"Could not calculate overall metrics: {metrics_e}")

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()