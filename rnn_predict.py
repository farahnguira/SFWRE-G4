import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ui.dash_app.use_data import load_and_preprocess_data, create_sequences 
import os
import pickle

def predict_demand(file_path, model_path_h5, feature_scalers_path, target_scalers_path, window_size=28, lags_to_create=[1, 7]):
    # Load preprocessed data. 
    # X_test_global, y_test_scaled_global, y_test_orig_global are from the 20% split in load_and_preprocess_data
    # grouped_from_load is the full aggregated df, used for per-group detailed predictions
    # The lags_to_create parameter must match what was used during training (via use_data.py)
    (X_train_global, _, X_test_global, y_test_scaled_global, y_test_orig_global, 
     _, _, grouped_from_load) = load_and_preprocess_data(file_path, window_size, lags_to_create=lags_to_create)

    try:
        model = tf.keras.models.load_model(model_path_h5, custom_objects={'Huber': tf.keras.losses.Huber})
        with open(feature_scalers_path, "rb") as f:
            feature_scalers_dict = pickle.load(f)
        with open(target_scalers_path, "rb") as f:
            target_scalers_dict = pickle.load(f) # Load dict of target scalers
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        return None, np.nan, np.nan

    # --- Overall MAE/RMSE on the test split from load_and_preprocess_data ---
    # This is complex because y_test_orig_global is concatenated from different groups,
    # and predictions need to be inverse_transformed using per-group scalers.
    # For simplicity in this iteration, we'll focus on per-group MAE/RMSE below.
    print("Skipping overall MAE/RMSE on pre-split test set in this version due to per-group scaling complexity for inverse transform.")
    overall_mae, overall_rmse = np.nan, np.nan


    # --- Per-group predictions and MAE/RMSE ---
    predictions_list = []
    
    lag_feature_names = [f'demand_kg_lag_{lag}' for lag in lags_to_create]
    base_numerical_features = ['day_of_week', 'month']
    numerical_features = base_numerical_features + lag_feature_names
    binary_features = ['is_weekend', 'holiday_flag', 'promotion_flag', 'disaster_flag', 'is_region_b_shortage']
    feature_columns_for_x = numerical_features + binary_features

    for (region, food_type), group_df_orig_full in grouped_from_load.groupby(['region', 'food_type']):
        group_df_orig_full = group_df_orig_full.sort_values('timestamp').reset_index(drop=True)
        
        # Drop rows with NaNs from lagging for this specific group
        # This ensures that the data used for prediction has the same preprocessing (dropna for lags)
        # as the data used for training the scalers and the model.
        group_df_for_pred = group_df_orig_full.dropna().reset_index(drop=True)

        if len(group_df_for_pred) <= window_size:
            print(f"Skipping group ({region}, {food_type}) for detailed prediction: too short after lagging/dropna ({len(group_df_for_pred)} rows).")
            continue

        group_df_scaled_features = group_df_for_pred.copy()
        
        current_feature_scalers = feature_scalers_dict.get((region, food_type))
        current_target_scaler = target_scalers_dict.get((region, food_type))

        if not current_feature_scalers or not current_target_scaler:
            print(f"Warning: Scalers not found for group ({region}, {food_type}). Skipping prediction for this group.")
            continue
        
        for col in numerical_features: 
            if col in current_feature_scalers:
                scaler = current_feature_scalers[col]
                group_df_scaled_features[col] = scaler.transform(group_df_for_pred[[col]])
            else: 
                print(f"Warning: Feature scaler for '{col}' not found in group ({region}, {food_type}). Using original values. This may lead to inaccurate predictions.")
        
        X_group_seq, y_group_actual_orig_seq = create_sequences(
            group_df_scaled_features, 'demand_kg', 
            window_size, feature_columns_for_x
        )

        if X_group_seq.shape[0] == 0:
            print(f"Skipping group ({region}, {food_type}): no sequences generated for prediction.")
            continue
            
        y_group_pred_scaled_seq = model.predict(X_group_seq, verbose=0)
        y_group_pred_orig_seq = current_target_scaler.inverse_transform(y_group_pred_scaled_seq).flatten()
        
        timestamps_for_pred = group_df_for_pred['timestamp'].iloc[window_size:window_size+len(y_group_pred_orig_seq)].values

        min_len = min(len(timestamps_for_pred), len(y_group_pred_orig_seq), len(y_group_actual_orig_seq))
        if min_len < len(y_group_pred_orig_seq) or min_len < len(y_group_actual_orig_seq):
            print(f"Warning: Aligning sequence lengths for group ({region}, {food_type}) from {len(y_group_pred_orig_seq)}/{len(y_group_actual_orig_seq)} to {min_len}.")
        
        timestamps_for_pred = timestamps_for_pred[:min_len]
        y_group_pred_orig_seq = y_group_pred_orig_seq[:min_len]
        y_group_actual_orig_seq = y_group_actual_orig_seq[:min_len]
        
        if min_len == 0:
            print(f"Skipping group ({region}, {food_type}): zero length sequences after alignment.")
            continue

        group_mae = mean_absolute_error(y_group_actual_orig_seq, y_group_pred_orig_seq)
        group_rmse = np.sqrt(mean_squared_error(y_group_actual_orig_seq, y_group_pred_orig_seq))
        print(f"{region} - {food_type}: MAE {group_mae:.2f}, RMSE {group_rmse:.2f} (on {min_len} sequences)")
        
        for i in range(len(y_group_pred_orig_seq)):
            predictions_list.append({
                'timestamp': timestamps_for_pred[i],
                'region': region,
                'food_type': food_type,
                'actual': y_group_actual_orig_seq[i],
                'predicted': y_group_pred_orig_seq[i],
                'predicted_lstm': y_group_pred_orig_seq[i], 
                'error': abs(y_group_actual_orig_seq[i] - y_group_pred_orig_seq[i]),
                'is_nonzero_pred': int(y_group_pred_orig_seq[i] > 1e-3)
            })
    
    pred_df = pd.DataFrame(predictions_list)
    
    final_output_dir = "data/demand_prediction" 
    os.makedirs(final_output_dir, exist_ok=True)
    final_output_path = os.path.join(final_output_dir, "predictions_rnn_script.csv")
    
    if not pred_df.empty:
        pred_df.to_csv(final_output_path, index=False, float_format='%.2f')
        print(f"Predictions saved to {final_output_path}")
    else:
        print("No predictions generated to save.")
        
    return pred_df, overall_mae, overall_rmse 

if __name__ == "__main__":
    # This script should ideally be run from the root of the SFWRE-G4 project
    data_file_path = "data/demand_prediction/cleaned_demand_data.csv"
    model_h5_path = "models/lstm_demand_model.h5"
    f_scalers_path = "models/feature_scalers_dict.pkl"
    t_scalers_path = "models/target_scalers_dict.pkl" 

    # Define final_output_path here as well for the print statement
    final_output_dir_main = "data/demand_prediction"
    final_output_filename_main = "predictions_rnn_script.csv"
    final_output_path_main = os.path.join(final_output_dir_main, final_output_filename_main)

    required_files = [data_file_path, model_h5_path, f_scalers_path, t_scalers_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("One or more required files not found:")
        for f in missing_files:
            print(f" - {f}")
        print("Please ensure all files exist. You might need to run the training script (rnn_train.py) first.")
    else:
        # Pass the lags_to_create to predict_demand, ensuring it matches training
        pred_df, _, _ = predict_demand(data_file_path, model_h5_path, f_scalers_path, t_scalers_path, lags_to_create=[1, 7])
        if pred_df is not None and not pred_df.empty:
            print(f"\nSample predictions (from {os.path.basename(final_output_path_main)}):")
            print(pred_df.head())
        elif pred_df is not None and pred_df.empty:
            print("Prediction DataFrame is empty (no predictions were made for any group).")
        else:
            print("Prediction failed or returned None.")