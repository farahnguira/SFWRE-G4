import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import os

logger = logging.getLogger(__name__)

def load_cleaned_data(path="data/demand_prediction/cleaned_demand_data.csv"):
    """Loads the cleaned data from the specified path."""
    try:
        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}. Shape: {df.shape}")
        # Ensure timestamp is treated as numeric for calculations
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']) # Drop rows where timestamp couldn't be converted
        df['timestamp'] = df['timestamp'].astype(int)
        return df
    except FileNotFoundError:
        logger.error(f"Error: The file '{path}' was not found.")
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading the data: {e}")
        raise

def aggregate_daily(df, by_type=False, by_location=False):
    """Aggregates demand by timestamp, optionally by type and location."""
    try:
        if df is None or df.empty:
            logger.warning("Input DataFrame is empty or None. Cannot aggregate.")
            return pd.DataFrame()

        group_cols = ['timestamp']
        if by_type:
            group_cols.append('food_type')
        if by_location:
            group_cols.append('region')

        # Find timestamps where Region_B had a disaster_flag=1 in the original data BEFORE aggregation
        shortage_timestamps = set(df[(df['region'] == 'Region_B') & (df['disaster_flag'] == 1)]['timestamp'].unique())
        logger.info(f"Identified {len(shortage_timestamps)} timestamps with Region_B shortages.")

        # Aggregate demand
        df_agg = df.groupby(group_cols)['demand_kg'].sum().reset_index()

        # Create the new global shortage feature in the aggregated dataframe
        df_agg['region_b_shortage_flag'] = df_agg['timestamp'].apply(lambda ts: 1 if ts in shortage_timestamps else 0).astype(int)

        # Rename aggregated column for clarity
        df_agg = df_agg.rename(columns={'demand_kg': 'demand_kg'}) # Keep name consistent for now

        logger.info(f"Data aggregated. Grouped by: {group_cols}. Shape: {df_agg.shape}")
        logger.info(f"Aggregated data includes 'region_b_shortage_flag'. Sum: {df_agg['region_b_shortage_flag'].sum()}")
        return df_agg

    except Exception as e:
        logger.error(f"An error occurred during daily aggregation: {e}", exc_info=True)
        return pd.DataFrame()


def _prepare_scaled_data_single(daily, window):
    """Prepares features, scales data, and creates sequences for one group (LSTM)."""
    if daily.empty or len(daily) <= window:
        logger.warning("Not enough data for scaling or sequence creation.")
        return None, None, None, None, None # Return Nones matching the expected output tuple

    daily_features = daily.copy()
    target_col = 'demand_kg'

    # Ensure timestamp is int for modulo operation
    daily_features['timestamp'] = daily_features['timestamp'].astype(int)

    # Add temporal features
    daily_features['day_of_week'] = (daily_features['timestamp'] % 7).astype(int)  # 0 to 6
    daily_features['is_weekend'] = daily_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0) # Assuming 5=Sat, 6=Sun

    # --- Removed is_holiday proxy based on disaster_flag ---
    # Use a placeholder for actual holidays if needed, otherwise set to 0
    daily_features['is_holiday'] = 0

    # --- Removed shortage_region_interaction ---
    # This is now handled by including 'region_b_shortage_flag' for all regions

    # Lag Features - Use demand_kg directly as quantity
    lags = [1, 2, 3, 7, 14, 28]
    for lag in lags:
        daily_features[f'quantity_lag_{lag}'] = daily_features[target_col].shift(lag)

    # Rolling Features - Use demand_kg directly as quantity
    roll_windows = [3, 7, 14, 28]
    for rw in roll_windows:
        rolling_mean = daily_features[target_col].rolling(window=rw, min_periods=1).mean()
        rolling_std = daily_features[target_col].rolling(window=rw, min_periods=1).std()
        rolling_median = daily_features[target_col].rolling(window=rw, min_periods=1).median()
        daily_features[f'quantity_roll_mean_{rw}'] = rolling_mean.shift(1) # Shift to prevent data leakage
        daily_features[f'quantity_roll_std_{rw}'] = rolling_std.shift(1)
        daily_features[f'quantity_roll_median_{rw}'] = rolling_median.shift(1)

    # Define Feature Columns - including the new global shortage flag
    feature_cols = [
        'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3', 'quantity_lag_7', 'quantity_lag_14', 'quantity_lag_28',
        'quantity_roll_mean_3', 'quantity_roll_mean_7', 'quantity_roll_mean_14', 'quantity_roll_mean_28',
        'quantity_roll_std_3', 'quantity_roll_std_7', 'quantity_roll_std_14', 'quantity_roll_std_28',
        'quantity_roll_median_3', 'quantity_roll_median_7', 'quantity_roll_median_14', 'quantity_roll_median_28',
        'day_of_week', 'is_weekend', 'is_holiday',
        'region_b_shortage_flag', # Added new global flag
        'timestamp' # Keep timestamp for potential analysis, but usually exclude from scaling/model input features
    ]

    # Ensure all defined feature columns actually exist, fill NaNs created by shifts/rolls
    existing_feature_cols = [col for col in daily_features.columns if col in feature_cols and col != 'timestamp']
    daily_features = daily_features.fillna(0) # Simple fillna strategy, consider more sophisticated methods if needed

    if daily_features.empty or len(daily_features) <= window:
        logger.warning("Data became empty or too short after feature engineering.")
        return None, None, None, None, None

    # Scaling
    if not existing_feature_cols:
        logger.warning("No valid feature columns found for scaling.")
        return None, None, None, None, None

    scaler_features = StandardScaler()
    # Scale only the feature columns, exclude timestamp
    scaled_features = scaler_features.fit_transform(daily_features[existing_feature_cols]).astype(np.float32)

    scaler_target = StandardScaler()
    scaled_target = scaler_target.fit_transform(daily_features[[target_col]]).astype(np.float32)

    # Create sequences
    X, y, timestamps_seq = [], [], []
    for i in range(window, len(scaled_features)):
        X.append(scaled_features[i-window:i])
        y.append(scaled_target[i])
        timestamps_seq.append(daily_features['timestamp'].iloc[i]) # Store corresponding timestamp for the prediction

    if not X:
        logger.warning("No sequences created.")
        return None, None, None, None, None

    X = np.array(X)
    y = np.array(y)
    timestamps_seq = np.array(timestamps_seq)

    logger.info(f"Prepared scaled data. X shape: {X.shape}, y shape: {y.shape}")
    return X, y, scaler_target, timestamps_seq, scaler_features # Return feature scaler as well


def prepare_scaled_data(daily_data, window=14, by_type=False):
    """Prepares scaled data for LSTM, potentially grouped by type and region."""
    X_dict, y_dict, scaler_dict, timestamps_dict = {}, {}, {}, {}
    scaler_features_dict = {} # Store feature scalers

    if daily_data is None or daily_data.empty:
        logger.error("Input daily_data is empty or None.")
        return X_dict, y_dict, scaler_dict, timestamps_dict # Return empty dicts

    group_cols = []
    if by_type and 'food_type' in daily_data.columns:
        group_cols.append('food_type')
    if 'region' in daily_data.columns: # Assuming region always exists after aggregation if by_location=True
        group_cols.append('region')

    if group_cols:
        grouped = daily_data.groupby(group_cols)
        logger.info(f"Grouping data by {group_cols} for scaling.")
        for name, group in grouped:
            logger.info(f"Processing group: {name}")
            X, y, scaler_target, timestamps_seq, scaler_features = _prepare_scaled_data_single(group.copy(), window)
            if X is not None and y is not None:
                X_dict[name] = X
                y_dict[name] = y
                scaler_dict[name] = scaler_target
                timestamps_dict[name] = timestamps_seq
                scaler_features_dict[name] = scaler_features # Store feature scaler
            else:
                logger.warning(f"Skipping group {name} due to insufficient data or errors.")
    else:
        logger.info("Processing data without grouping.")
        X, y, scaler_target, timestamps_seq, scaler_features = _prepare_scaled_data_single(daily_data.copy(), window)
        if X is not None and y is not None:
            X_dict['all'] = X
            y_dict['all'] = y
            scaler_dict['all'] = scaler_target
            timestamps_dict['all'] = timestamps_seq
            scaler_features_dict['all'] = scaler_features # Store feature scaler
        else:
            logger.warning("Skipping data processing due to insufficient data or errors.")

    # Return feature scalers dictionary as well, might be useful later
    return X_dict, y_dict, scaler_dict, timestamps_dict # scaler_features_dict could be returned if needed


def prepare_classification_data(daily, window=14):
    """Prepares features and target for the zero-demand classification task."""
    if daily is None or daily.empty or len(daily) <= window:
        logger.warning("Not enough data for classification preparation.")
        return None, None, None, None

    daily_features = daily.copy()
    target_col = 'demand_kg'

    # Create binary target: 1 if demand > 0, else 0
    daily_features['is_nonzero_demand'] = (daily_features[target_col] > 0).astype(int)
    y_class = daily_features['is_nonzero_demand'].values

    # --- Feature Engineering (Similar to LSTM, but potentially simpler) ---
    daily_features['timestamp'] = daily_features['timestamp'].astype(int)
    daily_features['day_of_week'] = (daily_features['timestamp'] % 7).astype(int)
    daily_features['is_weekend'] = daily_features['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    # Include the global shortage flag if it exists
    if 'region_b_shortage_flag' in daily_features.columns:
        daily_features['region_b_shortage_flag'] = daily_features['region_b_shortage_flag']
    else:
        daily_features['region_b_shortage_flag'] = 0 # Default if not present

    # Lag Features for the target itself (demand_kg)
    lags = [1, 7, 14]
    for lag in lags:
        daily_features[f'quantity_lag_{lag}'] = daily_features[target_col].shift(lag)

    # Rolling Features for the target
    roll_windows = [7, 14]
    for rw in roll_windows:
        rolling_mean = daily_features[target_col].rolling(window=rw, min_periods=1).mean()
        daily_features[f'quantity_roll_mean_{rw}'] = rolling_mean.shift(1)

    # Define Feature Columns for Classifier
    feature_cols_class = [
        'quantity_lag_1', 'quantity_lag_7', 'quantity_lag_14',
        'quantity_roll_mean_7', 'quantity_roll_mean_14',
        'day_of_week', 'is_weekend',
        'region_b_shortage_flag' # Include the global shortage flag
    ]

    existing_feature_cols_class = [col for col in daily_features.columns if col in feature_cols_class]
    daily_features = daily_features.fillna(0) # Fill NaNs

    if not existing_feature_cols_class:
         logger.warning("No valid feature columns found for classification.")
         return None, None, None, None

    # No scaling needed for RandomForest usually, but create sequences
    X_class_seq, y_class_seq, timestamps_class_seq = [], [], []
    features_array = daily_features[existing_feature_cols_class].values

    for i in range(window, len(features_array)):
        # Reshape window for compatibility if needed, or flatten
        X_class_seq.append(features_array[i-window:i].flatten()) # Flatten for RF
        y_class_seq.append(y_class[i])
        timestamps_class_seq.append(daily_features['timestamp'].iloc[i])

    if not X_class_seq:
        logger.warning("No sequences created for classification.")
        return None, None, None, None

    X_class_seq = np.array(X_class_seq)
    y_class_seq = np.array(y_class_seq)
    timestamps_class_seq = np.array(timestamps_class_seq)

    logger.info(f"Prepared classification data. X shape: {X_class_seq.shape}, y shape: {y_class_seq.shape}")
    return X_class_seq, y_class_seq, timestamps_class_seq, existing_feature_cols_class


if __name__ == "__main__":
    # Example Usage (Optional: Add test code here)
    logging.basicConfig(level=logging.INFO)
    logger.info("Running use_data.py script example...")
    try:
        # Construct the path relative to the script location or use absolute path
        script_dir = os.path.dirname(__file__) #<-- directory of the script
        # Go up two levels from ui/dash_app to the project root SFWRE-G4
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
        data_path = os.path.join(project_root, "data", "demand_prediction", "cleaned_demand_data.csv")

        df_raw = load_cleaned_data(path=data_path)
        if df_raw is not None and not df_raw.empty:
            daily_agg = aggregate_daily(df_raw, by_type=True, by_location=True)
            if not daily_agg.empty:
                logger.info("Aggregation successful. Sample:")
                logger.info(daily_agg.head())
                logger.info(f"Shortage flag sum in aggregated: {daily_agg['region_b_shortage_flag'].sum()}")

                # Test scaling preparation
                X_d, y_d, s_d, t_d = prepare_scaled_data(daily_agg, window=14, by_type=True)
                if X_d:
                    logger.info(f"Scaling preparation successful. Example group key: {list(X_d.keys())[0]}")
                    logger.info(f"X shape for group: {list(X_d.values())[0].shape}")
                else:
                    logger.warning("Scaling preparation returned empty dictionaries.")

                # Test classification preparation (using the first group's data for example)
                # Note: Classification might be better done on non-grouped data depending on goal
                first_group_key = list(daily_agg.groupby(['food_type', 'region']).groups.keys())[0]
                first_group_df = daily_agg.groupby(['food_type', 'region']).get_group(first_group_key)
                X_c, y_c, t_c, f_c = prepare_classification_data(first_group_df, window=14)
                if X_c is not None:
                     logger.info(f"Classification preparation successful for group {first_group_key}.")
                     logger.info(f"X_class shape: {X_c.shape}")
                else:
                    logger.warning(f"Classification preparation failed for group {first_group_key}.")

            else:
                logger.warning("Aggregation resulted in an empty DataFrame.")
        else:
            logger.warning("Failed to load raw data.")

    except Exception as e:
        logger.error(f"Error in example usage: {e}", exc_info=True)