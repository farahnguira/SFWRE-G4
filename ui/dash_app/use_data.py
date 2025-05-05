import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging
# Removed date import as it's no longer used

logger = logging.getLogger(__name__)

# --- Removed add_priority_feature function ---
# --- Removed HOLIDAYS and add_event_flags function ---
# --- Removed add_temporal_features function ---


def load_cleaned_data(path="data/demand_prediction/cleaned_demand_data.csv"):
    """Loads the cleaned data from the specified path."""
    try:
        # Correct the path relative to the project root if necessary
        # Assuming the script is run from the project root or the path is absolute
        if not os.path.exists(path):
            logger.error(f"Cleaned data file not found at {path}")
            raise FileNotFoundError(f"Cleaned data file not found at {path}")

        df = pd.read_csv(path)
        logger.info(f"Data loaded successfully from {path}. Shape: {df.shape}")

        # Basic validation (e.g., check if essential columns exist)
        # Removed the strict check for 'region' and 'food_type' as they might be handled differently post-cleaning
        # essential_cols = {'timestamp', 'demand_kg'}
        # missing_essential = essential_cols - set(df.columns)
        # if missing_essential:
        #     logger.error(f"Missing essential columns: {list(missing_essential)}")
        #     raise ValueError(f"Missing essential columns: {list(missing_essential)}")

        # Convert timestamp to datetime if needed (assuming it's already done or not required here)
        # df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df
    except FileNotFoundError:
        # Re-raise file not found to be handled by caller
        raise
    except Exception as e:
        logger.error(f"Error loading cleaned data: {e}")
        raise # Re-raise other exceptions


def aggregate_daily(df, by_type=False, by_location=False):
    """Aggregates demand by timestamp, optionally by type and location."""
    try:
        group_cols = ['timestamp'] # Group by timestamp instead of date
        if by_location and 'region' in df.columns: # Use 'region' column
            group_cols.append('region')
        if by_type and 'food_type' in df.columns: # Use 'food_type' column
            group_cols.append('food_type')

        # Aggregate quantity
        daily_agg = df.groupby(group_cols)['demand_kg'].sum().reset_index() # Use demand_kg

        # Create a complete timestamp range for each group to fill missing steps
        min_step = df['timestamp'].min()
        max_step = df['timestamp'].max()
        step_range = pd.RangeIndex(start=min_step, stop=max_step + 1, step=1) # Use RangeIndex for steps

        # Create all combinations of steps and grouping columns
        grouping_values = {}
        if by_location and 'region' in df.columns:
            locations = df['region'].dropna().unique()
            if len(locations) > 0:
                 grouping_values['region'] = locations
            else:
                 logger.warning("No valid regions found for grouping.")
                 by_location = False
                 if 'region' in group_cols: group_cols.remove('region')

        if by_type and 'food_type' in df.columns:
            types = df['food_type'].dropna().unique()
            if len(types) > 0:
                 grouping_values['food_type'] = types
            else:
                 logger.warning("No valid food types found for grouping.")
                 by_type = False
                 if 'food_type' in group_cols: group_cols.remove('food_type')


        if not grouping_values: # Only grouping by timestamp
            full_index_df = pd.DataFrame({'timestamp': step_range})
        else:
            index_tuples = [step_range] + [v for v in grouping_values.values()]
            multi_index = pd.MultiIndex.from_product(index_tuples, names=['timestamp'] + list(grouping_values.keys()))
            full_index_df = pd.DataFrame(index=multi_index).reset_index()


        # Merge aggregated data with the full index
        daily_filled = pd.merge(full_index_df, daily_agg, on=group_cols, how='left')

        # Fill missing quantities with 0
        daily_filled['demand_kg'] = daily_filled['demand_kg'].fillna(0) # Use demand_kg

        # Sort by timestamp and other group columns
        daily_filled = daily_filled.sort_values(by=group_cols).reset_index(drop=True)

        logger.info(f"Data aggregated by timestamp. Grouping by: {group_cols}. Shape: {daily_filled.shape}")
        return daily_filled

    except Exception as e:
        logger.error(f"Error during daily aggregation: {e}")
        raise

# --- Removed add_temporal_features function ---

def _prepare_scaled_data_single(daily, window): # Removed original_df parameter
    """Prepares features, scales data, and creates sequences for one group (LSTM)."""
    if daily.empty or len(daily) <= window:
        logger.warning(f"Not enough data for sequence creation. Need > {window} rows, got {len(daily)}.")
        return np.array([]), np.array([]), None, pd.Series(dtype='int64') # Return timestamp series

    # --- Feature Engineering ---
    daily_features = daily.copy()
    target_col = 'demand_kg' # Use demand_kg

    # Lag Features
    lags = [1, 2, 3, 7, 14, 28]
    for lag in lags:
        daily_features[f'quantity_lag_{lag}'] = daily_features[target_col].shift(lag)

    # Rolling Features
    roll_windows = [3, 7, 14, 28]
    for rw in roll_windows:
        daily_features[f'quantity_roll_mean_{rw}'] = daily_features[target_col].rolling(window=rw, min_periods=1).mean()
        daily_features[f'quantity_roll_std_{rw}'] = daily_features[target_col].rolling(window=rw, min_periods=1).std()
        daily_features[f'quantity_roll_median_{rw}'] = daily_features[target_col].rolling(window=rw, min_periods=1).median()

    # --- Removed Temporal Features (Date-based) ---
    # --- Removed Event/Holiday Flags (Date-based) ---
    # --- Removed Days Until Expiry ---
    # --- Removed Priority Features ---

    # Keep existing flags from the data
    if 'is_weekend' not in daily_features.columns: daily_features['is_weekend'] = 0
    if 'disaster_flag' not in daily_features.columns: daily_features['disaster_flag'] = 0


    # --- Define Feature Columns ---
    feature_cols = [
        # Lags
        'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3', 'quantity_lag_7', 'quantity_lag_14', 'quantity_lag_28',
        # Rolling Means
        'quantity_roll_mean_3', 'quantity_roll_mean_7', 'quantity_roll_mean_14', 'quantity_roll_mean_28',
        # Rolling Stds
        'quantity_roll_std_3', 'quantity_roll_std_7', 'quantity_roll_std_14', 'quantity_roll_std_28',
        # Rolling Medians
        'quantity_roll_median_3', 'quantity_roll_median_7', 'quantity_roll_median_14', 'quantity_roll_median_28',
        # Existing Flags
        'is_weekend', 'disaster_flag',
        # Timestamp itself (optional, can sometimes help)
        'timestamp'
    ]

    # --- Drop NaNs introduced by lags/rolling features ---
    nan_check_cols = [col for col in feature_cols if 'lag' in col or 'roll' in col]
    nan_check_cols = [col for col in nan_check_cols if col in daily_features.columns]
    if nan_check_cols:
        valid_index = daily_features.dropna(subset=nan_check_cols).index
        daily_features = daily_features.loc[valid_index]
    else:
        logger.warning("No lag/roll columns found to check for NaNs.")


    if daily_features.empty or len(daily_features) <= window:
        logger.warning(f"Not enough data after adding features and dropping NaNs. Need > {window} rows, got {len(daily_features)}.")
        return np.array([]), np.array([]), None, pd.Series(dtype='int64')

    # --- Scaling (Using StandardScaler) ---
    feature_cols = [col for col in feature_cols if col in daily_features.columns]
    if not feature_cols:
         logger.error("No feature columns available for scaling after processing.")
         return np.array([]), np.array([]), None, pd.Series(dtype='int64')

    # Scale target variable
    scaler_target = StandardScaler()
    scaled_target = scaler_target.fit_transform(daily_features[[target_col]]).astype(np.float32)

    # Scale features
    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(daily_features[feature_cols]).astype(np.float32)

    # Combine scaled target and features
    combined_scaled_data = np.hstack((scaled_target, scaled_features))
    final_timestamps = daily_features['timestamp'] # Use timestamp

    # --- Create Sequences ---
    X, y = [], []
    sequence_timestamps = [] # Use timestamp
    target_col_index = 0

    for i in range(len(combined_scaled_data) - window):
        X.append(combined_scaled_data[i : i + window, :]) # All columns (target + features)
        y.append(combined_scaled_data[i + window, target_col_index]) # Target column
        sequence_timestamps.append(final_timestamps.iloc[i + window])

    if not X:
        logger.warning("Sequence list X is empty after processing.")
        return np.array([]), np.array([]), None, pd.Series(dtype='int64') # Return None for scaler if no data

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    sequence_timestamps = pd.Series(sequence_timestamps, dtype='int64') # Use timestamp

    return X, y, scaler_target, sequence_timestamps


def prepare_scaled_data(daily_data, window=7, by_type=False): # Removed original_df parameter
    """Wrapper to prepare scaled data, handling grouping."""
    X_dict, y_dict, scaler_dict, timestamps_dict = {}, {}, {}, {} # Renamed dates_dict

    group_cols = []
    if by_type and 'food_type' in daily_data.columns:
        group_cols.append('food_type')
    if 'region' in daily_data.columns: # Use region
        group_cols.append('region')

    # --- Removed Priority Feature addition ---

    if not group_cols:
        logger.info("Preparing data for overall aggregation.")
        # Pass None for original_df as it's not needed in _prepare_scaled_data_single anymore
        X, y, scaler, timestamps = _prepare_scaled_data_single(daily_data, window) # Removed original_df
        if X.size > 0 and y.size > 0:
            X_dict['all'] = X
            y_dict['all'] = y
            scaler_dict['all'] = scaler
            timestamps_dict['all'] = timestamps # Contains timestamps
        else:
            logger.warning("No data prepared for overall aggregation.")
    else:
        try:
            grouped_data = daily_data.groupby(group_cols)
            for group_key, group_df in grouped_data:
                logger.info(f"Preparing data for group: {group_key}")
                # Pass None for original_df
                X, y, scaler, timestamps = _prepare_scaled_data_single(group_df.copy(), window) # Removed original_df
                if X.size > 0 and y.size > 0:
                    X_dict[group_key] = X
                    y_dict[group_key] = y
                    scaler_dict[group_key] = scaler
                    timestamps_dict[group_key] = timestamps # Contains timestamps
                else:
                     logger.warning(f"No data prepared for group: {group_key}")
        except KeyError as e:
             logger.error(f"Grouping failed. Column '{e}' not found in daily_data. Columns available: {daily_data.columns.tolist()}")
             return {}, {}, {}, {}
        except Exception as e:
             logger.error(f"Error during grouped data preparation: {e}", exc_info=True)
             raise

    return X_dict, y_dict, scaler_dict, timestamps_dict # Return timestamps_dict


def prepare_classification_data(daily, window=7): # Removed original_df parameter
    """Prepares features and sequences for the classification model."""
    # Target: 1 if demand_kg > 0, else 0
    y_binary = (daily['demand_kg'] > 0).astype(np.float32) # Use demand_kg

    daily_features = daily.copy()
    target_col = 'demand_kg' # Keep target temporarily for feature calculation

    # --- Feature Engineering (similar to LSTM prep) ---
    # Lag Features
    lags = [1, 2, 3, 7, 14, 28]
    for lag in lags:
        daily_features[f'quantity_lag_{lag}'] = daily_features[target_col].shift(lag)

    # Rolling Features
    roll_windows = [3, 7, 14, 28]
    for rw in roll_windows:
        daily_features[f'quantity_roll_mean_{rw}'] = daily_features[target_col].rolling(window=rw, min_periods=1).mean()
        daily_features[f'quantity_roll_std_{rw}'] = daily_features[target_col].rolling(window=rw, min_periods=1).std()
        daily_features[f'quantity_roll_median_{rw}'] = daily_features[target_col].rolling(window=rw, min_periods=1).median()

    # --- Removed Temporal Features (Date-based) ---
    # --- Removed Event/Holiday Flags (Date-based) ---
    # --- Removed Priority Features ---
    # --- Removed Days Until Expiry ---

    # Keep existing flags
    if 'is_weekend' not in daily_features.columns: daily_features['is_weekend'] = 0
    if 'disaster_flag' not in daily_features.columns: daily_features['disaster_flag'] = 0

    # Location Dummies (if region exists)
    if 'region' in daily_features.columns:
        location_dummies = pd.get_dummies(daily_features['region'], prefix='loc', dummy_na=False).astype(np.float32)
        daily_features = pd.concat([daily_features, location_dummies], axis=1)


    # --- Define Feature Columns for Classifier ---
    feature_cols = [
        # Lags
        'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3', 'quantity_lag_7', 'quantity_lag_14', 'quantity_lag_28',
        # Rolling Means
        'quantity_roll_mean_3', 'quantity_roll_mean_7', 'quantity_roll_mean_14', 'quantity_roll_mean_28',
        # Rolling Stds
        'quantity_roll_std_3', 'quantity_roll_std_7', 'quantity_roll_std_14', 'quantity_roll_std_28',
        # Rolling Medians
        'quantity_roll_median_3', 'quantity_roll_median_7', 'quantity_roll_median_14', 'quantity_roll_median_28',
        # Existing Flags
        'is_weekend', 'disaster_flag',
        # Timestamp itself (optional)
        'timestamp'
    ]
    # Add location dummies dynamically
    feature_cols += [col for col in daily_features if col.startswith('loc_')]

    # --- Drop NaNs ---
    nan_check_cols = [col for col in feature_cols if 'lag' in col or 'roll' in col]
    nan_check_cols = [col for col in nan_check_cols if col in daily_features.columns]
    if nan_check_cols:
        valid_index = daily_features.dropna(subset=nan_check_cols).index
        daily_features = daily_features.loc[valid_index]
        y_binary = y_binary.loc[valid_index]
    else:
        logger.warning("No lag/roll columns found for NaN check in classification data.")


    if daily_features.empty:
        logger.error("DataFrame empty after feature engineering and NaN removal in prepare_classification_data.")
        return np.array([]), np.array([]), pd.Series(dtype='int64')

    # --- Select Final Features and Timestamps ---
    final_feature_cols = [col for col in feature_cols if col in daily_features.columns]
    if not final_feature_cols:
        logger.error("No valid feature columns remaining for classification model.")
        return np.array([]), np.array([]), pd.Series(dtype='int64')

    features_final = daily_features[final_feature_cols].astype(np.float32)
    final_timestamps = daily_features['timestamp'] # Use timestamp

    # --- Create Sequences for Classifier ---
    X_clf, y_clf = [], []
    sequence_timestamps_clf = [] # Use timestamp

    features_np = features_final.to_numpy()
    y_binary_np = y_binary.to_numpy()

    for i in range(len(features_np) - window):
        X_clf.append(features_np[i : i + window, :])
        y_clf.append(y_binary_np[i + window])
        sequence_timestamps_clf.append(final_timestamps.iloc[i + window])

    if not X_clf:
        logger.warning("Sequence list X_clf is empty after processing.")
        return np.array([]), np.array([]), pd.Series(dtype='int64')

    X_clf = np.array(X_clf).astype(np.float32)
    y_clf = np.array(y_clf).astype(np.float32)
    sequence_timestamps_clf = pd.Series(sequence_timestamps_clf, dtype='int64') # Use timestamp

    logger.info(f"Classification data prepared. X_clf shape: {X_clf.shape}, y_clf shape: {y_clf.shape}")
    return X_clf, y_clf, sequence_timestamps_clf

# Example usage (optional) - Updated
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running use_data.py standalone example...")
    try:
        df_main = load_cleaned_data() # Loads from the new default path

        daily_agg_main = aggregate_daily(df_main, by_type=True, by_location=True)

        window_size = 14
        # Pass None for original_df
        X_d, y_d, scaler_d, timestamps_d = prepare_scaled_data(daily_agg_main, window=window_size, by_type=True) # Removed original_df

        if X_d:
            first_key = list(X_d.keys())[0]
            logger.info(f"LSTM Data Example (Group: {first_key}):")
            logger.info(f"  X shape: {X_d[first_key].shape}")
            logger.info(f"  y shape: {y_d[first_key].shape}")
            logger.info(f"  Timestamps length: {len(timestamps_d[first_key])}") # Now timestamps
            logger.info(f"  Scaler Type: {type(scaler_d[first_key])}")
        else:
            logger.warning("No LSTM data generated in example.")

        daily_agg_clf = aggregate_daily(df_main, by_type=False, by_location=True)
        # Removed call to add_priority_feature
        # Pass None for original_df
        X_c, y_c, timestamps_c = prepare_classification_data(daily_agg_clf, window=window_size) # Removed original_df

        if X_c.size > 0:
            logger.info("Classification Data Example:")
            logger.info(f"  X_clf shape: {X_c.shape}")
            logger.info(f"  y_clf shape: {y_c.shape}")
            logger.info(f"  Timestamps length: {len(timestamps_c)}") # Now timestamps
        else:
            logger.warning("No classification data generated in example.")

    except Exception as e:
        logger.error(f"Error in standalone example: {e}", exc_info=True)