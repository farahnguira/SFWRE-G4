import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os
import logging

logger = logging.getLogger(__name__)

def load_cleaned_data(path="data/demand_prediction/cleaned_demand_data.csv"):
    """Loads the cleaned data from the specified path."""
    try:
        if not os.path.exists(path):
            logger.error(f"Cleaned data file not found at {path}")
            raise FileNotFoundError(f"Cleaned data file not found at {path}")

        df = pd.read_csv(path)
        # Remove duplicates based on key columns
        df = df.drop_duplicates(subset=['timestamp', 'recipient_id', 'region', 'food_type', 'demand_kg'], keep='first')
        logger.info(f"Data loaded successfully from {path}. Shape after deduplication: {df.shape}")
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error loading cleaned data: {e}")
        raise

def aggregate_daily(df, by_type=False, by_location=False):
    """Aggregates demand by timestamp, optionally by type and location."""
    try:
        group_cols = ['timestamp']
        if by_location and 'region' in df.columns:
            group_cols.append('region')
        if by_type and 'food_type' in df.columns:
            group_cols.append('food_type')

        daily_agg = df.groupby(group_cols)['demand_kg'].sum().reset_index()

        min_step = df['timestamp'].min()
        max_step = df['timestamp'].max()
        step_range = pd.RangeIndex(start=min_step, stop=max_step + 1, step=1)

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

        if not grouping_values:
            full_index_df = pd.DataFrame({'timestamp': step_range})
        else:
            index_tuples = [step_range] + [v for v in grouping_values.values()]
            multi_index = pd.MultiIndex.from_product(index_tuples, names=['timestamp'] + list(grouping_values.keys()))
            full_index_df = pd.DataFrame(index=multi_index).reset_index()

        daily_filled = pd.merge(full_index_df, daily_agg, on=group_cols, how='left')
        daily_filled['demand_kg'] = daily_filled['demand_kg'].fillna(0)
        daily_filled = daily_filled.sort_values(by=group_cols).reset_index(drop=True)
        logger.info(f"Data aggregated by timestamp. Grouping by: {group_cols}. Shape: {daily_filled.shape}")
        return daily_filled
    except Exception as e:
        logger.error(f"Error during daily aggregation: {e}")
        raise

def _prepare_scaled_data_single(daily, window):
    """Prepares features, scales data, and creates sequences for one group (LSTM)."""
    if daily.empty or len(daily) <= window:
        logger.warning(f"Not enough data for sequence creation. Need > {window} rows, got {len(daily)}.")
        return np.array([]), np.array([]), None, pd.Series(dtype='int64')

    daily_features = daily.copy()
    target_col = 'demand_kg'

    # Add temporal features
    daily_features['day_of_week'] = (daily_features['timestamp'] % 7).astype(int)  # 0 to 6
    # Use disaster_flag as proxy for holiday in Region_A (shortage-driven spikes)
    if 'disaster_flag' in daily_features.columns:
        daily_features['is_holiday'] = daily_features['disaster_flag'].astype(int)
    else:
        daily_features['is_holiday'] = 0
    # Add interaction feature for Region_A during shortages
    if 'disaster_flag' in daily_features.columns and 'region' in daily_features.columns:
        daily_features['shortage_region_interaction'] = daily_features['disaster_flag'] * (daily_features['region'] == 'Region_A').astype(int)

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

    # Define Feature Columns
    feature_cols = [
        'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3', 'quantity_lag_7', 'quantity_lag_14', 'quantity_lag_28',
        'quantity_roll_mean_3', 'quantity_roll_mean_7', 'quantity_roll_mean_14', 'quantity_roll_mean_28',
        'quantity_roll_std_3', 'quantity_roll_std_7', 'quantity_roll_std_14', 'quantity_roll_std_28',
        'quantity_roll_median_3', 'quantity_roll_median_7', 'quantity_roll_median_14', 'quantity_roll_median_28',
        'day_of_week', 'is_weekend', 'is_holiday', 'shortage_region_interaction', 'timestamp'
    ]

    # Drop NaNs introduced by lags/rolling features
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

    # Scaling
    feature_cols = [col for col in feature_cols if col in daily_features.columns]
    if not feature_cols:
        logger.error("No feature columns available for scaling after processing.")
        return np.array([]), np.array([]), None, pd.Series(dtype='int64')

    scaler_target = StandardScaler()
    scaled_target = scaler_target.fit_transform(daily_features[[target_col]]).astype(np.float32)

    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(daily_features[feature_cols]).astype(np.float32)

    combined_scaled_data = np.hstack((scaled_target, scaled_features))
    final_timestamps = daily_features['timestamp']

    # Create Sequences
    X, y = [], []
    sequence_timestamps = []
    target_col_index = 0

    for i in range(len(combined_scaled_data) - window):
        X.append(combined_scaled_data[i : i + window, :])
        y.append(combined_scaled_data[i + window, target_col_index])
        sequence_timestamps.append(final_timestamps.iloc[i + window])

    if not X:
        logger.warning("Sequence list X is empty after processing.")
        return np.array([]), np.array([]), None, pd.Series(dtype='int64')

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    sequence_timestamps = pd.Series(sequence_timestamps, dtype='int64')
    return X, y, scaler_target, sequence_timestamps

def prepare_scaled_data(daily_data, window=14, by_type=False):
    """Wrapper to prepare scaled data, handling grouping."""
    X_dict, y_dict, scaler_dict, timestamps_dict = {}, {}, {}, {}
    group_cols = []
    if by_type and 'food_type' in daily_data.columns:
        group_cols.append('food_type')
    if 'region' in daily_data.columns:
        group_cols.append('region')

    if not group_cols:
        logger.info("Preparing data for overall aggregation.")
        X, y, scaler, timestamps = _prepare_scaled_data_single(daily_data, window)
        if X.size > 0 and y.size > 0:
            X_dict['all'] = X
            y_dict['all'] = y
            scaler_dict['all'] = scaler
            timestamps_dict['all'] = timestamps
        else:
            logger.warning("No data prepared for overall aggregation.")
    else:
        try:
            grouped_data = daily_data.groupby(group_cols)
            for group_key, group_df in grouped_data:
                logger.info(f"Preparing data for group: {group_key}")
                X, y, scaler, timestamps = _prepare_scaled_data_single(group_df.copy(), window)
                if X.size > 0 and y.size > 0:
                    X_dict[group_key] = X
                    y_dict[group_key] = y
                    scaler_dict[group_key] = scaler
                    timestamps_dict[group_key] = timestamps
                else:
                    logger.warning(f"No data prepared for group: {group_key}")
        except KeyError as e:
            logger.error(f"Grouping failed. Column '{e}' not found in daily_data. Columns available: {daily_data.columns.tolist()}")
            return {}, {}, {}, {}
        except Exception as e:
            logger.error(f"Error during grouped data preparation: {e}", exc_info=True)
            raise
    return X_dict, y_dict, scaler_dict, timestamps_dict

def prepare_classification_data(daily, window=14):
    """Prepares features and sequences for the classification model."""
    y_binary = (daily['demand_kg'] > 0).astype(np.float32)
    daily_features = daily.copy()
    target_col = 'demand_kg'

    # Add temporal features
    daily_features['day_of_week'] = (daily_features['timestamp'] % 7).astype(int)
    if 'disaster_flag' in daily_features.columns:
        daily_features['is_holiday'] = daily_features['disaster_flag'].astype(int)
    else:
        daily_features['is_holiday'] = 0
    if 'disaster_flag' in daily_features.columns and 'region' in daily_features.columns:
        daily_features['shortage_region_interaction'] = daily_features['disaster_flag'] * (daily_features['region'] == 'Region_A').astype(int)

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

    # Location Dummies
    if 'region' in daily_features.columns:
        location_dummies = pd.get_dummies(daily_features['region'], prefix='loc', dummy_na=False).astype(np.float32)
        daily_features = pd.concat([daily_features, location_dummies], axis=1)

    # Define Feature Columns
    feature_cols = [
        'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3', 'quantity_lag_7', 'quantity_lag_14', 'quantity_lag_28',
        'quantity_roll_mean_3', 'quantity_roll_mean_7', 'quantity_roll_mean_14', 'quantity_roll_mean_28',
        'quantity_roll_std_3', 'quantity_roll_std_7', 'quantity_roll_std_14', 'quantity_roll_std_28',
        'quantity_roll_median_3', 'quantity_roll_median_7', 'quantity_roll_median_14', 'quantity_roll_median_28',
        'day_of_week', 'is_weekend', 'is_holiday', 'shortage_region_interaction', 'timestamp'
    ]
    feature_cols += [col for col in daily_features if col.startswith('loc_')]

    # Drop NaNs
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

    final_feature_cols = [col for col in feature_cols if col in daily_features.columns]
    if not final_feature_cols:
        logger.error("No valid feature columns remaining for classification model.")
        return np.array([]), np.array([]), pd.Series(dtype='int64')

    features_final = daily_features[final_feature_cols].astype(np.float32)
    final_timestamps = daily_features['timestamp']

    # Create Sequences
    X_clf, y_clf = [], []
    sequence_timestamps_clf = []
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
    sequence_timestamps_clf = pd.Series(sequence_timestamps_clf, dtype='int64')
    logger.info(f"Classification data prepared. X_clf shape: {X_clf.shape}, y_clf shape: {y_clf.shape}")
    return X_clf, y_clf, sequence_timestamps_clf

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running use_data.py standalone example...")
    try:
        df_main = load_cleaned_data()
        daily_agg_main = aggregate_daily(df_main, by_type=True, by_location=True)
        window_size = 14
        X_d, y_d, scaler_d, timestamps_d = prepare_scaled_data(daily_agg_main, window=window_size, by_type=True)
        if X_d:
            first_key = list(X_d.keys())[0]
            logger.info(f"LSTM Data Example (Group: {first_key}):")
            logger.info(f"  X shape: {X_d[first_key].shape}")
            logger.info(f"  y shape: {y_d[first_key].shape}")
            logger.info(f"  Timestamps length: {len(timestamps_d[first_key])}")
            logger.info(f"  Scaler Type: {type(scaler_d[first_key])}")
        else:
            logger.warning("No LSTM data generated in example.")
        daily_agg_clf = aggregate_daily(df_main, by_type=False, by_location=True)
        X_c, y_c, timestamps_c = prepare_classification_data(daily_agg_clf, window=window_size)
        if X_c.size > 0:
            logger.info("Classification Data Example:")
            logger.info(f"  X_clf shape: {X_c.shape}")
            logger.info(f"  y_clf shape: {y_c.shape}")
            logger.info(f"  Timestamps length: {len(timestamps_c)}")
        else:
            logger.warning("No classification data generated in example.")
    except Exception as e:
        logger.error(f"Error in standalone example: {e}", exc_info=True)