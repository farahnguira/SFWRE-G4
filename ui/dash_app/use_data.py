import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def load_cleaned_data(path="amine/data/cleaned_data.csv"):
    """Load the cleaned donation/demand CSV."""
    parse_dates = ["donation_date", "expiry_date"]
    df = pd.read_csv(path, parse_dates=parse_dates)
    if "location" not in df.columns:
        logger.error("'location' column missing in cleaned_data.csv.")
        raise ValueError("'location' column required.")
    df['donation_date'] = pd.to_datetime(df['donation_date'], errors='coerce')
    if df['donation_date'].isnull().any():
        logger.error("NaNs found in donation_date after conversion.")
        raise ValueError("Invalid donation_date values.")
    return df

def aggregate_daily(df, by_type=False):
    """Aggregate quantities by donation_date, optionally by food type and location."""
    df['donation_date'] = pd.to_datetime(df['donation_date'], errors='coerce')
    if df['donation_date'].isnull().any():
        logger.error("NaNs found in donation_date during aggregation.")
        raise ValueError("Invalid donation_date values.")
    
    if by_type:
        daily = df.groupby([df['donation_date'].dt.date, 'type', 'location'])['quantity'].sum().reset_index()
        daily['donation_date'] = pd.to_datetime(daily['donation_date'])
        daily = daily.rename(columns={'quantity': 'total_quantity'})
        
        types = daily['type'].unique()
        locations = daily['location'].unique()
        full_dfs = []
        for t in types:
            for loc in locations:
                df_subset = daily[(daily['type'] == t) & (daily['location'] == loc)].copy()
                if df_subset.empty:
                    logger.warning(f"No data for type {t} in location {loc}. Skipping.")
                    continue
                full_range = pd.date_range(start=df_subset['donation_date'].min(), 
                                         end=df_subset['donation_date'].max(), freq='D')
                df_subset = (df_subset.set_index('donation_date')
                            .reindex(full_range, fill_value=0)
                            [['total_quantity', 'type', 'location']]
                            .reset_index())
                df_subset.columns = ['donation_date', 'total_quantity', 'type', 'location']
                df_subset['donation_date'] = pd.to_datetime(df_subset['donation_date'])
                full_dfs.append(df_subset)
        daily = pd.concat(full_dfs).reset_index(drop=True)
        daily['donation_date'] = pd.to_datetime(daily['donation_date'])
    else:
        daily = df.groupby([df['donation_date'].dt.date, 'location'])['quantity'].sum().reset_index()
        daily['donation_date'] = pd.to_datetime(daily['donation_date'])
        daily = daily.rename(columns={'quantity': 'total_quantity'})
        locations = daily['location'].unique()
        full_dfs = []
        for loc in locations:
            df_subset = daily[daily['location'] == loc].copy()
            full_range = pd.date_range(start=df_subset['donation_date'].min(), 
                                     end=df_subset['donation_date'].max(), freq='D')
            df_subset = (df_subset.set_index('donation_date')
                        .reindex(full_range, fill_value=0)
                        [['total_quantity', 'location']]
                        .reset_index())
            df_subset.columns = ['donation_date', 'total_quantity', 'location']
            df_subset['donation_date'] = pd.to_datetime(df_subset['donation_date'])
            full_dfs.append(df_subset)
        daily = pd.concat(full_dfs).reset_index(drop=True)
        daily['donation_date'] = pd.to_datetime(daily['donation_date'])
    
    if not pd.api.types.is_datetime64_any_dtype(daily['donation_date']):
        logger.error("donation_date is not datetime64[ns] after aggregation.")
        raise ValueError("donation_date must be datetime64[ns].")
    
    return daily

def make_sequences(daily, window=7):
    """Build input/output sequences for an RNN."""
    values = daily["total_quantity"].values
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i : i + window])
        y.append(values[i + window])
    X = pd.DataFrame(X)
    y = pd.Series(y, name="target")
    return X, y

def prepare_scaled_data(daily_data, window=7, by_type=False, original_df=None):
    """Prepare scaled data for LSTM, handling single or multiple types/locations, and return dates."""
    X_dict, y_dict, scaler_dict, dates_dict = {}, {}, {}, {}
    
    if by_type:
        if 'type' not in daily_data.columns or 'location' not in daily_data.columns:
            logger.error("'type' or 'location' column missing for by_type processing.")
            return X_dict, y_dict, scaler_dict, dates_dict
        
        for (food_type, loc), group in daily_data.groupby(['type', 'location']):
            key = f"{food_type}_{loc}"
            logger.info(f"Preparing data for type: {food_type}, location: {loc}")
            X, y, scaler, dates = _prepare_scaled_data_single(group, window, original_df)
            if X.size > 0 and y.size > 0:
                X_dict[key] = X
                y_dict[key] = y
                scaler_dict[key] = scaler
                dates_dict[key] = dates
            else:
                logger.warning(f"No data prepared for type: {food_type}, location: {loc}")
    else:
        if 'location' not in daily_data.columns:
            logger.error("'location' column missing for non-by_type processing.")
            return X_dict, y_dict, scaler_dict, dates_dict
        
        for loc, group in daily_data.groupby('location'):
            key = f"all_{loc}"
            logger.info(f"Preparing data for overall daily aggregation, location: {loc}")
            X, y, scaler, dates = _prepare_scaled_data_single(group, window, original_df)
            if X.size > 0 and y.size > 0:
                X_dict[key] = X
                y_dict[key] = y
                scaler_dict[key] = scaler
                dates_dict[key] = dates
            else:
                logger.warning(f"No data prepared for location: {loc}")
    
    return X_dict, y_dict, scaler_dict, dates_dict

def _prepare_scaled_data_single(daily, window, original_df):
    """Helper to prepare scaled data for a single food type/location with enhanced features."""
    if daily.empty or len(daily) <= window:
        logger.warning(f"Not enough data for sequence creation. Need > {window} rows, got {len(daily)}.")
        return np.array([]), np.array([]), None, pd.Series(dtype='datetime64[ns]')
    
    daily_features = daily.copy()
    
    daily_features['donation_date'] = pd.to_datetime(daily_features['donation_date'], errors='coerce')
    if daily_features['donation_date'].isnull().any():
        logger.error("NaNs found in donation_date in _prepare_scaled_data_single.")
        return np.array([]), np.array([]), None, pd.Series(dtype='datetime64[ns]')
    
    daily_features['quantity_lag_1'] = daily_features['total_quantity'].shift(1)
    daily_features['quantity_lag_7'] = daily_features['total_quantity'].shift(7)
    daily_features['quantity_roll_mean_7'] = daily_features['total_quantity'].rolling(window=7).mean()
    
    daily_features = add_temporal_features(daily)
    
    if 'location' in daily_features.columns:
        location_dummies = pd.get_dummies(daily_features['location'], prefix='loc', dummy_na=False).astype(np.float32)
        daily_features = pd.concat([daily_features, location_dummies], axis=1)
    
    if original_df is not None and 'donation_date' in original_df.columns:
        original_df_copy = original_df.copy()
        original_df_copy['donation_date'] = pd.to_datetime(original_df_copy['donation_date'], errors='coerce')
        original_df_copy['expiry_date'] = pd.to_datetime(original_df_copy['expiry_date'], errors='coerce')
        original_df_copy['days_until_expiry'] = (original_df_copy['expiry_date'] - 
                                               original_df_copy['donation_date']).dt.days
        days_until_expiry_by_date = original_df_copy.groupby([original_df_copy['donation_date'].dt.date, 
                                                            'location'])['days_until_expiry'].mean().reset_index()
        days_until_expiry_by_date['donation_date'] = pd.to_datetime(days_until_expiry_by_date['donation_date'])
        daily_features = daily_features.merge(
            days_until_expiry_by_date,
            left_on=['donation_date', 'location'],
            right_on=['donation_date', 'location'],
            how='left'
        )
        daily_features['days_until_expiry'] = daily_features['days_until_expiry'].fillna(
            daily_features['days_until_expiry'].mean())
    
    daily_features = daily_features.dropna()
    if daily_features.empty or len(daily_features) <= window:
        logger.warning(f"Not enough data after feature engineering. Need > {window} rows, got {len(daily_features)}.")
        return np.array([]), np.array([]), None, pd.Series(dtype='datetime64[ns]')
    
    target_col = 'total_quantity'
    feature_cols = ['quantity_lag_1', 'quantity_lag_7', 'quantity_roll_mean_7',
                    'day_of_week', 'month', 'day']
    feature_cols += [col for col in daily_features if col.startswith('dow_')]
    feature_cols += [col for col in daily_features if col.startswith('loc_')]
    if 'days_until_expiry' in daily_features.columns:
        feature_cols.append('days_until_expiry')
    
    feature_cols = [col for col in feature_cols if col in daily_features.columns]
    if not feature_cols:
        logger.error("No feature columns available for scaling.")
        return np.array([]), np.array([]), None, pd.Series(dtype='datetime64[ns]')
    
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(daily_features[[target_col]]).astype(np.float32)
    
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(daily_features[feature_cols]).astype(np.float32)
    
    combined_scaled_data = np.hstack((scaled_target, scaled_features))
    final_dates = daily_features['donation_date']
    
    X, y, sequence_dates = [], [], []
    target_col_index = 0
    for i in range(len(combined_scaled_data) - window):
        X.append(combined_scaled_data[i : i + window, :])
        y.append(combined_scaled_data[i + window, target_col_index])
        sequence_dates.append(final_dates.iloc[i + window])
    
    if not X:
        logger.warning("Sequence list X is empty after processing.")
        return np.array([]), np.array([]), scaler_target, pd.Series(dtype='datetime64[ns]')
    
    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.float32)
    sequence_dates = pd.Series(sequence_dates)
    
    return X, y, scaler_target, sequence_dates

def prepare_classification_data(daily, window=7, original_df=None):
    """Prepare data for binary classification with enhanced features."""
    y_binary = (daily['total_quantity'] > 0).astype(np.float32)
    
    daily_features = daily.copy()
    
    daily_features['donation_date'] = pd.to_datetime(daily_features['donation_date'], errors='coerce')
    if daily_features['donation_date'].isnull().any():
        logger.error("NaNs found in donation_date in prepare_classification_data.")
        return np.array([]), np.array([]), pd.Series(dtype='datetime64[ns]')
    
    daily_features['quantity_lag_1'] = daily_features['total_quantity'].shift(1)
    daily_features['quantity_lag_7'] = daily_features['total_quantity'].shift(7)
    daily_features['quantity_roll_mean_7'] = daily_features['total_quantity'].rolling(window=7).mean()
    
    daily_features = add_temporal_features(daily_features)
    
    if 'location' in daily_features.columns:
        location_dummies = pd.get_dummies(daily_features['location'], prefix='loc', dummy_na=False).astype(np.float32)
        daily_features = pd.concat([daily_features, location_dummies], axis=1)
    
    # Check for priority column (added by add_priority_feature)
    if 'priority' not in daily_features.columns:
        logger.warning("Priority column missing in daily_features. Adding default priority.")
        daily_features['priority'] = 1.0  # Default priority
    else:
        daily_features['priority'] = daily_features['priority'].fillna(daily_features['priority'].mean())
        if daily_features['priority'].isnull().all():
            logger.warning("All priority values are NaN. Setting default priority to 1.0.")
            daily_features['priority'] = 1.0
    
    # Add days_until_expiry if original_df is provided
    if original_df is not None:
        original_df_copy = original_df.copy()
        original_df_copy['donation_date'] = pd.to_datetime(original_df_copy['donation_date'], errors='coerce')
        original_df_copy['expiry_date'] = pd.to_datetime(original_df_copy['expiry_date'], errors='coerce')
        original_df_copy['days_until_expiry'] = (original_df_copy['expiry_date'] - 
                                               original_df_copy['donation_date']).dt.days
        days_until_expiry_by_date = original_df_copy.groupby([original_df_copy['donation_date'].dt.date, 
                                                            'location'])['days_until_expiry'].mean().reset_index()
        days_until_expiry_by_date['donation_date'] = pd.to_datetime(days_until_expiry_by_date['donation_date'])
        daily_features = daily_features.merge(
            days_until_expiry_by_date,
            left_on=['donation_date', 'location'],
            right_on=['donation_date', 'location'],
            how='left'
        )
        daily_features['days_until_expiry'] = daily_features['days_until_expiry'].fillna(
            daily_features['days_until_expiry'].mean())
    
    feature_cols = ['total_quantity', 'quantity_lag_1', 'quantity_lag_7', 'quantity_roll_mean_7',
                    'day_of_week', 'month', 'day']
    feature_cols += [col for col in daily_features if col.startswith('dow_')]
    feature_cols += [col for col in daily_features if col.startswith('loc_')]
    if 'priority' in daily_features.columns:
        feature_cols.append('priority')
    if 'days_until_expiry' in daily_features.columns:
        feature_cols.append('days_until_expiry')
    
    feature_cols = [col for col in feature_cols if col in daily_features.columns]
    if not feature_cols:
        logger.error("No valid feature columns found before dropna.")
        return np.array([]), np.array([]), pd.Series(dtype='datetime64[ns]')
    
    valid_index = daily_features.dropna(subset=feature_cols).index
    daily_features = daily_features.loc[valid_index]
    y_binary = y_binary.loc[valid_index]
    
    if daily_features.empty or len(daily_features) <= window:
        logger.warning("Not enough data for classification sequence creation.")
        return np.array([]), np.array([]), pd.Series(dtype='datetime64[ns]')
    
    combined_features = daily_features[feature_cols].values.astype(np.float32)
    final_dates = daily_features['donation_date']
    
    X_clf, y_clf, sequence_dates = [], [], []
    for i in range(len(combined_features) - window):
        X_clf.append(combined_features[i : i + window])
        y_clf.append(y_binary.iloc[i + window])
        sequence_dates.append(final_dates.iloc[i + window])
    
    if not X_clf:
        logger.warning("Sequence list X_clf is empty after processing.")
        return np.array([]), np.array([]), pd.Series(dtype='datetime64[ns]')
    
    X_clf = np.array(X_clf).astype(np.float32)
    y_clf = np.array(y_clf).astype(np.float32)
    sequence_dates = pd.Series(sequence_dates)
    
    return X_clf, y_clf, sequence_dates

def add_temporal_features(daily):
    """Add time-based features to improve predictions."""
    df_features = daily.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df_features['donation_date']):
        logger.warning("donation_date column was not datetime. Converting.")
        df_features['donation_date'] = pd.to_datetime(df_features['donation_date'], errors='coerce')
        if df_features['donation_date'].isnull().any():
            logger.error("NaNs introduced in donation_date during conversion.")
            raise ValueError("Could not convert all donation_dates to datetime")
    
    df_features['day_of_week'] = df_features['donation_date'].dt.dayofweek
    df_features['month'] = df_features['donation_date'].dt.month
    df_features['day'] = df_features['donation_date'].dt.day
    dow_dummies = pd.get_dummies(df_features['day_of_week'], prefix='dow', dummy_na=False).astype(np.float32)
    df_features = pd.concat([df_features, dow_dummies], axis=1)
    df_features[['day_of_week', 'month', 'day']] = df_features[['day_of_week', 'month', 'day']].astype(np.float32)
    return df_features

if __name__ == "__main__":
    df = load_cleaned_data()
    daily = aggregate_daily(df)
    X, y = make_sequences(daily)
    print("Daily shape:", daily.shape)
    print("X shape:", X.shape, "y shape:", y.shape)