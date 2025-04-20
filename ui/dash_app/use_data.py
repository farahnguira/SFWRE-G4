import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging # Import logging

logger = logging.getLogger(__name__) # Add logger if not present

def load_cleaned_data(path="amine/data/cleaned_data.csv"):
    """Load the cleaned donation/demand CSV."""
    # Parse dates, including donation_date if it exists
    parse_dates = ["expiry_date"]
    if "donation_date" in pd.read_csv(path, nrows=1).columns:
        parse_dates.append("donation_date")
    df = pd.read_csv(path, parse_dates=parse_dates)
    return df

def aggregate_daily(df, by_type=False):
    """Aggregate quantities by expiry_date, optionally by food type."""
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])
    
    if by_type:
        daily = df.groupby([df['expiry_date'].dt.date, 'type'])['quantity'].sum().reset_index()
        daily['expiry_date'] = pd.to_datetime(daily['expiry_date'])
        daily = daily.rename(columns={'quantity': 'total_quantity'})
        
        types = daily['type'].unique()
        full_dfs = []
        for t in types:
            df_type = daily[daily['type'] == t].copy()
            full_range = pd.date_range(start=df_type['expiry_date'].min(), end=df_type['expiry_date'].max(), freq='D')
            df_type = (df_type.set_index('expiry_date')
                       .reindex(full_range, fill_value=0)
                       [['total_quantity']]
                       .reset_index())
            df_type['type'] = t
            df_type.columns = ['expiry_date', 'total_quantity', 'type']
            df_type['expiry_date'] = pd.to_datetime(df_type['expiry_date'])
            full_dfs.append(df_type)
        daily = pd.concat(full_dfs).reset_index(drop=True)
        daily['expiry_date'] = pd.to_datetime(daily['expiry_date'])
    else:
        daily = df.groupby(df['expiry_date'].dt.date)['quantity'].sum().reset_index()
        daily['expiry_date'] = pd.to_datetime(daily['expiry_date'])
        full_range = pd.date_range(start=daily['expiry_date'].min(), end=daily['expiry_date'].max(), freq='D')
        daily = daily.set_index('expiry_date').reindex(full_range, fill_value=0).reset_index()
        daily.columns = ['expiry_date', 'total_quantity']
        daily['expiry_date'] = pd.to_datetime(daily['expiry_date'])
    
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

def prepare_scaled_data(daily, window=7, by_type=False, original_df=None):
    """Prepare scaled data for LSTM model with temporal features, optionally by type."""
    if by_type:
        types = daily['type'].unique()
        X_dict, y_dict, scaler_dict = {}, {}, {}
        for t in types:
            df_type = daily[daily['type'] == t].copy()
            X, y, scaler = _prepare_scaled_data_single(df_type, window, original_df[original_df['type'] == t] if original_df is not None else None)
            X_dict[t] = X
            y_dict[t] = y
            scaler_dict[t] = scaler
        return X_dict, y_dict, scaler_dict
    else:
        return _prepare_scaled_data_single(daily, window, original_df)

def _prepare_scaled_data_single(daily, window, original_df):
    """Helper to prepare scaled data for a single food type."""
    if daily.empty or len(daily) <= window:
        logger.warning(f"Not enough data for sequence creation. Need > {window} rows, got {len(daily)}.")
        return np.array([]), np.array([]), None # Return empty arrays and None scaler

    # Scale the target variable ('total_quantity')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(daily[['total_quantity']]).astype(np.float32) # Scale and ensure float32

    # --- Feature Engineering ---
    all_features_list = [scaled_values] # Start with the scaled target

    # Add temporal features
    daily_with_features = add_temporal_features(daily)
    temporal_cols = ['day_of_week', 'month', 'day'] + [col for col in daily_with_features if col.startswith('dow_')]
    temporal_features = daily_with_features[temporal_cols].values.astype(np.float32) # Already float32 from add_temporal_features
    all_features_list.append(temporal_features)

    # Add days until expiry if possible
    if original_df is not None and 'donation_date' in original_df.columns and 'type' in daily.columns:
        food_type = daily['type'].iloc[0]
        original_df_type = original_df[original_df['type'] == food_type].copy() # Filter original_df for the current type and copy

        if not original_df_type.empty:
            # Use .loc for modifications
            original_df_type.loc[:, 'expiry_date'] = pd.to_datetime(original_df_type['expiry_date'], errors='coerce')
            original_df_type.loc[:, 'donation_date'] = pd.to_datetime(original_df_type['donation_date'], errors='coerce')
            original_df_type.dropna(subset=['expiry_date', 'donation_date'], inplace=True) # Drop rows where conversion failed

            if not original_df_type.empty:
                original_df_type.loc[:, 'days_until_expiry'] = (original_df_type['expiry_date'] - original_df_type['donation_date']).dt.days

                # Aggregate mean days_until_expiry by date for this type
                days_until_expiry_by_date = original_df_type.groupby(original_df_type['expiry_date'].dt.date)['days_until_expiry'].mean().reset_index()
                days_until_expiry_by_date = days_until_expiry_by_date.rename(columns={'expiry_date': 'date_key'})
                days_until_expiry_by_date['date_key'] = pd.to_datetime(days_until_expiry_by_date['date_key']) # Ensure date_key is datetime for merge

                # Merge onto the daily dataframe for this type
                daily_merged = daily.merge(
                    days_until_expiry_by_date,
                    left_on='expiry_date', # Merge directly on expiry_date if it's the index or a column
                    right_on='date_key',
                    how='left'
                )

                # Fill NaNs robustly (e.g., with 0) and ensure numeric type
                days_until_expiry_feature = daily_merged['days_until_expiry'].fillna(0).values.reshape(-1, 1).astype(np.float32)
                all_features_list.append(days_until_expiry_feature)
            else:
                 logger.warning(f"No valid 'days_until_expiry' data after filtering/cleaning for type {food_type}.")
        else:
            logger.warning(f"Original data frame filtered for type {food_type} is empty.")


    # Combine all features horizontally
    combined_features = np.hstack(all_features_list)

    # Check for NaNs before creating sequences
    if np.isnan(combined_features).any():
        logger.warning(f"NaNs detected in combined features before sequencing. Replacing with 0.")
        combined_features = np.nan_to_num(combined_features) # Replace NaNs with 0

    # --- Create Sequences ---
    X, y = [], []
    # Target 'y' is the scaled quantity at the *next* time step
    target_values = combined_features[:, 0] # First column is scaled_quantity

    for i in range(len(combined_features) - window):
        X.append(combined_features[i : i + window, :]) # Sequence of features (window, num_features)
        y.append(target_values[i + window])          # Target value at the end of the window + 1

    if not X: # Handle case where no sequences could be created
        logger.warning("Sequence list X is empty after processing.")
        return np.array([]), np.array([]), scaler

    X = np.array(X).astype(np.float32) # Ensure final array is float32
    y = np.array(y).astype(np.float32) # Ensure final array is float32

    return X, y, scaler

def prepare_classification_data(daily, window=7, original_df=None):
    """Prepare data for binary classification (zero vs non-zero)."""
    # Target: 1 if total_quantity > 0, else 0
    y_binary = (daily['total_quantity'] > 0).astype(np.float32)

    # --- Feature Engineering ---
    # Start with the quantity itself (or log-transformed/scaled version if preferred)
    base_features = daily[['total_quantity']].values.astype(np.float32)
    all_features_list = [base_features]

    # Add temporal features
    daily_with_features = add_temporal_features(daily)
    temporal_cols = ['day_of_week', 'month', 'day'] + [col for col in daily_with_features if col.startswith('dow_')]
    temporal_features = daily_with_features[temporal_cols].values.astype(np.float32)
    all_features_list.append(temporal_features)

    # Add priority feature (already added in main script, just need to select it)
    if 'priority' in daily_with_features.columns:
         priority_feature = daily_with_features[['priority']].fillna(0).values.astype(np.float32) # Fill NaNs and ensure float
         all_features_list.append(priority_feature)

    # Add days until expiry if possible (similar logic as in _prepare_scaled_data_single)
    if original_df is not None and 'donation_date' in original_df.columns:
        original_df_copy = original_df.copy() # Work on a copy
        # Use .loc for modifications
        original_df_copy.loc[:, 'expiry_date'] = pd.to_datetime(original_df_copy['expiry_date'], errors='coerce')
        original_df_copy.loc[:, 'donation_date'] = pd.to_datetime(original_df_copy['donation_date'], errors='coerce')
        original_df_copy.dropna(subset=['expiry_date', 'donation_date'], inplace=True)

        if not original_df_copy.empty:
            original_df_copy.loc[:, 'days_until_expiry'] = (original_df_copy['expiry_date'] - original_df_copy['donation_date']).dt.days
            days_until_expiry_by_date = original_df_copy.groupby(original_df_copy['expiry_date'].dt.date)['days_until_expiry'].mean().reset_index()
            days_until_expiry_by_date = days_until_expiry_by_date.rename(columns={'expiry_date': 'date_key'})
            # Ensure date_key is datetime
            days_until_expiry_by_date['date_key'] = pd.to_datetime(days_until_expiry_by_date['date_key'])

            # --- Add this line ---
            # Ensure the left key ('expiry_date' in daily) is also datetime before merging
            daily['expiry_date'] = pd.to_datetime(daily['expiry_date'], errors='coerce')
            # --- End of addition ---

            daily_merged = daily.merge(
                days_until_expiry_by_date,
                left_on='expiry_date', # Should now be datetime64[ns]
                right_on='date_key',   # Is datetime64[ns]
                how='left'
            )
            # Check if 'days_until_expiry_x' or similar exists and handle potential duplicate columns if needed
            # For simplicity, assuming 'days_until_expiry' is the correct column after merge
            days_until_expiry_feature = daily_merged['days_until_expiry'].fillna(0).values.reshape(-1, 1).astype(np.float32)
            all_features_list.append(days_until_expiry_feature)

    # Combine all features horizontally
    combined_features = np.hstack(all_features_list)

    # Check for NaNs before creating sequences
    if np.isnan(combined_features).any():
        logger.warning(f"NaNs detected in combined features for classification. Replacing with 0.")
        combined_features = np.nan_to_num(combined_features)

    # --- Create Sequences ---
    X_clf, y_clf = [], []
    if len(combined_features) <= window:
         logger.warning("Not enough data for classification sequence creation.")
         return np.array([]), np.array([])

    for i in range(len(combined_features) - window):
        X_clf.append(combined_features[i : i + window])
        y_clf.append(y_binary[i + window]) # Target is binary label at t+1

    if not X_clf:
        logger.warning("Sequence list X_clf is empty after processing.")
        return np.array([]), np.array([])

    X_clf = np.array(X_clf).astype(np.float32) # Ensure float32
    y_clf = np.array(y_clf).astype(np.float32) # Ensure float32

    return X_clf, y_clf

def add_temporal_features(daily):
    """Add time-based features to improve predictions"""
    df_features = daily.copy() # Work on a copy

    # --- Add this check and conversion ---
    if not pd.api.types.is_datetime64_any_dtype(df_features['expiry_date']):
        logger.warning("expiry_date column was not datetime in add_temporal_features. Converting.")
        df_features['expiry_date'] = pd.to_datetime(df_features['expiry_date'], errors='coerce')
        # Handle potential errors during conversion if necessary
        if df_features['expiry_date'].isnull().any():
             logger.error("NaNs introduced in expiry_date during conversion in add_temporal_features. Check data.")
             # Option: Drop rows with NaT dates, or raise an error
             # df_features = df_features.dropna(subset=['expiry_date'])
             # raise ValueError("Could not convert all expiry_dates to datetime")
    # --- End of addition ---

    # Now it's safe to use .dt
    df_features['day_of_week'] = df_features['expiry_date'].dt.dayofweek
    df_features['month'] = df_features['expiry_date'].dt.month
    df_features['day'] = df_features['expiry_date'].dt.day
    dow_dummies = pd.get_dummies(df_features['day_of_week'], prefix='dow', dummy_na=False).astype(np.float32) # Convert dummies to float
    df_features = pd.concat([df_features, dow_dummies], axis=1)
    # Ensure base temporal features are numeric
    df_features[['day_of_week', 'month', 'day']] = df_features[['day_of_week', 'month', 'day']].astype(np.float32)
    return df_features

if __name__ == "__main__":
    df = load_cleaned_data()
    daily = aggregate_daily(df)
    X, y = make_sequences(daily)
    print("Daily shape:", daily.shape)
    print("X shape:", X.shape, "y shape:", y.shape)