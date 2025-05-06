import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_lagged_features(df, group_cols, lag_col, lags):
    df_orig = df.copy()
    for lag in lags:
        df[f'{lag_col}_lag_{lag}'] = df.groupby(group_cols)[lag_col].shift(lag)
    for lag in lags:
        df[f'{lag_col}_lag_{lag}'] = df[f'{lag_col}_lag_{lag}'].fillna(0) 
    return df

def load_and_preprocess_data(file_path, window_size=28, lags_to_create=[1, 7]):
    df = pd.read_csv(file_path)

    region_b_disaster_timestamps = df[df['region'] == 'Region_B'][df['disaster_flag'] == 1]['timestamp'].unique()
    df['is_region_b_shortage'] = df['timestamp'].isin(region_b_disaster_timestamps).astype(int)

    agg_dict = {
        'demand_kg': 'sum',
        'is_weekend': 'max',
        'holiday_flag': 'max',
        'promotion_flag': 'max',
        'disaster_flag': 'max',
        'is_region_b_shortage': 'max',
        'day_of_week': 'mean',
        'month': 'mean'
    }
    grouped_df_agg = df.groupby(['timestamp', 'region', 'food_type']).agg(agg_dict).reset_index()

    grouped_df_agg = create_lagged_features(grouped_df_agg, ['region', 'food_type'], 'demand_kg', lags_to_create)

    lag_feature_names = [f'demand_kg_lag_{lag}' for lag in lags_to_create]
    numerical_features = ['day_of_week', 'month'] + lag_feature_names
    binary_features = ['is_weekend', 'holiday_flag', 'promotion_flag', 'disaster_flag', 'is_region_b_shortage']
    
    feature_columns_for_x = numerical_features + binary_features

    all_X_train, all_y_train_scaled, all_X_test, all_y_test_scaled = [], [], [], []
    all_y_test_orig_for_eval = []

    feature_scalers_dict = {}
    target_scalers_dict = {}

    for (region, food_type), group_data in grouped_df_agg.groupby(['region', 'food_type']):
        group_data = group_data.sort_values('timestamp').reset_index(drop=True)
        
        group_data = group_data.dropna().reset_index(drop=True)

        if len(group_data) <= window_size + 1:
            print(f"Skipping group ({region}, {food_type}) due to insufficient data after lagging/dropna: {len(group_data)} rows")
            continue

        train_g_df, test_g_df = train_test_split(group_data, test_size=0.2, shuffle=False)

        if len(train_g_df) <= window_size or len(test_g_df) <= window_size:
            print(f"Skipping group ({region}, {food_type}) after split: insufficient data for sequences.")
            continue
            
        train_g_df_scaled_features = train_g_df.copy()
        test_g_df_scaled_features = test_g_df.copy()
        
        group_feature_scalers = {}
        for col in numerical_features:
            scaler = MinMaxScaler()
            train_g_df_scaled_features[col] = scaler.fit_transform(train_g_df[[col]])
            test_g_df_scaled_features[col] = scaler.transform(test_g_df[[col]])
            group_feature_scalers[col] = scaler
        feature_scalers_dict[(region, food_type)] = group_feature_scalers

        target_scaler = MinMaxScaler()
        train_g_y_scaled = target_scaler.fit_transform(train_g_df[['demand_kg']])
        test_g_y_scaled = target_scaler.transform(test_g_df[['demand_kg']])
        target_scalers_dict[(region, food_type)] = target_scaler
        
        train_g_df_scaled_features['demand_kg_scaled_target'] = train_g_y_scaled
        test_g_df_scaled_features['demand_kg_scaled_target'] = test_g_y_scaled

        X_train_g, y_train_g_scaled_seq = create_sequences(
            train_g_df_scaled_features, 'demand_kg_scaled_target',
            window_size, feature_columns_for_x
        )
        X_test_g, y_test_g_scaled_seq = create_sequences(
            test_g_df_scaled_features, 'demand_kg_scaled_target',
            window_size, feature_columns_for_x
        )
        
        _, y_test_orig_g_seq = create_sequences(
            test_g_df, 'demand_kg',
            window_size, feature_columns_for_x
        )

        if X_train_g.shape[0] > 0:
            all_X_train.append(X_train_g)
            all_y_train_scaled.append(y_train_g_scaled_seq)
        if X_test_g.shape[0] > 0:
            all_X_test.append(X_test_g)
            all_y_test_scaled.append(y_test_g_scaled_seq)
            all_y_test_orig_for_eval.append(y_test_orig_g_seq)

    if not all_X_train or not all_y_train_scaled:
        print("No training data generated after processing all groups.")
        return (np.empty((0, window_size, len(feature_columns_for_x))), np.empty(0),
                np.empty((0, window_size, len(feature_columns_for_x))), np.empty(0), np.empty(0),
                feature_scalers_dict, target_scalers_dict, grouped_df_agg)

    X_train = np.concatenate(all_X_train, axis=0)
    y_train_scaled = np.concatenate(all_y_train_scaled, axis=0)
    
    X_test = np.concatenate(all_X_test, axis=0) if all_X_test else np.empty((0, window_size, len(feature_columns_for_x)))
    y_test_scaled = np.concatenate(all_y_test_scaled, axis=0) if all_y_test_scaled else np.empty(0)
    y_test_orig_concatenated = np.concatenate(all_y_test_orig_for_eval, axis=0) if all_y_test_orig_for_eval else np.empty(0)
    
    return X_train, y_train_scaled, X_test, y_test_scaled, y_test_orig_concatenated, \
           feature_scalers_dict, target_scalers_dict, grouped_df_agg


def create_sequences(data_df, target_col_name, window_size, feature_cols):
    X, y = [], []
    y_values = data_df[target_col_name].values 
    
    for i in range(len(data_df) - window_size):
        if not all(col in data_df.columns for col in feature_cols):
            missing_cols = [col for col in feature_cols if col not in data_df.columns]
            raise ValueError(f"Missing columns in data_df: {missing_cols}")
        
        X.append(data_df[feature_cols].iloc[i:i + window_size].values)
        y.append(y_values[i + window_size]) 
    return np.array(X), np.array(y)


if __name__ == "__main__":
    file_path = "../../data/demand_prediction/cleaned_demand_data.csv" 
    (X_train, y_train_s, X_test, y_test_s, y_test_orig, 
     f_scalers, t_scalers, grouped) = load_and_preprocess_data(file_path)
    
    print(f"X_train shape: {X_train.shape}, y_train_scaled shape: {y_train_s.shape}")
    if X_test.shape[0] > 0:
        print(f"X_test shape: {X_test.shape}, y_test_scaled shape: {y_test_s.shape}, y_test_orig shape: {y_test_orig.shape}")
    else:
        print("X_test is empty.")
    
    print(f"Number of feature scaler groups: {len(f_scalers)}")
    print(f"Number of target scaler groups: {len(t_scalers)}")
    if f_scalers:
        first_group_key = list(f_scalers.keys())[0]
        print(f"Feature scalers for group {first_group_key}: {f_scalers[first_group_key].keys()}")
        print(f"Target scaler for group {first_group_key}: {t_scalers[first_group_key]}")
    print("Sample grouped (aggregated with lags) data:")
    print(grouped.head())
    print("Columns in grouped data:", grouped.columns.tolist())