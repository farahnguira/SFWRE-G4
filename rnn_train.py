import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
import json

from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.makedirs("docs", exist_ok=True)
os.makedirs("models", exist_ok=True)

def train_classifier(X_train, y_train, X_val, y_val, save_path, threshold_save_path):
    """Trains the RandomForestClassifier and finds the optimal threshold."""
    logger.info("Training RandomForestClassifier...")
    n_samples_train, window, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    X_train_reshaped = X_train.reshape((n_samples_train, window * n_features))
    X_val_reshaped = X_val.reshape((n_samples_val, window * n_features))

    expected_labels = [0, 1]
    classifier = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=5)
    classifier.fit(X_train_reshaped, y_train)

    y_pred_val_default = classifier.predict(X_val_reshaped)
    logger.info("Classifier Validation Report (Default Threshold 0.5):")
    logger.info("\n" + classification_report(y_val, y_pred_val_default, labels=expected_labels, zero_division=0))
    logger.info("Classifier Confusion Matrix (Default Threshold 0.5):")
    logger.info("\n" + str(confusion_matrix(y_val, y_pred_val_default, labels=expected_labels)))

    best_threshold = 0.5
    best_f1 = 0.0

    if hasattr(classifier, 'n_classes_') and classifier.n_classes_ > 1:
        logger.info("Finding optimal classification threshold...")
        y_pred_proba_val = classifier.predict_proba(X_val_reshaped)
        if y_pred_proba_val.shape[1] == 2:
            y_pred_proba_pos_class = y_pred_proba_val[:, 1]
            thresholds = np.arange(0.1, 0.91, 0.05)
            for threshold in thresholds:
                y_pred_binary = (y_pred_proba_pos_class >= threshold).astype(int)
                current_f1 = f1_score(y_val, y_pred_binary, pos_label=1, labels=expected_labels, zero_division=0)
                logger.debug(f"Threshold: {threshold:.2f}, F1-Score: {current_f1:.4f}")
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold
            logger.info(f"Optimal Threshold found: {best_threshold:.2f} with F1-Score: {best_f1:.4f}")
        else:
            logger.warning(f"predict_proba returned shape {y_pred_proba_val.shape}, expected 2 columns. Skipping threshold optimization.")
    else:
        single_class = classifier.classes_[0] if hasattr(classifier, 'classes_') and len(classifier.classes_) > 0 else 'unknown'
        logger.warning(f"Classifier trained on a single class ({single_class}). Skipping threshold optimization. Using default threshold {best_threshold}.")

    try:
        with open(threshold_save_path, 'w') as f:
            json.dump({'optimal_threshold': best_threshold}, f)
        logger.info(f"Optimal threshold saved to {threshold_save_path}")
    except Exception as e:
        logger.error(f"Failed to save optimal threshold: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier, f)
    logger.info(f"Classifier saved to {save_path}")
    return classifier, best_threshold

def build_lstm_model(input_shape):
    """Builds the Bidirectional LSTM model with MultiHeadAttention."""
    inputs = Input(shape=input_shape)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_out = Bidirectional(LSTM(32, return_sequences=True))(lstm_out)
    lstm_out = Dropout(0.3)(lstm_out)
    num_heads = 4
    key_dim = 64
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        query=lstm_out, value=lstm_out, key=lstm_out
    )
    condensed_output = GlobalAveragePooling1D()(attention_output)
    x = Dense(32, activation='relu')(condensed_output)
    x = Dropout(0.4)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss=Huber(delta=2.0), metrics=['mae'])
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, key, scaler):
    """Trains a single LSTM model for a group and saves it."""
    try:
        models_dir = "models"
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        model.summary(print_fn=logger.info)

        filename_key = "_".join(map(str, key)).replace(" ", "_").replace("'", "").replace(",", "")
        model_filename = f"demand_model_{filename_key}.keras"
        scaler_filename = f"scaler_{filename_key}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        scaler_path = os.path.join(models_dir, scaler_filename)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        best_model = load_model(model_path)
        val_loss, val_mae = best_model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Best Model Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

        # Baseline: 7-day moving average
        y_pred_scaled = best_model.predict(X_val).flatten()
        y_val_orig = scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_orig = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

        df_temp = pd.DataFrame({'y_val_orig': y_val_orig})
        df_temp['ma_baseline'] = df_temp['y_val_orig'].rolling(window=7, min_periods=1).mean().fillna(df_temp['y_val_orig'])
        ma_baseline = df_temp['ma_baseline'].values

        def calculate_metrics(actual, predicted):
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
            return mae, rmse, mape

        lstm_mae, lstm_rmse, lstm_mape = calculate_metrics(y_val_orig, y_pred_orig)
        ma_mae, ma_rmse, ma_mape = calculate_metrics(y_val_orig, ma_baseline)

        logger.info(f"Group {key} - LSTM Metrics: MAE={lstm_mae:.4f}, RMSE={lstm_rmse:.4f}, MAPE={lstm_mape:.2f}%")
        logger.info(f"Group {key} - Moving Average Baseline: MAE={ma_mae:.4f}, RMSE={ma_rmse:.4f}, MAPE={ma_mape:.2f}%")

        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler for {key} saved to {scaler_path}")
        return best_model, history
    except Exception as e:
        logger.error(f"Failed to train LSTM model for {key}: {e}", exc_info=True)
        raise

def main():
    """Main training pipeline."""
    try:
        window_size = 14  # Increased to capture longer patterns
        models_dir = "models"
        classifier_filename = "classifier.pkl"
        threshold_filename = "classifier_threshold.json"
        classifier_path = os.path.join(models_dir, classifier_filename)
        threshold_path = os.path.join(models_dir, threshold_filename)
        data_path = "data/demand_prediction/cleaned_demand_data.csv"

        logger.info("Starting training pipeline...")
        os.makedirs(models_dir, exist_ok=True)

        logger.info("Loading data...")
        df = load_cleaned_data(path=data_path)

        # Skip classifier training due to single-class issue
        logger.info("Skipping classifier training (all demands non-zero).")

        logger.info("Preparing data for LSTM models...")
        daily_lstm_agg = aggregate_daily(df, by_type=True, by_location=True)
        X_dict, y_dict, scaler_dict, timestamps_dict = prepare_scaled_data(
            daily_lstm_agg, window=window_size, by_type=True
        )

        if not X_dict:
            logger.error("No data prepared for LSTM models. Exiting.")
            return

        group_keys = list(X_dict.keys())
        for key in group_keys:
            logger.info(f"--- Starting training for group: {key} ---")
            X_scaled, y_scaled = X_dict[key], y_dict[key]
            scaler = scaler_dict[key]

            if X_scaled.shape[0] < 10:
                logger.warning(f"Skipping group {key}: Insufficient data samples ({X_scaled.shape[0]}).")
                continue

            logger.info(f"Splitting LSTM data for group: {key}")
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, test_size=0.2, shuffle=False
            )

            train_lstm_model(X_train, y_train, X_val, y_val, key, scaler)
            logger.info(f"--- Finished training for group: {key} ---")

        logger.info("Training pipeline complete.")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()