import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score # Added f1_score
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential, load_model, Model # Import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, MultiHeadAttention, GlobalAveragePooling1D # Changed Attention to MultiHeadAttention, added GlobalAveragePooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import json # Added json for saving threshold

# Correct import path for use_data - Updated to reflect new structure
from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data # Removed add_temporal_features, add_priority_feature

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directories exist
os.makedirs("docs", exist_ok=True)
# Removed data/demand_prediction creation, assuming it exists or clean script creates it
os.makedirs("models", exist_ok=True)

def train_classifier(X_train, y_train, X_val, y_val, save_path, threshold_save_path): # Added threshold_save_path
    """Trains the RandomForestClassifier and finds the optimal threshold."""
    logger.info("Training RandomForestClassifier...")
    # Reshape data for RandomForest (samples, features)
    n_samples_train, window, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    X_train_reshaped = X_train.reshape((n_samples_train, window * n_features))
    X_val_reshaped = X_val.reshape((n_samples_val, window * n_features))

    # Define expected labels
    expected_labels = [0, 1]

    # Initialize and train the classifier
    classifier = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=20, min_samples_split=5)
    classifier.fit(X_train_reshaped, y_train)

    # Evaluate on validation set (using default 0.5 threshold for initial report)
    y_pred_val_default = classifier.predict(X_val_reshaped)
    logger.info("Classifier Validation Report (Default Threshold 0.5):")
    # Add labels parameter
    logger.info("\n" + classification_report(y_val, y_pred_val_default, labels=expected_labels, zero_division=0))
    logger.info("Classifier Confusion Matrix (Default Threshold 0.5):")
    # Add labels parameter
    logger.info("\n" + str(confusion_matrix(y_val, y_pred_val_default, labels=expected_labels)))

    # --- Find Optimal Threshold based on F1-score ---
    best_threshold = 0.5 # Default threshold
    best_f1 = 0.0

    # Check if the classifier learned more than one class
    if hasattr(classifier, 'n_classes_') and classifier.n_classes_ > 1:
        logger.info("Finding optimal classification threshold...")
        y_pred_proba_val = classifier.predict_proba(X_val_reshaped)

        # Ensure probabilities for both classes are available
        if y_pred_proba_val.shape[1] == 2:
            y_pred_proba_pos_class = y_pred_proba_val[:, 1] # Probabilities for the positive class (1)
            thresholds = np.arange(0.1, 0.91, 0.05) # Test thresholds from 0.1 to 0.9

            for threshold in thresholds:
                y_pred_binary = (y_pred_proba_pos_class >= threshold).astype(int)
                # Specify pos_label=1 for f1_score if needed, and labels for consistency
                current_f1 = f1_score(y_val, y_pred_binary, pos_label=1, labels=expected_labels, zero_division=0)
                logger.debug(f"Threshold: {threshold:.2f}, F1-Score: {current_f1:.4f}")
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold
            logger.info(f"Optimal Threshold found: {best_threshold:.2f} with F1-Score: {best_f1:.4f}")
        else:
            # This case should ideally not happen if n_classes_ > 1, but as a safeguard:
            logger.warning(f"predict_proba returned shape {y_pred_proba_val.shape}, expected 2 columns. Skipping threshold optimization.")
            # Keep default threshold
    else:
        # Handle case where only one class was present in training data
        single_class = classifier.classes_[0] if hasattr(classifier, 'classes_') and len(classifier.classes_) > 0 else 'unknown'
        logger.warning(f"Classifier trained on a single class ({single_class}). Skipping threshold optimization. Using default threshold {best_threshold}.")
        # Optional: Set threshold based on the single class found, e.g., 0.0 if only class 0, 1.0 if only class 1.
        # However, sticking to 0.5 might be safer if prediction logic expects a standard threshold.

    # Save the optimal threshold
    try:
        with open(threshold_save_path, 'w') as f:
            json.dump({'optimal_threshold': best_threshold}, f)
        logger.info(f"Optimal threshold saved to {threshold_save_path}")
    except Exception as e:
        logger.error(f"Failed to save optimal threshold: {e}")
    # --- End Threshold Optimization ---


    # Save the trained classifier
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifier, f)
    logger.info(f"Classifier saved to {save_path}")
    return classifier, best_threshold # Return threshold as well


# Define model using Functional API
def build_lstm_model(input_shape):
    """Builds the Bidirectional LSTM model with MultiHeadAttention."""
    inputs = Input(shape=input_shape)

    # RNN layers
    # Ensure the output dimension of the last RNN layer is suitable for MHA key_dim or projection
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm_out = Dropout(0.3)(lstm_out)
    lstm_out = Bidirectional(LSTM(32, return_sequences=True))(lstm_out) # Output shape: (batch, seq_len, 64)
    lstm_out = Dropout(0.3)(lstm_out)

    # Multi-Head Attention (Self-Attention)
    # The input query, value, key are all the output from the LSTM layer
    num_heads = 4
    key_dim = 64 # Dimension for query/key/value, can be adjusted
    # Ensure key_dim is appropriate, often related to the LSTM output dimension (64 here)
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(
        query=lstm_out, value=lstm_out, key=lstm_out
    )
    # attention_output shape: (batch, seq_len, key_dim)

    # Condense the sequence output from attention before Dense layers
    # GlobalAveragePooling1D averages across the sequence length dimension
    condensed_output = GlobalAveragePooling1D()(attention_output) # Shape: (batch, key_dim)

    # Dense layers
    x = Dense(32, activation='relu')(condensed_output)
    x = Dropout(0.4)(x)
    outputs = Dense(1)(x) # Output layer for regression

    # Create the Model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm_model(X_train, y_train, X_val, y_val, key, scaler): # Added scaler parameter
    """Trains a single LSTM model for a group and saves it."""
    try:
        models_dir = "models"
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(input_shape)
        model.summary(print_fn=logger.info)

        # Sanitize key for filename
        # Ensure key components are strings before joining
        filename_key = "_".join(map(str, key)).replace(" ", "_").replace("'", "").replace(",", "")
        model_filename = f"demand_model_{filename_key}.keras"
        scaler_filename = f"scaler_{filename_key}.pkl" # Scaler filename
        model_path = os.path.join(models_dir, model_filename)
        scaler_path = os.path.join(models_dir, scaler_filename) # Scaler path

        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )

        # Load the best model saved by ModelCheckpoint
        best_model = load_model(model_path)
        val_loss, val_mae = best_model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Best Model Validation Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

        # --- Save the scaler associated with this model ---
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler for {key} saved to {scaler_path}")
        # --- End scaler saving ---

        return best_model, history # Return the best model and history

    except Exception as e:
        logger.error(f"Failed to train LSTM model for {key}: {e}", exc_info=True)
        raise # Re-raise the exception to stop the main loop if needed


def main():
    """Main training pipeline."""
    try:
        # --- Setup ---
        window_size = 14
        models_dir = "models"
        classifier_filename = "classifier.pkl"
        threshold_filename = "classifier_threshold.json" # Filename for threshold
        classifier_path = os.path.join(models_dir, classifier_filename)
        threshold_path = os.path.join(models_dir, threshold_filename) # Full path for threshold file
        data_path = "data/demand_prediction/cleaned_demand_data.csv" # Updated data path
        # --- End Setup ---

        logger.info("Starting training pipeline...")
        os.makedirs(models_dir, exist_ok=True)

        # --- Load Data ---
        logger.info("Loading data...")
        df = load_cleaned_data(path=data_path) # Use updated path

        # --- Prepare Data for Classifier ---
        logger.info("Preparing data for classifier...")
        # Aggregate by timestamp and region for classifier
        daily_clf_agg = aggregate_daily(df, by_type=False, by_location=True)
        # Removed call to add_priority_feature
        # Call prepare_classification_data without original_df
        X_clf, y_clf, _ = prepare_classification_data(daily_clf_agg, window=window_size)

        if X_clf.size == 0 or y_clf.size == 0:
            logger.error("No data generated for classifier. Exiting.")
            return

        # Log class distribution
        unique, counts = np.unique(y_clf, return_counts=True)
        logger.info(f"y_clf class distribution before split: {dict(zip(unique, counts))}")

        # Split classifier data (chronological)
        logger.info("Splitting data for classifier...")
        X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, shuffle=False
        )

        # --- Train Classifier ---
        logger.info("Training classifier...")
        # Pass threshold_path to the function
        _, optimal_threshold = train_classifier(X_train_clf, y_train_clf, X_val_clf, y_val_clf, classifier_path, threshold_path)
        logger.info(f"Classifier trained and saved to {classifier_path}. Optimal threshold {optimal_threshold:.2f} saved to {threshold_path}")


        # --- Prepare Data for LSTM ---
        logger.info("Preparing data for LSTM models...")
        # Aggregate by timestamp, type, and region for LSTM
        daily_lstm_agg = aggregate_daily(df, by_type=True, by_location=True)
        # Call prepare_scaled_data without original_df
        X_dict, y_dict, scaler_dict, timestamps_dict = prepare_scaled_data(
            daily_lstm_agg, window=window_size, by_type=True
        )

        if not X_dict:
            logger.error("No data prepared for LSTM models. Exiting.")
            return

        # --- Train LSTM Models per Group ---
        group_keys = list(X_dict.keys())
        for key in group_keys:
            logger.info(f"--- Starting training for group: {key} ---")

            X_scaled, y_scaled = X_dict[key], y_dict[key]
            scaler = scaler_dict[key] # Get the scaler for this group

            if X_scaled.shape[0] < 10:
                logger.warning(f"Skipping group {key}: Insufficient data samples ({X_scaled.shape[0]}).")
                continue

            # Split data chronologically
            logger.info(f"Splitting LSTM data for group: {key}")
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_scaled, test_size=0.2, shuffle=False
            )

            # Train the model (saves best via checkpoint)
            # Pass scaler to save it alongside the model
            train_lstm_model(X_train, y_train, X_val, y_val, key, scaler)

            logger.info(f"--- Finished training for group: {key} ---")

        logger.info("Training pipeline complete.")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()