import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from ui.dash_app.use_data import load_and_preprocess_data
import os
import pickle

def build_model(window_size, n_features):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(window_size, n_features)),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss=tf.keras.losses.Huber(), metrics=['mae'])
    return model

def train_model(file_path, window_size=28, epochs=100, batch_size=32):
    # Load and preprocess data
    # y_train_scaled and y_test_scaled are already scaled per group by load_and_preprocess_data
    # y_test_orig_for_eval is the original scale of the test targets for overall evaluation (not used directly in training)
    (X_train, y_train_scaled, X_test, y_test_scaled, _,
     feature_scalers_dict, target_scalers_dict, _) = load_and_preprocess_data(file_path, window_size)

    if X_train.shape[0] == 0 or y_train_scaled.shape[0] == 0:
        print("Error: No training data loaded. Aborting training.")
        return None, None, None, None
        
    # y_train_scaled is already prepared
    # y_test_scaled is already prepared for validation
    
    if X_test.shape[0] > 0 and y_test_scaled.shape[0] > 0:
        validation_data = (X_test, y_test_scaled)
    else:
        print("Warning: Test data is empty. Training without validation split from preprocessed data.")
        validation_data = None

    # Build model
    n_features = X_train.shape[2]
    model = build_model(window_size, n_features)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss' if validation_data else 'loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss' if validation_data else 'loss', patience=7, factor=0.2, min_lr=1e-6, verbose=1)
    
    print(f"Starting training with X_train shape: {X_train.shape}, y_train_scaled shape: {y_train_scaled.shape}")
    if validation_data:
        print(f"Validation data X_test shape: {X_test.shape}, y_test_scaled shape: {y_test_scaled.shape}")

    history = model.fit(
        X_train, y_train_scaled,
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_demand_model.h5")
    with open("models/feature_scalers_dict.pkl", "wb") as f:
        pickle.dump(feature_scalers_dict, f)
    with open("models/target_scalers_dict.pkl", "wb") as f:
        pickle.dump(target_scalers_dict, f)
    
    print("Model training complete.")
    print("Model saved to models/lstm_demand_model.h5")
    print("Feature scalers dictionary saved to models/feature_scalers_dict.pkl")
    print("Target scalers dictionary saved to models/target_scalers_dict.pkl")
    
    return model, feature_scalers_dict, target_scalers_dict, history

if __name__ == "__main__":
    # This script should ideally be run from the root of the SFWRE-G4 project
    # so that relative paths like "data/..." and "models/..." work correctly.
    file_path = "data/demand_prediction/cleaned_demand_data.csv"
    
    if not os.path.exists(file_path):
        print(f"{file_path} not found.")
        # Provide guidance on how to generate the file if it's missing
        print(f"Please ensure '{file_path}' exists. You might need to run data generation/cleaning scripts first.")
    else:
        model, f_scalers, t_scalers, history = train_model(file_path)
        if history:
            print("Sample history (last 5 epochs val_loss, if available):")
            if 'val_loss' in history.history:
                 print(history.history['val_loss'][-5:])
            elif 'loss' in history.history:
                 print(history.history['loss'][-5:])
            else:
                print("No loss or val_loss found in history.")