import os # Make sure os is imported
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data, add_temporal_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_priority_feature(daily, original_df):
    # ... (function definition as before) ...
    try:
        # Ensure donation_date is datetime in original_df
        original_df['donation_date'] = pd.to_datetime(original_df['donation_date'], errors='coerce')

        # Group by donation_date and location
        priority_by_date = original_df.groupby([original_df['donation_date'].dt.date,
                                              'location'])['priority'].mean().reset_index()
        priority_by_date['donation_date'] = pd.to_datetime(priority_by_date['donation_date'])

        # Log column types and shapes
        logger.info(f"daily dtypes: {daily.dtypes}")
        logger.info(f"priority_by_date dtypes: {priority_by_date.dtypes}")
        logger.info(f"daily shape before merge: {daily.shape}")
        logger.info(f"priority_by_date shape: {priority_by_date.shape}")

        # Merge on donation_date and location
        daily = daily.merge(
            priority_by_date,
            left_on=['donation_date', 'location'],
            right_on=['donation_date', 'location'],
            how='left'
        )

        # Log shape after merge
        logger.info(f"daily shape after merge: {daily.shape}")

        # Check if priority column exists
        if 'priority' not in daily.columns:
            logger.warning("Priority column not added after merge. Creating default priority column.")
            daily['priority'] = 1.0  # Default priority value
        else:
            # Fill NaNs in priority
            daily['priority'] = daily['priority'].fillna(daily['priority'].mean())
            if daily['priority'].isnull().all():
                logger.warning("All priority values are NaN. Setting default priority to 1.0.")
                daily['priority'] = 1.0

        logger.info("Priority feature added successfully.")
        logger.info(f"daily columns after merge: {daily.columns.tolist()}")
        return daily
    except Exception as e:
        logger.error(f"Failed to add priority feature: {e}")
        raise

# --- Add the train_classifier function definition ---
def train_classifier(X_train, y_train, X_val, y_val):
    """Train a RandomForestClassifier for zero/non-zero prediction."""
    try:
        # Reshape data for RandomForestClassifier (needs 2D input)
        # Combine the window steps and features dimensions
        n_samples_train, window_size, n_features = X_train.shape
        X_train_reshaped = X_train.reshape((n_samples_train, window_size * n_features))

        n_samples_val, _, _ = X_val.shape
        X_val_reshaped = X_val.reshape((n_samples_val, window_size * n_features))

        logger.info(f"Training set class distribution: {np.bincount(y_train.astype(int))}")
        logger.info(f"Validation set class distribution: {np.bincount(y_val.astype(int))}")

        # Initialize and train the classifier
        # Use class_weight='balanced' to handle potential imbalance
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        clf.fit(X_train_reshaped, y_train)

        # Evaluate on validation set
        y_pred_val = clf.predict(X_val_reshaped)
        logger.info("Classification Report:\n" + classification_report(y_val, y_pred_val, zero_division=0))

        return clf
    except Exception as e:
        logger.error(f"Failed during classifier training: {e}")
        raise
# --- End of train_classifier function definition ---

# --- Define build_lstm_model (ensure it's defined) ---
def build_lstm_model(input_shape):
    """Build an LSTM model for quantity prediction."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
# --- End of build_lstm_model definition ---

# --- Define train_lstm_model (ensure it's defined) ---
def train_lstm_model(X_train, y_train, X_val, y_val, food_type):
    """Train an LSTM model for a specific food type."""
    try:
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

        checkpoint_filepath = f"models/demand_model_{food_type}.keras"
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=16,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )

        logger.info(f"Loading best model from {checkpoint_filepath}")
        best_model = load_model(checkpoint_filepath)

        return best_model, history
    except Exception as e:
        logger.error(f"Failed to train LSTM model for {food_type}: {e}")
        raise
# --- End of train_lstm_model definition ---


def main():
    """Main function to train the classifier and LSTM models."""
    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df = load_cleaned_data(path="amine/data/cleaned_data.csv")
        daily = aggregate_daily(df, by_type=False)
        daily_by_type = aggregate_daily(df, by_type=True)

        # Add priority feature for classification
        logger.info("Adding priority feature...")
        daily = add_priority_feature(daily, df) # Ensure this function runs correctly

        # --- Start: Classifier Training Logic ---
        logger.info("Preparing classification data...")
        window_clf = 7
        X_clf, y_clf, _ = prepare_classification_data(daily, window=window_clf, original_df=df) # Unpack 3, ignore dates

        if y_clf.size == 0:
            logger.error("prepare_classification_data returned empty y_clf. Cannot proceed.")
            raise ValueError("Cannot train classifier with empty data.")

        logger.info(f"y_clf class distribution before split: {np.bincount(y_clf.astype(int))}")

        stratify_param = y_clf if len(np.unique(y_clf)) >= 2 else None
        if len(X_clf) < 5:
             logger.error(f"Not enough samples ({len(X_clf)}) for classifier train/test split.")
             raise ValueError("Insufficient samples for classifier train/test split.")

        logger.info("Splitting data for classifier...")
        X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, shuffle=True, stratify=stratify_param, random_state=42
        )

        logger.info("Training classifier...")
        # Now this call should work
        classifier = train_classifier(X_train_clf, y_train_clf, X_val_clf, y_val_clf)

        # Save the newly trained classifier
        os.makedirs("models", exist_ok=True)
        with open('models/classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        logger.info("Classifier trained and saved to models/classifier.pkl")
        # --- End: Classifier Training Logic ---

        # --- Start: LSTM Training Logic ---
        # ... (rest of the main function as before) ...
        logger.info("Preparing data for LSTM models...")
        window_lstm = 7
        X_dict, y_dict, scaler_dict, _ = prepare_scaled_data(daily_by_type, window=window_lstm, by_type=True, original_df=df) # Unpack 4, ignore dates
        types = list(X_dict.keys())

        for t in types:
            logger.info(f"Training LSTM model for food type: {t}")
            if t not in X_dict or t not in y_dict or t not in scaler_dict:
                 logger.warning(f"Skipping type {t} due to missing data in dictionaries.")
                 continue
            if X_dict[t] is None or X_dict[t].shape[0] == 0:
                logger.warning(f"Skipping type {t} due to insufficient data for LSTM.")
                continue

            X, y, scaler = X_dict[t], y_dict[t], scaler_dict[t]

            if scaler is None:
                 logger.warning(f"Skipping type {t} due to missing scaler.")
                 continue

            if len(X) < 5:
                logger.warning(f"Skipping type {t}: Insufficient samples ({len(X)}) for LSTM train/test split.")
                continue

            logger.info(f"Splitting LSTM data for type: {t}")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                 logger.warning(f"Skipping type {t}: Train or validation set empty after split.")
                 continue

            model, history = train_lstm_model(X_train, y_train, X_val, y_val, t)

            with open(f'models/scaler_{t}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Best model for {t} saved by checkpoint; scaler saved.")
        # --- End: LSTM Training Logic ---

        logger.info("Training complete.")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()