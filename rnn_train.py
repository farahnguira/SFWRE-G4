import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import preprocessing functions
from ui.dash_app.use_data import load_cleaned_data, aggregate_daily, prepare_scaled_data, prepare_classification_data

def add_priority_feature(daily, original_df):
    """Add the priority feature to the daily DataFrame for classification."""
    try:
        priority_by_date = original_df.groupby(original_df['expiry_date'].dt.date)['priority'].mean().reset_index()
        daily = daily.merge(
            priority_by_date,
            left_on=daily['expiry_date'].dt.date,
            right_on='expiry_date',
            how='left'
        )
        daily['priority'] = daily['priority'].fillna(daily['priority'].mean())
        logger.info("Priority feature added successfully.")
        return daily
    except Exception as e:
        logger.error(f"Failed to add priority feature: {e}")
        raise

def train_classifier(X_train, y_train, X_val, y_val):
    """Train a RandomForestClassifier for zero/non-zero prediction."""
    try:
        # Log class distribution
        logger.info(f"Training set class distribution: {np.bincount(y_train.astype(int))}")
        logger.info(f"Validation set class distribution: {np.bincount(y_val.astype(int))}")
        
        # Check if both classes are present in training data
        if len(np.unique(y_train)) < 2:
            logger.warning("Training data contains only one class. Using a default classifier that predicts all non-zero.")
            # Create a dummy classifier that always predicts 1 (non-zero)
            class DummyClassifier:
                def fit(self, X, y):
                    return self
                def predict(self, X):
                    return np.ones(X.shape[0], dtype=int)
                def predict_proba(self, X):
                    # Return 0 probability for class 0, 1 for class 1
                    return np.column_stack((np.zeros(X.shape[0]), np.ones(X.shape[0])))
            classifier = DummyClassifier()
            classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            y_pred = classifier.predict(X_val.reshape(X_val.shape[0], -1))
            logger.info("Classification Report (dummy classifier):\n" + classification_report(y_val, y_pred))
        else:
            classifier = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: 3}, random_state=42)
            classifier.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            y_pred = classifier.predict(X_val.reshape(X_val.shape[0], -1))
            logger.info("Classification Report:\n" + classification_report(y_val, y_pred))
        return classifier
    except Exception as e:
        logger.error(f"Failed to train classifier: {e}")
        raise

def build_lstm_model(input_shape):
    """Build an LSTM model for quantity prediction."""
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.4),
        LSTM(64),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_lstm_model(X_train, y_train, X_val, y_val, food_type):
    """Train an LSTM model for a specific food type."""
    try:
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        checkpoint = ModelCheckpoint(
            filepath=f"models/demand_model_{food_type}.h5",
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
        return model, history
    except Exception as e:
        logger.error(f"Failed to train LSTM model for {food_type}: {e}")
        raise

def main():
    """Main function to train the classifier and LSTM models."""
    try:
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df = load_cleaned_data(path="amine/data/cleaned_data.csv")
        if 'donation_date' not in df.columns:
            logger.warning("donation_date column not found in cleaned_data.csv. Skipping days_until_expiry feature.")
        daily = aggregate_daily(df, by_type=False)
        daily_by_type = aggregate_daily(df, by_type=True)

        # Add priority feature for classification
        daily = add_priority_feature(daily, df)

        # Train classifier for zero/non-zero prediction
        logger.info("Training classifier...")
        X_clf, y_clf = prepare_classification_data(daily, window=7, original_df=df)
        # Log y_clf distribution
        logger.info(f"y_clf class distribution: {np.bincount(y_clf.astype(int))}")
        X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
            X_clf, y_clf, test_size=0.2, shuffle=True, stratify=y_clf, random_state=42
        )
        classifier = train_classifier(X_train_clf, y_train_clf, X_val_clf, y_val_clf)

        # Save the classifier
        os.makedirs("models", exist_ok=True)
        with open('models/classifier.pkl', 'wb') as f:
            pickle.dump(classifier, f)
        logger.info("Classifier saved to models/classifier.pkl")

        # Train LSTM models per food type
        logger.info("Preparing data for LSTM models...")
        X_dict, y_dict, scaler_dict = prepare_scaled_data(daily_by_type, window=7, by_type=True, original_df=df)
        types = list(X_dict.keys())

        for t in types:
            logger.info(f"Training LSTM model for food type: {t}")
            X, y, scaler = X_dict[t], y_dict[t], scaler_dict[t]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            model, history = train_lstm_model(X_train, y_train, X_val, y_val, t)

            # Save the model and scaler
            model.save(f"models/demand_model_{t}.h5")
            with open(f'models/scaler_{t}.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Model and scaler for {t} saved.")

        logger.info("Training complete; all models saved in models/ directory")

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()