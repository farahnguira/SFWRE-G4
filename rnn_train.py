# rnn_train.py

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from ui.dash_app.use_data import prepare_data

def make_windows(data, feats, window_size):
    """
    Construit X (shape=(n_samples, window_size, n_feats)) et y (n_samples,)
    à partir du DataFrame déjà trié.
    """
    X, y = [], []
    arr = data[feats + ['demand_kg']].values
    for i in range(len(arr) - window_size):
        X.append(arr[i:i+window_size, :-1])
        y.append(arr[i+window_size, -1])
    return np.array(X), np.array(y)

def train_all_models(data_path, window_size):
    """
    Entraîne un LSTM 64→32 pour chaque groupe (food_type, region).
    Renvoie un dict { (ft, rg): model }.
    """
    df, feature_cols = prepare_data(data_path)
    counts = df.groupby(['food_type','region']).size()
    print("Données disponibles par groupe (jours) :")
    print(counts, "\n")

    models_dict = {}
    for (ft, rg), grp in df.groupby(['food_type', 'region']):
        n = len(grp)
        if n <= window_size:
            print(f"WARNING: Skipping {ft},{rg} (only {n} days < window+1={window_size+1})")
            continue

        X, y = make_windows(grp, feature_cols, window_size)
        split = int(0.8 * len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]

        inp = layers.Input(shape=(window_size, X_tr.shape[2]))
        x = layers.LSTM(64, return_sequences=True)(inp)
        x = layers.LSTM(32)(x)
        out = layers.Dense(1)(x)
        model = models.Model(inp, out)
        model.compile(optimizer='adam', loss='mse')

        es = tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)
        model.fit(
            X_tr, y_tr,
            epochs=200,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[es],
            verbose=2
        )

        # Affichage des métriques de validation
        preds = model.predict(X_val).ravel()
        mae  = np.mean(np.abs(preds - y_val))
        rmse = np.sqrt(np.mean((preds - y_val)**2))
        mape = np.mean(np.abs((preds - y_val) / y_val)) * 100
        print(f"{ft},{rg}  ->  MAE={mae:.1f}, RMSE={rmse:.1f}, MAPE={mape:.1f}%\n")

        models_dict[(ft, rg)] = model

    return models_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraîne un LSTM par (food_type, region)."
    )
    parser.add_argument(
        'data_path',
        nargs='?',
        default='data/demand_prediction/cleaned_demand_data.csv',
        help='Chemin vers cleaned_demand_data.csv (défaut %(default)s)'
    )
    parser.add_argument(
        '-w', '--window',
        type=int,
        default=28,
        help='Taille de la fenêtre temporelle (par défaut %(default)s)'
    )
    args = parser.parse_args()

    train_all_models(args.data_path, args.window)
