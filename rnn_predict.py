# predict.py

import argparse
import pandas as pd
from ui.dash_app.use_data import prepare_data
from rnn_train import make_windows, train_all_models

def predict_all(models, data_path, window_size, output_csv='predictions.csv'):
    """
    Génère et sauve predictions.csv pour chaque (food_type, region).
    """
    df, feature_cols = prepare_data(data_path)
    records = []

    for (ft, rg), grp in df.groupby(['food_type', 'region']):
        if len(grp) <= window_size or (ft, rg) not in models:
            continue
        X, _ = make_windows(grp, feature_cols, window_size)
        preds = models[(ft, rg)].predict(X).ravel()
        dates = grp['date'].iloc[window_size:].values
        for dt, p in zip(dates, preds):
            records.append({
                'food_type': ft,
                'region': rg,
                'date': dt,
                'prediction_kg': p
            })

    if not records:
        print("No predictions generated. Check if models were trained and window_size is correct.")
        return

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train+Predict LSTM")
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

    models = train_all_models(args.data_path, args.window)
    predict_all(models, args.data_path, args.window)
