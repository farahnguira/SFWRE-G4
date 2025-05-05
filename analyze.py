import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, recall_score

# Load per-type predictions
df = pd.read_csv("ui/dash_app/pred_combined.csv")
df['date'] = pd.to_datetime(df['date'])

# Compute metrics per type and location
metrics = df.groupby(['type', 'location']).apply(
    lambda x: pd.Series({
        'MAE': mean_absolute_error(x['actual'], x['predicted']),
        'RMSE': np.sqrt(mean_squared_error(x['actual'], x['predicted'])),
        'Recall': recall_score(x['actual'] > 0, x['predicted'] > 0, zero_division=0),
        'MAPE': np.mean(np.abs((x['actual'] - x['predicted']) / x['actual'].replace(0, np.nan))) * 100 if (x['actual'] > 0).any() else np.nan
    })
).reset_index()

print("Per-Type Metrics:")
print(metrics)

# Visualize trends for Tunis Center
plt.figure(figsize=(12, 6))
sns.lineplot(data=df[df['location'] == 'Tunis Center'], x="date", y="predicted", hue="type", style="type")
plt.title("Predicted Demand by Food Type (Tunis Center)")
plt.xlabel("Date")
plt.ylabel("Predicted Quantity (units)")
plt.xticks(rotation=45)
plt.legend(title="Food Type")
plt.tight_layout()
plt.savefig("ui/dash_app/type_specific_pred_tunis.png")
plt.show()

# Visualize actual vs predicted for each type (Tunis Center)
for food_type in df['type'].unique():
    plt.figure(figsize=(10, 5))
    type_df = df[(df['type'] == food_type) & (df['location'] == 'Tunis Center')]
    sns.lineplot(data=type_df, x="date", y="actual", label="Actual", color="blue", linestyle="--")
    sns.lineplot(data=type_df, x="date", y="predicted", label="Predicted", color="green")
    plt.title(f"Actual vs Predicted Demand for {food_type} (Tunis Center)")
    plt.xlabel("Date")
    plt.ylabel("Quantity (units)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ui/dash_app/actual_vs_pred_{food_type}_tunis.png")
    plt.show()

# Error distribution by type
plt.figure(figsize=(10, 6))
df['error'] = df['predicted'] - df['actual']
sns.boxplot(data=df[df['location'] == 'Tunis Center'], x="type", y="error")
plt.title("Prediction Error Distribution by Food Type (Tunis Center)")
plt.xlabel("Food Type")
plt.ylabel("Error (Predicted - Actual, units)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ui/dash_app/error_by_type_tunis.png")
plt.show()