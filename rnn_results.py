import numpy as np
import pickle

# Load preprocessed data
X = np.load("amine/X_lstm.npy")
y = np.load("amine/y_lstm.npy")
with open("amine/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Check shapes and NaN values
print("X shape:", X.shape)  # Should be (7944, 7, 5)
print("y shape:", y.shape)  # Should be (7944,)
print("X dtype:", X.dtype)  # Should be float64
print("y dtype:", y.dtype)  # Should be float64
print("Any NaN in X:", np.isnan(X).any())
print("Any NaN in y:", np.isnan(y).any())

# Verify scenario features
print("Weekend flags in X:", np.unique(X[:, :, 1]))  # Should be [0, 1]
print("Disaster flags in X:", np.unique(X[:, :, 2]))  # Should be [0, 1]
print("Region_A flags in X:", np.unique(X[:, :, 3]))  # Should be [0, 1]
print("Region_B flags in X:", np.unique(X[:, :, 4]))  # Should be [0, 1]

# Check demand_kg range (normalized)
print("Demand_kg range in X:", X[:, :, 0].min(), X[:, :, 0].max())  # Should be [0, 1]
print("Demand_kg range in y:", y.min(), y.max())  # Should be [0, 1]

# Verify scaler
print("Scaler min:", scaler.data_min_)  # Should be ~10 kg
print("Scaler max:", scaler.data_max_)  # Should be ~100 kg