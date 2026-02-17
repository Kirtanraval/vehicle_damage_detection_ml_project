import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Training Data
TRAIN_DATA_PATH = "C:/Users/RK/OneDrive/Desktop/vehicle_damage_detection/data/X_train.npy"
try:
    X_train = np.load(TRAIN_DATA_PATH)
except FileNotFoundError:
    print(f"❌ Training data not found at {TRAIN_DATA_PATH}! Run `train_model.py` first.")
    exit()

# Train the Scaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the Scaler
SCALER_PATH = "C:/Users/RK/OneDrive/Desktop/vehicle_damage_detection/models/scaler.pkl"
joblib.dump(scaler, SCALER_PATH)
print(f"🎉 Scaler saved successfully at: {SCALER_PATH}")
