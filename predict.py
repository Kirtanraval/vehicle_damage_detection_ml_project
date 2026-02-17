import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load models
pca = joblib.load("models/pca_model.pkl")
model = joblib.load("models/softmax_model.pkl")

IMG_SIZE = (128, 128)

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

def predict_damage(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f" Error: Could not read image '{image_path}'. Check the file path.")
        return

    img = cv2.resize(img, IMG_SIZE)
    features = extract_hog_features(img).reshape(1, -1)

    # Apply PCA
    features_pca = pca.transform(features)

    # Predict
    prediction = model.predict(features_pca)[0]
    label = "Accidental" if prediction == 0 else "Non-Accidental"
    print(f"🔹 Prediction: {label}")

# Example Usage
predict_damage("C:/Users/RK/OneDrive/Desktop/vehicle_damage_detection/dataset/test/Nat/003.jpg")
