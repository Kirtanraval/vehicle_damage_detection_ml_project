import os
import cv2
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from skimage.feature import hog

# Dataset path
DATASET_PATH = "C:/Users/RK/OneDrive/Desktop/vehicle_damage_detection/dataset"

# Image processing parameters
IMG_SIZE = (128, 128)

# Function to extract HOG features
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Loading dataset
def load_dataset(folder):
    X, y = [], []
    for label, category in enumerate(["Acc", "Nat"]):  # Acc = Accidental, Nat = Non-Accidental
        folder_path = os.path.join(DATASET_PATH, folder, category)
        if not os.path.exists(folder_path):
            print(f"⚠️ Warning: Folder {folder_path} does not exist.")
            continue

        for file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue  

            img = cv2.resize(img, IMG_SIZE)
            features = extract_hog_features(img)
            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# Load training dataset
X_train, y_train = load_dataset("train")

# Apply PCA for dimensionality reduction
n_components = min(500, min(X_train.shape))  
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train)

# Train Softmax Classifier
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_train_pca, y_train)

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(pca, "models/pca_model.pkl")
joblib.dump(model, "models/softmax_model.pkl")

print(" Model trained and saved successfully!")
