from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Load trained models
pca = joblib.load("models/pca_model.pkl")
model = joblib.load("models/softmax_model.pkl")

app = Flask("Vehicle Damage Detection App")

IMG_SIZE = (128, 128)

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            image_path = "uploaded.jpg"
            file.save(image_path)

            img = cv2.imread(image_path)
            img = cv2.resize(img, IMG_SIZE)
            features = extract_hog_features(img).reshape(1, -1)

            # Apply PCA
            features_pca = pca.transform(features)

            # Predict
            prediction = model.predict(features_pca)[0]
            label = "Accidental" if prediction == 0 else "Non-Accidental"

            return render_template("index.html", prediction=label)

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
