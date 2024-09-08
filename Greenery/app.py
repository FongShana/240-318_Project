from flask import Flask, render_template, request
import cv2
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the model (Ensure the path is correct)
MODEL_PATH = 'leaf_color_model.pkl'
model = joblib.load(MODEL_PATH)

# Function to extract color features from an image
def extract_color_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize the image to a fixed size
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist.flatten()  # Flatten the histogram to create a feature vector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']

    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded file temporarily
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Extract features and make a prediction
        features = extract_color_features(filepath).reshape(1, -1)
        prediction = model.predict(features)

        # Interpret the result
        if prediction == 0:
            result = 'Green Leaf'
        elif prediction == 1:
            result = 'Brown Leaf'
        else:
            result = 'Not a Leaf'

        # Clean up the file after prediction
        os.remove(filepath)

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    # Make sure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
