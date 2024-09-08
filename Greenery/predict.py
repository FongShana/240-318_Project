import cv2
import numpy as np
import joblib
import os

# Function to extract color features from an image
def extract_color_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize the image to a fixed size
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist.flatten()  # Flatten the histogram to create a feature vector

# Function to predict the color of a leaf or determine if it's not a leaf
def predict_leaf_color(image_path, model_path='leaf_color_model.pkl'):
    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: The model file {model_path} does not exist.")
    
    model = joblib.load(model_path)
    
    # Extract features from the image
    features = extract_color_features(image_path)
    
    # Reshape features to match the model's input
    features = features.reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Return the result
    if prediction == 0:
        return 'Green Leaf'
    elif prediction == 1:
        return 'Brown Leaf'
    else:
        return 'Not a Leaf'
