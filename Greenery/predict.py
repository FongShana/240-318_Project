import cv2
import numpy as np
import joblib
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to extract color features from an image
def extract_color_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize the image to a fixed size
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist.flatten()  # Flatten the histogram to create a feature vector

# Function to predict the color of a leaf
def predict_leaf_color(model_path='leaf_color_model.pkl'):
    # Open a file dialog to select an image file
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    image_path = askopenfilename(title="Select Leaf Image",
                                 filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])

    # Check if the image file was selected
    if not image_path:
        print("No image file selected.")
        return

    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Error: The model file {model_path} does not exist.")
        return
    
    model = joblib.load(model_path)
    
    # Extract features from the selected image
    features = extract_color_features(image_path)
    
    # Reshape features to match the model's input
    features = features.reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(features)
    
    # Print the result
    if prediction == 0:
        result = 'Green Leaf'
    elif prediction == 1:
        result = 'Brown Leaf'
    else:
        result = 'Not a Leaf'
    
    print(f'The image is predicted to be: {result}')

# Run the prediction function
predict_leaf_color()
