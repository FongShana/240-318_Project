import numpy as np
import cv2
import pandas as pd
import os

# Function to extract color features from an image
def extract_color_features(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize the image to a fixed size
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    hist = cv2.calcHist([image_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return hist.flatten()  # Flatten the histogram to create a feature vector

# Define paths to your images
image_paths = [
    r'Greenery\dataset\not_leaf\NotLeaf19.jpg'
]
labels = [2]  # 0 for green, 1 for brown, 2 for not a leaf

# Extract features and save them in a list
data = []
for i, image_path in enumerate(image_paths):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        continue
    features = extract_color_features(image_path)
    data.append(np.append(features, labels[i]))  # Append features and label

# Check if the CSV file exists
csv_file = 'leaf_features.csv'
if os.path.exists(csv_file):
    # Load existing data
    existing_data = pd.read_csv(csv_file, header=None)
    # Convert new data to DataFrame
    new_data_df = pd.DataFrame(data)
    # Append new data to existing data
    combined_data = pd.concat([existing_data, new_data_df], ignore_index=True)
else:
    # If CSV doesn't exist, create a new DataFrame
    combined_data = pd.DataFrame(data)

# Save the combined data back to the CSV
combined_data.to_csv(csv_file, index=False, header=False)

print("Features extracted and saved to leaf_features.csv")
