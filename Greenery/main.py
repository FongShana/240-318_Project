from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('leaf_features.csv', header=None)

# Split the data into features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Ensure balanced classes with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the balance of classes after splitting
unique_train, counts_train = np.unique(y_train, return_counts=True)
print(f"Training set class distribution: {dict(zip(unique_train, counts_train))}")
unique_test, counts_test = np.unique(y_test, return_counts=True)
print(f"Test set class distribution: {dict(zip(unique_test, counts_test))}")

# Initialize the model
model = LogisticRegression(max_iter=10000)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the model
joblib.dump(model, 'leaf_color_model.pkl')
