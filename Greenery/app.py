from flask import Flask, render_template, request
import os
from predict import predict_leaf_color  # Import the prediction function from predict.py

app = Flask(__name__)

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

        # Use the predict.py function to make the prediction
        result = predict_leaf_color(filepath)

        # Clean up the file after prediction
        os.remove(filepath)

        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    # Make sure the uploads directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
