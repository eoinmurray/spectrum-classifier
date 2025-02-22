from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from joblib import load
from extract_features import extract_features  # Import common logic

app = Flask(__name__)
CORS(app)

# Load the trained model from the "models" directory.
model = load("models/model-1.joblib")
def predict_spectrum_from_file(file_obj):
    """
    Accepts a file-like object containing spectrum data, loads the CSV data,
    extracts features using the extract_features function, and returns the predicted class.
    """
    try:
        # Attempt to load data assuming comma delimiter.
        data = np.loadtxt(file_obj, delimiter=',')
    except Exception as e:
        # Reset file pointer and try tab delimiter if comma fails.
        file_obj.seek(0)
        try:
            data = np.loadtxt(file_obj, delimiter='\t')
        except Exception as e2:
            return f"Error reading file: {e2}"
    
    # Ensure the data is 2D (handle single-line case)
    if data.ndim == 1:
        return "Error: Invalid data format: Expected 2D array"
    
    energy = data[:, 0]
    intensity = data[:, 1]

    features, _ = extract_features(energy, intensity)
    # Reshape features for prediction (model expects a 2D array)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    
    return f"Predicted class: {prediction[0]}"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a file to be uploaded with the key 'file' as form-data.
    Returns a JSON response with the predicted class or an error message.
    """
    
    print(request.files)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    # Process the file and get the prediction
    result = predict_spectrum_from_file(file)

    if result.startswith("Error"):
        return jsonify({'error': result}), 400

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
