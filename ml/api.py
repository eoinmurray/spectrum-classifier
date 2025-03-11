from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from joblib import load
from train import extract_features  # Import common logic

# Configure the Flask app to serve static files from the "dist" directory at "/"
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
    print("Data loaded with comma delimiter")
  except Exception as e:
    # Reset file pointer and try tab delimiter if comma fails.
    print(f"Failed to load data with comma delimiter: {e}")
    file_obj.seek(0)
    try:
      data = np.loadtxt(file_obj, delimiter='\t')
      print("Data loaded with tab delimiter")
    except Exception as e2:
      print(f"Failed to load data with tab delimiter: {e2}")
      return f"Error reading file: {e2}"
  
  # Ensure the data is 2D (handle single-line case)
  if data.ndim == 1:
    print("Error: Invalid data format: Expected 2D array")
    return "Error: Invalid data format: Expected 2D array"
  
  energy = data[:, 0]
  intensity = data[:, 1]

  features, _ = extract_features(energy, intensity)
  print(f"Extracted features: {features}")
  # Reshape features for prediction (model expects a 2D array)
  features_array = np.array(features).reshape(1, -1)
  prediction = model.predict(features_array)
  print(f"Prediction: {prediction[0]}")
  
  return f"{prediction[0]}"

@app.route('/', methods=['GET'])
def index():
  print("Received GET request at /api")
  return jsonify({'message': 'Hello, world!'})

@app.route('/api', methods=['GET'])
def api_index():
  print("Received GET request at /api")
  return jsonify({'message': 'Hello, world!'})

@app.route('/api/predict', methods=['POST'])
def predict():
  """
  Expects either a file to be uploaded with the key 'file' as form-data,
  or a JSON object with 'energy' and 'intensity' arrays.
  Returns a JSON response with the predicted class or an error message.
  """
  print("Received POST request at /api/predict")
  
  data = request.get_json()
  if 'energy' not in data or 'intensity' not in data:
    print('JSON must contain "energy" and "intensity" arrays')
    return jsonify({'error': 'JSON must contain "energy" and "intensity" arrays'}), 400

  energy = np.array(data['energy'])
  intensity = np.array(data['intensity'])

  if energy.shape != intensity.shape:
    print("Energy and intensity arrays must have the same shape")
    return jsonify({'error': 'Energy and intensity arrays must have the same shape'}), 400

  features, _ = extract_features(energy, intensity)
  [main_peak_energy, relative_top, amplitudes_top] = _
  # print(f"Extracted features from JSON: {features}")
  
  if all(f == 0 for f in features):
    print("Extracted features are all zeros")
    return jsonify({'error': 'Extracted features are all zeros'}), 400
  
  features_array = np.array(features).reshape(1, -1)
  prediction = model.predict(features_array)
  print(f"JSON prediction result: {prediction[0]}")

  return jsonify({
    'prediction': float(prediction[0]),
    'main_peak_energy': float(main_peak_energy),
    'peak_centers': relative_top.tolist(),
    'peak_amplitudes': amplitudes_top.tolist()
  })


if __name__ == '__main__':
  print("Starting Flask app on port 3001")
  app.run(debug=True, host="0.0.0.0", port=3001)
