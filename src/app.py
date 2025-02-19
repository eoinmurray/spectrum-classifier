import gradio as gr
import numpy as np
from joblib import load
from extract_features import extract_features  # Import common logic

# Load the trained model from the "data" directory.
model = load("model.joblib")

def predict_spectrum(file_obj):
    """
    Accepts a file object, loads the CSV data, extracts features using the extract_features function,
    and returns the predicted class.
    """
    try:
        # Gradio passes a file object with a 'name' attribute.
        filepath = file_obj.name if hasattr(file_obj, "name") else file_obj
        data = np.loadtxt(filepath, delimiter=',')
    except Exception as e:
        return f"Error reading file: {e}"

    # Ensure the data is 2D (handle single-line case)
    if data.ndim == 1:
        raise ValueError("Invalid data format: Expected 2D array")

    energy = data[:, 0]
    intensity = data[:, 1]

    features = extract_features(energy, intensity)
    # Reshape features for prediction (model expects 2D array)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return f"Predicted class: {prediction[0]}"

# Create the Gradio interface.
interface = gr.Interface(
    fn=predict_spectrum,
    inputs=gr.File(label="Upload Spectrum File (CSV with two columns)"),
    outputs="text",
    title="Spectrum Classifier",
    description="Upload a comma-delimited spectrum file containing energy and intensity columns."
)

interface.launch()
