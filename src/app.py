import gradio as gr
import numpy as np
from joblib import load
from extract_features import extract_features  # Import common logic

# Load the trained model from the "data" directory.
model = load("models/model.joblib")

def predict_spectrum(file_obj):
  """
  Accepts a file object, loads the CSV data, extracts features using the extract_features function,
  and returns the predicted class.
  """
  try:
    # Gradio passes a file object with a 'name' attribute.
    filepath = file_obj.name if hasattr(file_obj, "name") else file_obj
    try:
      data = np.loadtxt(filepath, delimiter=',')
    except ValueError:
      data = np.loadtxt(filepath, delimiter='\t')
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

def second_page_function(input_text):
  """
  A simple function for the second page that echoes the input text.
  """
  return f"You entered: {input_text}"

# Create the Gradio interface with tabs.
with gr.Blocks() as demo:
  with gr.Tab("Spectrum Classifier"):
    gr.Markdown("## Spectrum Classifier")
    gr.Markdown("Upload a comma-delimited spectrum file containing energy and intensity columns.")
    file_input = gr.File(label="Upload Spectrum File (CSV with two columns)")
    output_text = gr.Textbox()
    file_input.change(predict_spectrum, inputs=file_input, outputs=output_text)
  
  with gr.Tab("Second Page"):
    gr.Markdown("## Second Page")
    gr.Markdown("This is the second page of the Gradio app.")
    text_input = gr.Textbox(label="Enter some text")
    text_output = gr.Textbox()
    text_input.change(second_page_function, inputs=text_input, outputs=text_output)

demo.launch()

