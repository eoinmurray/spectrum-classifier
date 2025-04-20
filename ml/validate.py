import os
import re
import numpy as np
import pandas as pd
from joblib import load
from train import extract_features  # Import common logic

# Load the trained model from the "models" directory.
model = load("models/model-1.joblib")

def extract_ground_truth(filename):
    """
    Extract the ground truth label from the filename.
    Assumes filename format contains '_label_{label}_id_'.
    For example: "Square 5_label_1_id_YnaJBO.txt" will extract "1".
    """
    match = re.search(r'_label_(.*?)_id_', filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Could not extract label from filename: {filename}")

def predict_spectrum(filepath):
    """
    Loads TXT data from the given file path, extracts features using the extract_features function,
    and returns the predicted class.
    """
    try:
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {e}")

    # Ensure the data is 2D (handle single-line case)
    if data.ndim == 1:
        raise ValueError(f"Invalid data format in file {filepath}: Expected 2D array")

    energy = data[:, 0]
    intensity = data[:, 1]

    features, _ = extract_features(energy, intensity)
    # Reshape features for prediction (model expects a 2D array)
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    return prediction[0]

def main():
    validation_dir = "datasets/converted-validation2"
    records = []

    # Iterate over all TXT files in the validation directory.
    for file_name in os.listdir(validation_dir):
        file_path = os.path.join(validation_dir, file_name)
        if os.path.isfile(file_path) and file_name.lower().endswith('.txt'):
            try:
                # Extract the ground truth label from the file name.
                ground_truth = extract_ground_truth(file_name)
                # Get the prediction from the model.
                predicted = predict_spectrum(file_path)
                match_str = "MATCH" if str(ground_truth) == str(predicted) else "MISMATCH"
                # Append the record.
                records.append({
                    "filename": file_name,
                    "ground_truth": ground_truth,
                    "prediction": predicted,
                    "match": match_str
                })
            except Exception as e:
                # Record error information for debugging.
                records.append({
                    "filename": file_name,
                    "ground_truth": None,
                    "prediction": None,
                    "match": f"ERROR: {e}"
                })

    # Create a DataFrame from the records and print it.
    df = pd.DataFrame(records)
    
    # Print statistics per ground_truth
    stats_per_ground_truth = df.groupby('ground_truth')['match'].value_counts(normalize=True).unstack().fillna(0) * 100
    print("Statistics per ground_truth (as percentage):")
    print(stats_per_ground_truth)
    

if __name__ == "__main__":
    main()
