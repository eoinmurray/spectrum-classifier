import os
import glob
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from joblib import dump
from scipy.signal import find_peaks

# Global parameters
MAX_PEAKS = 15          # Maximum number of peaks to use as features
PROMINENCE = 0.00001      # Prominence value for peak detection

def extract_features(energy, intensity):
    peaks, _ = find_peaks(intensity, prominence=PROMINENCE)
    if len(peaks) == 0:
        # If no peaks, pad with zeros.
        return [0] * (MAX_PEAKS * 2), []
    
    amplitudes = intensity[peaks]
    centers = energy[peaks]
    
    # Identify the highest peak.
    highest_peak_idx = np.argmax(amplitudes)
    main_peak_energy = centers[highest_peak_idx]
    
    # Calculate relative energies (binding energies) with respect to the highest peak.
    relative_centers = centers - main_peak_energy
    
    # Sort peaks by amplitude (highest first)
    sort_idx = np.argsort(amplitudes)[::-1]
    relative_centers_sorted = relative_centers[sort_idx]
    amplitudes_sorted = amplitudes[sort_idx]
    
    # Select the top MAX_PEAKS peaks.
    relative_top = relative_centers_sorted[:MAX_PEAKS]
    amplitudes_top = amplitudes_sorted[:MAX_PEAKS]
    
    # Pad with zeros if fewer than MAX_PEAKS are found.
    if len(relative_top) < MAX_PEAKS:
        pad_length = MAX_PEAKS - len(relative_top)
        relative_top = np.pad(relative_top, (0, pad_length), constant_values=0)
        amplitudes_top = np.pad(amplitudes_top, (0, pad_length), constant_values=0)

    # Combine into a flat feature vector: [relative_center1, amplitude1, ..., relative_center_MAX_PEAKS, amplitude_MAX_PEAKS]
    features = []
    for rel_center, amplitude in zip(relative_top, amplitudes_top):
        features.extend([rel_center, amplitude])

    return features, [main_peak_energy, relative_top, amplitudes_top]


def main():
    # Set the data directory for spectra and model saving
    spectra_dir = "datasets/converted-training"
    model_dir = "models"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filename = os.path.join(model_dir, "model-1.joblib")

    print(f"Reading spectra from {spectra_dir}")

    file_pattern = os.path.join(spectra_dir, "SQ-*.txt")
    files = glob.glob(file_pattern)

    # Lists to hold features and labels
    features_list = []
    labels_list = []

    # Process each file in the directory
    for file in files:
        # Extract the label from the filename using a regex.
        # For example, from "SQ-01_label_1_id_0D0FH7.txt" we extract "1".
        basename = os.path.basename(file)
        label_match = re.search(r"label_(\d+)", basename)
        if label_match:
            label = int(label_match.group(1))
        else:
            continue  # Skip file if no label is found.

        # Read the comma-delimited spectrum data (energy and intensity).
        try:
            data = np.loadtxt(file, delimiter=',', skiprows=1)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        # Ensure data is 2D (handle case of a single line).
        if data.ndim == 1:
            data = data.reshape(-1, 2)

        energy = data[:, 0]
        intensity = data[:, 1]

        # Extract features using the common function.
        features, _ = extract_features(energy, intensity)
        
        features_list.append(features)
        labels_list.append(label)

    # Convert lists to NumPy arrays for machine learning.
    X = np.array(features_list)
    y = np.array(labels_list)

    # Initialize the classifier.
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Define the cross-validation scheme.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate the classifier using cross-validation.
    cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    print("Cross-validation accuracy scores:", cv_scores)
    print("Mean accuracy: {:.2f}%".format(np.mean(cv_scores) * 100))

    # Generate cross-validated predictions and display a classification report.
    y_pred = cross_val_predict(clf, X, y, cv=cv)
    print("\nClassification Report:\n", classification_report(y, y_pred))

    # Retrain the classifier on the entire dataset for the final model.
    clf.fit(X, y)
    # Save the model as a joblib file
    dump(clf, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == "__main__":
    main()