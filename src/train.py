import os
import glob
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report
from joblib import dump
from extract_features import extract_features  # Import common logic

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
        data = np.loadtxt(file, delimiter=',')
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

dump(clf, model_filename)
print(f"Model saved to {model_filename}")
