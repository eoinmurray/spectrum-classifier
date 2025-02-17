import numpy as np
from scipy.signal import find_peaks

# Global parameters
MAX_PEAKS = 10         # Maximum number of peaks to use as features
PROMINENCE = 0.01      # Prominence value for peak detection

def extract_features(energy, intensity):
    """
    Detect peaks in the intensity data using the global PROMINENCE and return a feature vector consisting
    of the centers (energy positions) and amplitudes (intensity values) of the top peaks.

    The peaks are sorted by amplitude in descending order (highest amplitude first).
    If fewer than MAX_PEAKS peaks are found, the feature vector is padded with zeros.

    The output feature vector has the form:
        [center1, amplitude1, center2, amplitude2, ..., center_MAX_PEAKS, amplitude_MAX_PEAKS]
    """
    # Detect peaks using the global PROMINENCE
    peaks, _ = find_peaks(intensity, prominence=PROMINENCE)

    if len(peaks) > 0:
        amplitudes = intensity[peaks]
        centers = energy[peaks]
        # Sort peaks by amplitude (highest first)
        sort_idx = np.argsort(amplitudes)[::-1]
        centers_sorted = centers[sort_idx]
        amplitudes_sorted = amplitudes[sort_idx]
        centers_top = centers_sorted[:MAX_PEAKS]
        amplitudes_top = amplitudes_sorted[:MAX_PEAKS]
    else:
        centers_top = np.array([])
        amplitudes_top = np.array([])

    # Pad with zeros if fewer than MAX_PEAKS are found.
    if len(centers_top) < MAX_PEAKS:
        pad_length = MAX_PEAKS - len(centers_top)
        centers_top = np.pad(centers_top, (0, pad_length), constant_values=0)
        amplitudes_top = np.pad(amplitudes_top, (0, pad_length), constant_values=0)

    # Create a flat feature vector: [center1, amplitude1, center2, amplitude2, ...]
    features = []
    for center, amplitude in zip(centers_top, amplitudes_top):
        features.extend([center, amplitude])
    return features
