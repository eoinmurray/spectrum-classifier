import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

# Global parameters
MAX_PEAKS = 10         # Maximum number of peaks to use as features
PROMINENCE = 0.01      # Prominence value for peak detection

def extract_features(energy, intensity):
    peaks, _ = find_peaks(intensity, prominence=PROMINENCE)
    
    if len(peaks) == 0:
        # If no peaks, pad with zeros.
        return [0] * (MAX_PEAKS * 2)
    
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

    plt.plot(energy - main_peak_energy, intensity)
    plt.plot(relative_centers_sorted, amplitudes_sorted, 'x')
    plt.show()
    
    
    return features
