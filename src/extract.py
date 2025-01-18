import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from type_defs import Peak
import pywt
import pywt.data

def extract_peaks(z, prominence, max_peaks=15):
    if z.size != 2044:
        raise ValueError("z must have 2044 points.")
    
    x = np.arange(z.size)
    percentage_prominence = prominence * z.max()
    peak_idxs, _ = find_peaks(z, prominence=percentage_prominence)
    
    if not peak_idxs.size:
        return [Peak(center=0, amplitude=0, fwhm=0).to_dict()] * max_peaks

    peaks = []
    for idx in peak_idxs:
        amplitude = z[idx]
        half_max = amplitude / 2
        
        if idx > 0:
            left_x = np.interp(half_max, z[:idx+1][::-1], x[:idx+1][::-1])
        else:
            left_x = x[idx]

        if idx < len(z) - 1:
            right_x = np.interp(half_max, z[idx:], x[idx:])
        else:
            right_x = x[idx]
        fwhm = right_x - left_x
        
        peaks.append(Peak(center=x[idx], amplitude=amplitude, fwhm=fwhm).to_dict())

    return sorted(peaks, key=lambda p: p['amplitude'], reverse=True)[:max_peaks]

def calculate_skew(peaks):
    max_peak = max(peaks, key=lambda p: p['amplitude'])
    left_count = sum(p['center'] < max_peak['center'] for p in peaks)
    right_count = sum(p['center'] > max_peak['center'] for p in peaks)

    skew = 0
    if abs(left_count - right_count) > 1:
        skew = -1 if left_count > right_count else 1

    return skew

def flatten_peak_features(row, max_peaks=15, features=["center", "amplitude", "fwhm"]):
    processed_peaks = []
    for peak in row["peaks"][:max_peaks]:
        processed_peaks.extend([peak.get(feature, 0) for feature in features])
    processed_peaks.extend([0] * (max_peaks * len(features) - len(processed_peaks)))
    return processed_peaks

def smooth_z(z):
    cA, cD = pywt.dwt(z, 'db1')  # Decompose using the wavelet transform
    cD = np.zeros_like(cD)       # Zero out detail coefficients to remove noise
    smoothed = pywt.idwt(cA, cD, 'db1')  # Reconstruct the signal
    return smoothed

def main(input_file: str, output_dir: str, prominence: float = 0.1, max_peaks: int = 15):
    if not os.path.isfile(input_file):
      return print(f"Input file {input_file} does not exist.")
  
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_json(input_file)
    
    print(f"Processing {input_file} with {len(df)} spectra")

    df['z'] = df['z'].map(np.array)#.map(smooth_z)
    df['peaks'] = df['z'].map(lambda z: extract_peaks(z, prominence, max_peaks))
    df['skew'] = df['peaks'].map(lambda peaks: calculate_skew(peaks))
    df = df[(df['power'] >= 4.0) | (df['power'].isna())]

    df.to_json(os.path.join(output_dir, "stats.json"), orient="records", indent=4)
    print(f"Saved {len(df)} spectra to stats.json")
    
    print("df of stats.json:")
    print(df)

    peak_data = df.apply(flatten_peak_features, axis=1)
    peak_columns = [
        f"peak_{i}_{feature}"
        for i in range(1, 16)
        for feature in ["center", "amplitude", "fwhm"]
    ]
    peak_df = pd.DataFrame(peak_data.tolist(), columns=peak_columns)

    print("df of training.json:")
    print(peak_df)
    
    training_df = df[["id", "qd_id", "target_label", "power", "rotator", "skew"]].join(peak_df)

    training_df.to_json(os.path.join(output_dir, "training.json"), orient="records", indent=4)
    print(f"Saved {len(training_df)} spectra to training.json")
