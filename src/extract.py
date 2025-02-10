#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from typing import List, Dict

def extract_peaks(z: np.ndarray, energy_values: np.ndarray, prominence: float, max_peaks: int = 15) -> List[Dict[str, float]]:
    assert(z.size == energy_values.size)
    x = energy_values
    
    perc_prom = prominence * z.max()
    idxs, _ = find_peaks(z, prominence=perc_prom)
    if not idxs.size:
        return [{'center': 0, 'amplitude': 0, 'fwhm': 0}] * max_peaks

    peaks = []
    for idx in idxs:
        amplitude = float(z[idx])
        # half_max = amplitude / 2
        # left_x = np.interp(half_max, z[:idx+1][::-1], x[:idx+1][::-1]) if idx > 0 else x[idx]
        # right_x = np.interp(half_max, z[idx:], x[idx:]) if idx < len(z) - 1 else x[idx]
        peaks.append({
          'center': x[idx], 
          'amplitude': amplitude, 
          # 'fwhm': float(right_x - left_x)
        })
    return sorted(peaks, key=lambda p: p['amplitude'], reverse=True)[:max_peaks]

def flatten_peak_features(row: pd.Series, max_peaks: int = 15, features: List[str] = ["center", "amplitude"]) -> List[float]:
    """Flatten peak features from a row into a single list."""
    feats = [peak.get(f, 0) for peak in row["peaks"][:max_peaks] for f in features]
    padding_length = max_peaks * len(features) - len(feats)
    if padding_length > 0:
        feats.extend([0] * padding_length)
    return feats

def main(input_file: str, output_file: str, prominence: float = 0.1, max_peaks: int = 15, smooth: bool = False) -> None:
    """Process a JSON file of spectra, extract peak features, and save the training data."""
    if not os.path.isfile(input_file):
        sys.exit(f"Input file {input_file} does not exist.")

    try:
        df = pd.read_json(input_file)
    except ValueError as e:
        sys.exit(f"Error reading JSON: {e}")

    print(f"Processing {len(df)} spectra from {input_file}")

    df['intensity'] = df['intensity'].map(np.array)
    df['energy_values'] = df['energy_values'].map(np.array)

    df['peaks'] = df.apply(lambda row: extract_peaks(row['intensity'], row['energy_values'], prominence, max_peaks), axis=1)
    
    peak_data = df.apply(flatten_peak_features, axis=1)

    cols = [f"peak_{i}_{feat}" for i in range(1, max_peaks + 1) for feat in ["center", "amplitude"]]
    peak_df = pd.DataFrame(peak_data.tolist(), columns=cols)

    required_cols = ["id", "qd_id", "target_label"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        sys.exit(f"Missing required columns in input JSON: {missing}")

    training_df = df[required_cols].join(peak_df)
    training_df.to_json(output_file, orient="records", indent=4)
    print(training_df)
    
    label_counts = training_df['target_label'].value_counts()
    print("Counts of unique target labels:")
    print(label_counts)
    
    print(f"Saved {len(training_df)} spectra to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process spectra JSON and extract peak features.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Output JSON file path")
    parser.add_argument("--prominence", type=float, default=0.1, help="Prominence factor for peak detection")
    parser.add_argument("--max_peaks", type=int, default=15, help="Max number of peaks to extract")
    parser.add_argument("--smooth", action="store_true", help="Smooth intensity data before peak extraction")
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.prominence, args.max_peaks, args.smooth)
