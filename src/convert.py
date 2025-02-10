import os
import glob
import random
import string
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import argparse

def generate_random_id(length: int = 6) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def load_lab_file(filepath: str, shape: Tuple[int, int]) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")
    df = pd.read_csv(filepath, sep=r'\s+', header=None)
    
    wavelengths = df.iloc[1:, 0]
    
    # wavelengths = wavelengths.iloc[1:]
    
    h, c, eV = 6.62607015e-34, 3e8, 1.60218e-19
    energy = (h * c) / ((wavelengths * 1e-9) * eV)
    power_values = df.iloc[0, 1::2]
    df = df.drop(df.columns[2::2], axis=1).drop(df.index[0]).drop(df.columns[0], axis=1)
    z = df.transpose()[::-1].reset_index(drop=True).iloc[::-1].values[:shape[0], :]
    z = z[:, ::-1]
    
    assert(z.shape[1] == energy.shape[0])
    
    filename = os.path.basename(filepath)
    power_values = power_values[:shape[0]] / np.max(power_values)
    return {
        "shape": shape,
        "filename": filename,
        "id": generate_random_id(),
        "label": filename.split('.')[0],
        "intensity": z / np.max(z),
        "power_values": power_values,
        "energy_values": energy.values
    }

def process_files(files, shape=(23, 2046)):
    spectrums = []
    for filepath in files:
        label_prefix = "label2_#"
        label_files = glob.glob(os.path.join(os.path.dirname(filepath), f"{label_prefix}*"))
        if not label_files:
            continue
        label = os.path.basename(label_files[0]).replace(label_prefix, "").replace(".txt", "")
        
        qd_id = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        labfile = load_lab_file(filepath, shape)

        for row in labfile["intensity"]:
            intensity = row
            uid = generate_random_id()

            assert(intensity.shape[0] == labfile["energy_values"].shape[0])

            spectrum = {
              "qd_id": qd_id, 
              "id": uid, 
              "target_label": label, 
              "intensity": intensity,
              "energy_values": labfile["energy_values"]
            }
            
            spectrums.append(spectrum)
    return spectrums

def main(input_dir: str, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    power_files = glob.glob(os.path.join(input_dir, "**", "*power*.dat"), recursive=True)
    rotator_files = glob.glob(os.path.join(input_dir, "**", "*rotator*.dat"), recursive=True)
    print(f"Found {len(power_files) + len(rotator_files)} files in {input_dir}.")

    spectrums = process_files(power_files) + process_files(rotator_files)
    df = pd.DataFrame([s for s in spectrums])
    
    print(df)
    
    df.to_json(output_file, orient="records", indent=4)
    print(f"Saved {len(spectrums)} spectrums to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str, help="Input directory with lab files")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
