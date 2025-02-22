import os
import glob
import random
import string
from typing import Any, Dict, Tuple, List
import numpy as np
import pandas as pd
import shutil

def generate_random_id(length: int = 6) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def load_lab_file(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} does not exist.")
    df = pd.read_csv(filepath, sep=r'\s+', header=None)
    
    # Wavelengths are assumed to be in the first column (skip the header row)
    wavelengths = df.iloc[1:, 0]
    
    # Calculate energy values: E = (h*c)/(λ*1e-9 * e)
    h, c, eV = 6.62607015e-34, 3e8, 1.60218e-19
    energy = (h * c) / ((wavelengths * 1e-9) * eV)
    
    # Extract power values from the first row (every other column starting at index 1)
    power_values = df.iloc[0, 1::2]
    # Drop every second column, and remove the first row and first column
    df = df.drop(df.columns[2::2], axis=1).drop(df.index[0]).drop(df.columns[0], axis=1)
    # Process intensity values: transpose, flip
    z = df.transpose()[::-1].reset_index(drop=True).iloc[::-1].values
    z = z[:, ::-1]
    
    assert z.shape[1] == energy.shape[0]
    
    filename = os.path.basename(filepath)
    power_values = power_values / np.max(power_values)
    
    return {
        "filename": filename,
        "id": generate_random_id(),
        "label": filename.split('.')[0],
        "intensity": z / np.max(z),
        "power_values": power_values,
        "energy": energy.values
    }

def process_files(files: List[str], label_prefix = "label2_#") -> List[Dict[str, Any]]:
    spectrums = []
    for filepath in files:

        label_files = glob.glob(os.path.join(os.path.dirname(filepath), f"{label_prefix}*"))
        if not label_files:
            continue
        label = os.path.basename(label_files[0]).replace(label_prefix, "").replace(".txt", "")
        
        # Extract the quantum_dot_id from two directories up
        quantum_dot_id = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        labfile = load_lab_file(filepath)

        for row in labfile["intensity"]:
            intensity = row
            uid = generate_random_id()
            assert intensity.shape[0] == labfile["energy"].shape[0]
            spectrum = {
                "quantum_dot_id": quantum_dot_id,
                "id": uid,
                "label": label,
                "intensity": intensity,
                "energy": labfile["energy"]
            }
            spectrums.append(spectrum)
    return spectrums

def convert_data(input_dir: str) -> pd.DataFrame:
    """
    Searches for lab files in input_dir and returns a DataFrame of spectra.
    """
    power_files = glob.glob(os.path.join(input_dir, "**", "*power*.dat"), recursive=True)
    rotator_files = glob.glob(os.path.join(input_dir, "**", "*rotator*.dat"), recursive=True)
    total_files = len(power_files) + len(rotator_files)
    print(f"Found {total_files} lab files in '{input_dir}'.")

    spectrums = process_files(power_files) + process_files(rotator_files)
    df = pd.DataFrame(spectrums)
    print("Conversion complete. DataFrame preview:")
    print(df.head())
    return df

def main():
    # --- Parameters (update these paths and settings as needed) ---
    INPUT_DIR = "./datasets/training-labfiles"  # Directory containing lab files
    OUTPUT_DIR = f"./datasets/converted-training/"
    GLOB_PATTERNS = ["**/*power*.dat", "**/*rotator*.dat"]
    # INPUT_DIR = "./datasets/validation-labfiles"  # Directory containing lab files
    # OUTPUT_DIR = f"./datasets/converted-validation/"
    # GLOB_PATTERNS = ["**/*single*.dat"]

    if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)
    else:
      if os.listdir(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

    print("Step 1: Converting lab files to spectra (in memory)...")

    # power_files = glob.glob(os.path.join(INPUT_DIR, "**", "*power*.dat"), recursive=True)
    # rotator_files = glob.glob(os.path.join(INPUT_DIR, "**", "*rotator*.dat"), recursive=True)
    
    files = []
    for pattern in GLOB_PATTERNS:
        files.extend(glob.glob(os.path.join(INPUT_DIR, pattern), recursive=True))
    
    total_files = len(files)
    print(f"Found {total_files} lab files in '{INPUT_DIR}'.")

    spectrums = process_files(files, "label2_#")
    converted_df = pd.DataFrame(spectrums)
    print("Conversion complete. DataFrame preview:")
    print(converted_df.sample(10))

    if converted_df.empty:
        print(f"No lab files found in '{INPUT_DIR}'. Exiting.")
        return

    print("Step 2: Saving spectra to disk...")
    for index, row in converted_df.iterrows():
        intensity = row["intensity"]
        energy = row["energy"]
        label = row["label"]
        quantum_dot_id = row["quantum_dot_id"]
        id = row["id"]

        data = np.column_stack((energy, intensity))
        np.savetxt(f"./{OUTPUT_DIR}/{quantum_dot_id}_label_{label}_id_{id}.txt", data, delimiter=",")
        
    print("Step 2: Conversion complete. Data saved to disk.")

if __name__ == "__main__":
    main()
