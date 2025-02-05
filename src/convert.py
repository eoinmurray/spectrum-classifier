import pandas as pd
import os
import glob
import random
import string
from type_defs import Spectrum, LabFile

def generate_random_id(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def process_files(files, limit, spectrum_type, shape=(23, 2046), max_col=2044):
    spectrum_array = []
    for filepath in files[:limit]:
        label_prefix = "label2_#"
        label_filename = glob.glob(os.path.join(os.path.dirname(filepath), f"{label_prefix}*"))[0]
        label = os.path.basename(label_filename).replace(label_prefix, "").replace(".txt", "")
        
        qd_id = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        labfile = LabFile.from_lab_file(filepath, shape=shape)
        # label = generate_random_id()

        for index, row in enumerate(labfile.intensity):
          truncated_row = row[:max_col]
          uid = generate_random_id()
          args = {
              "z": truncated_row,
              "id": uid,
              "qd_id": qd_id,
              "target_label": label
          }
          args["power" if spectrum_type == "power" else "rotator"] = index
          spec = Spectrum(**args)
          spectrum_array.append(spec)
    return spectrum_array

def main(input_dir: str, output_file: str, limit: int):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    power_files = glob.glob(os.path.join(input_dir, "**", "*power*.dat"), recursive=True)
    rotator_files = glob.glob(os.path.join(input_dir, "**", "*rotator*.dat"), recursive=True)
    print(f"Found {len(power_files) + len(rotator_files)} files in {input_dir}.")
    power_spectrum_array = process_files(power_files, limit, "power")
    rotator_spectrum_array = process_files(rotator_files, limit, "rotator")
    
    spectrums = power_spectrum_array + rotator_spectrum_array
    dicts = [spectrum.to_dict() for spectrum in spectrums]
    df = pd.DataFrame(dicts)
    print(df)
    df.to_json(output_file, orient="records", indent=4)
    print(f"Saved {len(spectrums)} spectrums to {output_file}.")