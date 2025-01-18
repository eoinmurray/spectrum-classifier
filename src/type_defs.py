import json
import os
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Optional, List
from dataclasses import dataclass, field, asdict
import random
import string

@dataclass
class Peak:
    center: Optional[int] = 0.0
    amplitude: Optional[float] = 0.0
    fwhm: Optional[float] = 0.0

    def to_dict(self) -> Dict[str, Any]:
        def convert(val):
            if isinstance(val, np.integer):
                return int(val)
            elif isinstance(val, np.floating):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, pd.Series):
                return val.to_list()
            elif isinstance(val, dict):
                return {k: convert(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [convert(x) for x in val]
            elif isinstance(val, tuple):
                return [convert(x) for x in val]
            return val

        return convert(asdict(self))


@dataclass
class Spectrum:
    qd_id: Optional[str] = None
    id: Optional[str] = None
    target_label: Optional[str] = None
    z: Optional[np.ndarray] = field(default=None)
    skew: Optional[int] = None
    power: Optional[float] = None
    rotator: Optional[float] = None
    peaks: Optional[List[Peak]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        def convert(val):
            if isinstance(val, np.integer):
                return int(val)
            elif isinstance(val, np.floating):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, pd.Series):
                return val.to_list()
            elif isinstance(val, dict):
                return {k: convert(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [convert(x) for x in val]
            elif isinstance(val, tuple):
                return [convert(x) for x in val]
            return val

        return convert(asdict(self))

    def save(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            s = json.dumps(self.to_dict(), indent=4)
            f.write(s)
        print(f"Saved spectrum data to {output_path}")

    @classmethod
    def load(cls, filepath: str) -> 'Spectrum':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The file {filepath} does not exist.")
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert arrays back to np.array if needed
        array_keys = ['z']
        for key in array_keys:
            if data.get(key) is not None:
                data[key] = np.array(data[key])

        return cls(**data)

@dataclass
class LabFile:
    shape: Tuple[int, int]
    filename: Optional[str] = None
    id: Optional[str] = None
    label: Optional[str] = None
    output_filename: Optional[str] = None
    output_image_filename: Optional[str] = None
    intensity: Optional[np.ndarray] = field(default=None)
    power_values: Optional[np.ndarray] = field(default=None)
    energy_values: Optional[np.ndarray] = field(default=None)
    peaks: Optional[np.ndarray] = field(default=None)
    peak_parameters: Optional[np.ndarray] = field(default=None)
    fitted_intensity: Optional[np.ndarray] = field(default=None)
    initial_guess: Optional[np.ndarray] = field(default=None)

    @classmethod
    def from_lab_file(self, filepath: str, shape: Tuple[int,int]) -> 'LabFile':
        if not os.path.exists(filepath):
            print(f"The file {filepath} does not exist.")
            raise FileNotFoundError(f"The file {filepath} does not exist.")

        df = pd.read_csv(filepath, sep=r'\s+', header=None)
        wavelengths = df.iloc[1:, 0]
        h, c, eV = 6.62607015e-34, 3.0e8, 1.60218e-19
        energy = (h * c) / ((wavelengths * 1e-9) * eV)

        power_values = df.iloc[0, 1::2]
        df = df.drop(df.columns[2::2], axis=1).drop(df.index[0]).drop(df.columns[0], axis=1)
        z = df.transpose()[::-1].reset_index(drop=True).iloc[::-1].values
        z = z[:shape[0], :]
        
        z = z[:, ::-1]

        filename = os.path.basename(filepath)
        power_values = power_values[:shape[0]] / np.max(power_values)
        
        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    
        return self(
            shape=shape,
            filename=filename,
            id=random_id,
            label=filename.split('.')[0],
            intensity=z / np.max(z),
            fitted_intensity=np.zeros(shape),
            power_values=power_values,
            energy_values=energy.values
        )
