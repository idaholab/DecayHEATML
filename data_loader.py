import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any

class DataLoader:
    """Load and preprocess decay heat data from pickle files"""

    def __init__(self, pickle_path: str):
        self.pickle_path = pickle_path
        self.loaded_cases = None
        self.df = None
        self.time_steps = None

    def load_and_preprocess(self, normalization_factor: float = None) -> Tuple[Dict[int, Any], pd.DataFrame]:
        """Main loading method with optional normalization"""
        self._load_raw_data()
        self._add_time_steps()
        self._create_dataframe(normalization_factor)
        return self.loaded_cases, self.df

    def _load_raw_data(self):
        with open(self.pickle_path, 'rb') as handle:
            self.loaded_cases = pickle.load(handle)

    def _add_time_steps(self):
        steps = 100
        end_time = 10e4 * 365
        self.time_steps = list(np.logspace(1e-4, np.log10(end_time), steps) - 1)
        for case in self.loaded_cases.values():
            case['time_steps'] = self.time_steps

    def _create_dataframe(self, normalization_factor):
        rows = []
        for case_id, case_data in self.loaded_cases.items():
            for t, heat in zip(np.log(case_data['time_steps']),
                                np.log(case_data['decay_heat'])):
                rows.append({
                    'time': t,
                    'power_density': case_data['power_density'],
                    'humidity_content': case_data['humidity_content'],
                    'air_ingress': case_data['air_ingress'],
                    'decay_heat': heat
                })
        self.df = pd.DataFrame(rows)

        if normalization_factor:
            self._apply_normalization(normalization_factor)

    def _apply_normalization(self, factor):
        """Normalize concentration values for Cl-based data"""
        for col in [c for c in self.df.columns if 'Concentration' in c]:
            self.df[col] /= factor
