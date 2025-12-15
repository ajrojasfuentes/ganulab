# ganulab/data/encode.py

"""
GANU-Lab Categorical Encoding Utility.
"""
from typing import Union, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from ganulab.data.scale import Scale

class Encode:
    def __init__(self, mode: str = "onehot", normalize_labels: bool = False):
        self.mode = mode.lower()
        self.normalize_labels = normalize_labels
        self.is_fitted = False
        
        if self.mode == "onehot":
            self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        elif self.mode == "label":
            self.encoder = LabelEncoder()
            self.scaler = Scale(n_quantiles=1000) if normalize_labels else None # Usamos Scale simplificado
        else:
            raise ValueError("Mode must be 'onehot' or 'label'.")

    def fit(self, data: Union[pd.Series, pd.DataFrame]):
        # Extraer valores numpy para sklearn
        values = data.values
        if self.mode == "label":
            if values.ndim > 1: values = values.ravel()
            self.encoder.fit(values)
            if self.scaler:
                # Hack temporal: Scale espera DF, le damos uno dummy
                transformed = self.encoder.transform(values)
                self.scaler.fit(pd.DataFrame(transformed))
        else:
            if values.ndim == 1: values = values.reshape(-1, 1)
            self.encoder.fit(values)
            
        self.is_fitted = True
        return self

    def transform(self, data: Union[pd.Series, pd.DataFrame]) -> np.ndarray:
        if not self.is_fitted: raise RuntimeError("Not fitted.")
        values = data.values
        
        if self.mode == "label":
            if values.ndim > 1: values = values.ravel()
            encoded = self.encoder.transform(values)
            if self.scaler:
                encoded = self.scaler.transform(pd.DataFrame(encoded)).values.ravel()
            return encoded
        else:
            if values.ndim == 1: values = values.reshape(-1, 1)
            return self.encoder.transform(values)