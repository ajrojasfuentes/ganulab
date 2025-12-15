# ganulab/data/processing.py

"""
GANU-Lab Data Processing Facade.
"""
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def compute_weights(y: pd.Series) -> np.ndarray:
    y_np = y.values.ravel()
    classes = np.unique(y_np)
    return compute_class_weight('balanced', classes=classes, y=y_np)