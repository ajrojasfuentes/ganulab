# ganulab/data/scale.py

"""
GANU-Lab Data Normalization Utility.

Provides a simplified wrapper around QuantileTransformer strictly for Pandas DataFrames.
"""

from __future__ import annotations
from typing import List, Optional, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

class Scale:
    """
    Normalizador numérico exclusivo para Pandas DataFrames.
    Preserva nombres de columnas e índices.
    """

    def __init__(
        self,
        n_quantiles: int = 1000,
        output_distribution: str = "normal",
        random_state: int = 42,
        subsample: int = int(1e9),
    ):
        self.scaler = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            random_state=random_state,
            subsample=subsample,
        )
        self.feature_names_in_: List[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "Scale":
        """Ajusta el scaler a los datos."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        self.feature_names_in_ = list(df.columns)
        
        # Ajuste dinámico de n_quantiles
        if self.scaler.n_quantiles > len(df):
            self.scaler.n_quantiles = len(df)

        self.scaler.fit(df.values)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica la normalización y devuelve un DataFrame con la misma estructura."""
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted.")
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame.")

        # Verificar columnas
        missing = set(self.feature_names_in_) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in input: {missing}")

        # Ordenar columnas para coincidir con fit
        df = df[self.feature_names_in_]
        
        values_scaled = self.scaler.transform(df.values)
        return pd.DataFrame(values_scaled, columns=df.columns, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # Nota: Eliminamos inverse_transform complejo. 
    # Si se requiere invertir, se asume que se pasa un array (predicción del modelo)
    # o un DataFrame con las mismas columnas que el original.