# ganulab/data/split.py

"""
GANU-Lab Data Splitting Utility.

Provides strategy-based splitting strictly for Pandas DataFrames.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd

class Split:
    """
    Utilidad para segmentar DataFrames.
    Soporta estrategias: 'random', 'identity', 'radial'.
    """

    # --- 1. Random Split ---
    @staticmethod
    def random(df: pd.DataFrame, plane: tuple[str, str], n_segments: int, random_state=None):
        x_col = plane[0]
        y_col = plane[1]

        if n_segments <= 0:
            raise ValueError("n_segments debe ser >= 1")

        if x_col not in df.columns or y_col not in df.columns:
            raise KeyError(f"Las columnas '{x_col}' y/o '{y_col}' no existen en el DataFrame.")

        n = len(df)
        if n_segments > n:
            raise ValueError("n_segments no puede ser mayor que el número de filas del DataFrame.")

        # Mezclar filas pseudoaleatoriamente
        df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Partir los índices en n bloques casi iguales
        indices = np.arange(n)
        chunks = np.array_split(indices, n_segments)

        # Construir cada segmento
        segments = [
            df_shuffled.iloc[chunk].copy().reset_index(drop=True)
            for chunk in chunks if len(chunk) > 0
        ]

        return segments

    # --- 2. Identity Split ---
    @staticmethod
    def identity(df: pd.DataFrame, plane: tuple[str, str], n_segments: int):

        x_col = plane[0]
        y_col = plane[1]

        # Proyección sobre la dirección perpendicular a y = x
        d = df[y_col] - df[x_col]

        # Asigna a cada fila un índice de segmento según cuantiles de d
        bands = pd.qcut(d, q=n_segments, labels=False, duplicates='drop')

        # Construye una lista de DataFrames, uno por banda
        segments = [df[bands == i].copy() for i in range(bands.min(), bands.max() + 1)]

        return segments

    # --- 3. Radial Split ---
    @staticmethod
    def radial(df: pd.DataFrame, plane: tuple[str, str], n_segments: int, center=(0.0, 0.0)):

        x_col = plane[0]
        y_col = plane[1]

        cx, cy = center

        # Radio euclidiano de cada punto respecto al centro
        r = np.sqrt((df[x_col] - cx)**2 + (df[y_col] - cy)**2)

        # Dividimos el radio en n cuantiles → segmentos más o menos balanceados
        rings = pd.qcut(r, q=n_segments, labels=False, duplicates='drop')

        segments = [
            df[rings == i].copy()
            for i in range(rings.min(), rings.max() + 1)
        ]

        return segments