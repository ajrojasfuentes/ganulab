import torch
import pandas as pd
import numpy as np
from typing import Union, Optional

class Converter:
    def __init__(self):
        self.columns = None

    def to_numpy(self, data: Union[pd.DataFrame, torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Hub central: Convierte cualquier entrada a una vista NumPy (Shared Memory).
        """
        # CASO 1: DataFrame
        if isinstance(data, pd.DataFrame):
            # Guardamos columnas si es la primera vez o si estamos actualizando
            if self.columns is None or len(data.columns) == data.shape[1]:
                self.columns = data.columns
            return data.values # Devuelve vista si es homogéneo

        # CASO 2: Tensor
        elif isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy() # Comparte memoria si está en CPU

        # CASO 3: Ya es NumPy
        elif isinstance(data, np.ndarray):
            return data

        else:
            raise TypeError(f"Tipo {type(data)} no soportado.")

    def to_tensor(self, data: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Convierte la entrada a Tensor. 
        Mantiene memoria compartida SOLO si la entrada ya es float32.
        """
        # Paso 1: Obtener la vista numpy (shared memory si es posible)
        arr = self.to_numpy(data)
        
        # Paso 2: Crear el tensor
        # CRÍTICO: Si el array es float64 (default pandas), torch.from_numpy crea un DoubleTensor.
        # Si luego haces .float(), ROMPES la memoria compartida.
        # Asumimos que el usuario gestiona el tipo antes si quiere Zero-Copy estricto.
        tensor = torch.from_numpy(arr)
        
        # Advertencia silenciosa: Si el tensor resultante es Float64, PyTorch se quejará al entrenar.
        # Si quieres forzar float32 aquí manteniendo shared memory, el origen DEBE ser float32.
        return tensor

    def to_dataframe(self, data: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> pd.DataFrame:
        """
        Convierte la entrada a DataFrame usando las columnas guardadas.
        """
        # Paso 1: Obtener la vista numpy
        arr = self.to_numpy(data)

        # Validación de metadatos
        if self.columns is None:
             # Fallback si no hay columnas previas: crear genéricas
            cols = [f"col_{i}" for i in range(arr.shape[1])]
            # Opcional: raise ValueError("No hay columnas registradas. Pasa un DF primero.")
        else:
            if arr.shape[1] != len(self.columns):
                raise ValueError(f"Dimensión incorrecta: {arr.shape[1]} vs {len(self.columns)} cols guardadas.")
            cols = self.columns

        # Paso 2: Crear DataFrame
        # pd.DataFrame(arr, copy=False) intenta evitar la copia
        return pd.DataFrame(arr, columns=cols, copy=False)