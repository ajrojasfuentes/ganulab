"""
GANU-Lab Dataset Wrapper.

Provides the 'NeuralDataset' class, a simplified PyTorch Dataset wrapper 
strictly for Pandas DataFrames. It handles direct conversion to Tensors 
and column selection.
"""

from __future__ import annotations
from typing import Union, Optional, List, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class NeuralDataset(Dataset):
    """
    PyTorch Dataset wrapper optimized for GANU-Lab pipelines.
    
    Assumption: Input 'data' is ALWAYS a numeric Pandas DataFrame.
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        features: Union[List[str], List[int]],
        targets: Optional[Union[List[str], List[int]]] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            data: Input DataFrame containing both features and targets (Numeric).
            features: List of column names (strings) or indices (ints) for X.
            targets: List of column names (strings) or indices (ints) for y. Optional.
            device: Target device for tensors ('cpu', 'cuda').
            dtype: Tensor data type (default: torch.float32).
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a Pandas DataFrame.")

        self.device = device
        
        # 1. Extract and Convert Features
        # We use .values to get numpy array directly, then to_tensor
        x_data = self._select_columns(data, features)
        self.x = torch.tensor(x_data.values, device=device, dtype=dtype)
        
        # 2. Extract and Convert Targets (if present)
        if targets is not None:
            y_data = self._select_columns(data, targets)
            self.y = torch.tensor(y_data.values, device=device, dtype=dtype)
            
            # Validation
            if len(self.x) != len(self.y):
                raise ValueError(f"Length mismatch: Features({len(self.x)}) != Targets({len(self.y)})")
        else:
            self.y = None

    def _select_columns(self, data: pd.DataFrame, selectors: List[Any]) -> pd.DataFrame:
        """
        Helper to slice the DataFrame based on selectors (names or indices).
        """
        if not selectors:
            return pd.DataFrame()

        first_sel = selectors[0]
        
        # Integer Indexing (iloc)
        if isinstance(first_sel, int):
            return data.iloc[:, selectors]
        
        # Label Indexing (loc/standard)
        return data[selectors]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.x[idx], self.y[idx]
        return self.x[idx]

    def to(self, device: str) -> "NeuralDataset":
        """
        Moves internal tensors to the specified device.
        """
        self.device = device
        self.x = self.x.to(device)
        if self.y is not None:
            self.y = self.y.to(device)
        return self
    
    def to_loader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """
        Convenience factory to generate a PyTorch DataLoader from this dataset.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)