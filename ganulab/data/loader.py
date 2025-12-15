"""
GANU-Lab Data Loading Utilities.

This module simplifies the creation of PyTorch DataLoaders and provides
high-level pipelines to go from raw DataFrames to ready-to-train loaders
(Train/Val/Test) in a single step.
"""

import os
from typing import Optional, Union, Tuple, List, Any, Dict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import processing

# ==============================================================================
# 1. LOADER FACTORY
# ==============================================================================

def get_loader(
    features: Union[pd.DataFrame, np.ndarray, torch.Tensor, processing.NeuralDataset],
    targets: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Creates an optimized PyTorch DataLoader from raw data or a Dataset.

    Args:
        features: Input features or a pre-made NeuralDataset.
        targets: Target labels (optional if features is not a Dataset).
        batch_size: Samples per batch.
        shuffle: Whether to shuffle data (recommended for Training).
        num_workers: Number of subprocesses for data loading.
        device: Target device for the tensors ('cpu' or 'cuda').
        pin_memory: If True, copies tensors into CUDA pinned memory before transfer.
        drop_last: If True, drops the last incomplete batch.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    # 1. Resolve Dataset
    if isinstance(features, processing.NeuralDataset):
        dataset = features
        # If dataset is already on GPU, pin_memory/workers might be redundant/counter-productive
        if dataset.device != 'cpu':
            pin_memory = False
            num_workers = 0 
    else:
        # Create dataset on the fly
        # Strategy: Keep dataset on CPU RAM for the Loader to fetch, 
        # then move to GPU via pin_memory during iteration.
        dataset = processing.NeuralDataset(features, targets, device="cpu")

    # 2. Optimize settings based on environment
    if device == "cpu":
        pin_memory = False # No gain on CPU-only training
    
    # 3. Create Loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # Persistent workers speeds up training if num_workers > 0
        persistent_workers=(num_workers > 0) 
    )
    
    return loader

# ==============================================================================
# 2. HIGH-LEVEL PIPELINES (Raw -> Ready)
# ==============================================================================

def build_pipelines(
    data: pd.DataFrame,
    target_column: Optional[Union[str, List[str]]] = None,
    val_size: float = 0.1,
    test_size: float = 0.1,
    batch_size: int = 32,
    scaler_method: str = "minmax",
    time_series: bool = False,
    device: str = "cpu",
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, processing.DataScaler]:
    """
    Automates the entire data preparation pipeline:
    Split -> Scale (Fit on Train) -> Wrap in Datasets -> Create Loaders.

    Args:
        data: Raw DataFrame containing features and (optionally) targets.
        target_column: Name(s) of the target column(s). If None, it's Unsupervised.
        val_size: Proportion of validation set.
        test_size: Proportion of test set.
        batch_size: Batch size for loaders.
        scaler_method: 'minmax', 'standard', or 'quantile'.
        time_series: If True, splits without shuffling (for sequential data).
        device: Target device for training ('cuda'/'cpu').

    Returns:
        (train_loader, val_loader, test_loader, scaler)
    """
    # 1. Split Data
    train_df, val_df, test_df = processing.split_data(
        data, val_size, test_size, 
        shuffle=not time_series, 
        time_series=time_series, 
        random_state=random_state
    )

    # 2. Separate Features / Targets
    def split_xy(df):
        if target_column:
            y = df[target_column]
            X = df.drop(columns=target_column)
            return X, y
        return df, None

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    # 3. Fit Scaler (ONLY on Training Features)
    scaler = processing.DataScaler(method=scaler_method)
    
    # Scale Features
    X_train_norm = scaler.fit_transform(X_train)
    X_val_norm = scaler.transform(X_val)
    X_test_norm = scaler.transform(X_test)
    
    # Scale Targets? 
    # Usually better NOT to scale classification targets (integers).
    # For regression, user might want to scale manually or we assume raw targets.
    # Here we keep targets raw (e.g. for classification or physical values).
    
    # 4. Create Loaders
    # Train: Shuffle=True (unless time series)
    train_loader = get_loader(X_train_norm, y_train, batch_size, shuffle=not time_series, device=device)
    
    # Val/Test: Shuffle=False (Evaluation order usually matters less, but consistency is good)
    val_loader = get_loader(X_val_norm, y_val, batch_size, shuffle=False, device=device)
    test_loader = get_loader(X_test_norm, y_test, batch_size, shuffle=False, device=device)

    print(f"[Data] Pipeline Ready. Train: {len(X_train)} samples, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return train_loader, val_loader, test_loader, scaler