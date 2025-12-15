# ganulab/utils/io.py

"""
GANU-Lab Input/Output Utilities.

This module provides unified interfaces for loading and saving data across various
formats (CSV, Parquet, ROOT, Excel, PyTorch Tensors) and managing neural network 
checkpoints. It handles optimization details such as engine selection (PyArrow) 
and separator detection automatically.
"""

import os
import csv
import sys
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ------------------------------------------------------------------------------
# Optional Dependencies
# ------------------------------------------------------------------------------

# Support for ROOT files (High Energy Physics)
try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

# Support for Progress Bars
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Fallback if tqdm is missing to prevent crashes
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def update(self, n=1): pass
        def set_description(self, desc): pass
        def __iter__(self): return iter(self.iterable) if self.iterable else iter([])

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

COMMON_DELIMITERS = [
    ",", "\t", ";", "|", " ", ":",
    "~", "^", "#", "@", "$", "%",
    "&", "*", "+", "-", "=", "/",
    "\\", "_", "!"
]

# ==============================================================================
# 0. HELPERS
# ==============================================================================

def _detect_separator(path: Path, encoding: str = 'utf-8', num_lines: int = 5) -> str:
    """
    Heuristically detects the delimiter of a text file by analyzing the first few lines.

    Args:
        path (Path): Path to the input file.
        encoding (str): File encoding. Defaults to 'utf-8'.
        num_lines (int): Number of lines to sample. Defaults to 5.

    Returns:
        str: The detected delimiter (defaults to ',' if detection fails).
    """
    try:
        with open(path, 'r', encoding=encoding, errors='ignore') as f:
            sample_lines = [f.readline() for _ in range(num_lines)]
            sample = "".join(sample_lines)
            
            # Strategy 1: Python's csv.Sniffer
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=COMMON_DELIMITERS)
                return dialect.delimiter
            except csv.Error:
                pass

            # Strategy 2: Frequency analysis
            candidates = []
            for char in COMMON_DELIMITERS:
                counts = [line.count(char) for line in sample_lines if line.strip()]
                # Check consistency: delimiter must appear in all sampled non-empty lines
                if len(counts) > 0 and counts[0] > 0 and all(c == counts[0] for c in counts):
                    candidates.append((char, counts[0]))
            
            if candidates:
                # Sort by frequency (descending)
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
            
            return ","
    except Exception:
        return ","

def to_tensor(
    data: Union[pd.DataFrame, pd.Series, np.ndarray, List, Dict], 
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Robustly converts various data structures into PyTorch Tensors.

    Args:
        data: Input data (DataFrame, Array, List, or Dict).
        device: Target device (e.g., 'cpu', 'cuda').
        dtype: Target data type. Defaults to torch.float32.

    Returns:
        Union[torch.Tensor, Dict]: The converted tensor or dictionary of tensors.
    
    Raises:
        ValueError: If data contains non-numeric types.
        TypeError: If conversion fails.
    """
    # 1. Passthrough if already a tensor
    if isinstance(data, torch.Tensor):
        return data.to(device=device, dtype=dtype)

    # 2. Recursive handling for Dictionaries
    if isinstance(data, dict):
        return {k: to_tensor(v, device, dtype) for k, v in data.items()}

    # 3. Extract values from Pandas objects
    if isinstance(data, (pd.DataFrame, pd.Series)):
        values = data.values
    else:
        values = data

    try:
        # Handle Python lists by converting to numpy first
        if isinstance(values, list):
            values = np.array(values)
        
        # Verify numeric type safety
        if values.dtype.kind not in 'iufb': # int, uint, float, bool
             raise ValueError("Data contains non-numeric types.")

        tensor = torch.from_numpy(values).to(dtype=dtype)
        
        if device:
            tensor = tensor.to(device)
            
        return tensor
        
    except Exception as e:
        # Last resort fallback
        try:
            return torch.tensor(values, device=device, dtype=dtype)
        except Exception:
            raise TypeError(f"Failed to convert data to Tensor: {str(e)}")

# ==============================================================================
# 1. GENERIC LOADERS
# ==============================================================================

def load_file(
    path: Union[str, Path], 
    columns: Optional[List[str]] = None, 
    as_tensor: bool = False,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    verbose: bool = False,
    **kwargs
) -> Union[pd.DataFrame, torch.Tensor, Dict[str, Any], Any]:
    """
    Loads a single file with automatic format detection.
    
    Supported formats: .csv, .tsv, .txt, .xlsx, .parquet, .root, .pt, .pth

    Args:
        path (Union[str, Path]): File path.
        columns (List[str], optional): Specific columns/keys to load.
        as_tensor (bool): If True, converts the result to a PyTorch Tensor.
        device (str): Target device if as_tensor is True.
        dtype (torch.dtype): Target dtype if as_tensor is True.
        verbose (bool): If True, prints loading status.
        **kwargs: Additional arguments passed to the underlying loader.

    Returns:
        The loaded data (DataFrame, Tensor, or Object).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    data = None
    
    if verbose:
        print(f"[IO] Loading file: {path.name} ...")

    try:
        # --- Text Formats (CSV, TSV, TXT) ---
        if ext in [".csv", ".tsv", ".txt"]:
            # Detect separator
            sep = kwargs.pop('sep', kwargs.pop('delimiter', None))
            if sep is None: 
                sep = _detect_separator(path)
            
            # Optimize engine selection (Prefer PyArrow > C > Python)
            engine = kwargs.pop('engine', 'auto')
            if engine == 'auto':
                try:
                    import pyarrow
                    engine = "pyarrow" 
                except ImportError:
                    engine = "c"
            
            # Construct read options dynamically
            read_opts = {
                'filepath_or_buffer': path,
                'usecols': columns,
                'sep': sep,
                'engine': engine,
            }
            # 'low_memory' arg is only valid for 'c' engine
            if engine == 'c': 
                read_opts['low_memory'] = False
            
            read_opts.update(kwargs)
            
            data = pd.read_csv(**read_opts)
            
        # --- Excel ---
        elif ext in [".xlsx", ".xls"]:
            data = pd.read_excel(path, usecols=columns, **kwargs)
            
        # --- Parquet ---
        elif ext == ".parquet":
            data = pd.read_parquet(path, columns=columns, **kwargs)
            
        # --- ROOT (HEP) ---
        elif ext == ".root":
            if not HAS_UPROOT: 
                raise ImportError("Package 'uproot' is required to read .root files.")
            tree_name = kwargs.get("tree_name", "tree")
            with uproot.open(path) as f:
                data = f[tree_name].arrays(columns, library="pd")
        
        # --- PyTorch Binary ---
        elif ext in [".pt", ".pth"]:
            raw_data = torch.load(path, map_location=device, **kwargs)
            # Filter dictionary keys if 'columns' is provided
            if columns is not None and isinstance(raw_data, dict):
                data = {k: v for k, v in raw_data.items() if k in columns}
            else:
                data = raw_data
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        # --- Final Conversion ---
        if as_tensor:
            return to_tensor(data, device=device, dtype=dtype)
            
        return data
            
    except Exception as e:
        raise RuntimeError(f"Critical error loading {path}: {str(e)}") from e

def load_dataset(
    input_path: Union[str, Path], 
    extension: Optional[str] = None,
    columns: Optional[List[str]] = None,
    as_tensor: bool = False,
    device: str = "cpu",
    recursive: bool = False,
    verbose: bool = True,
    **kwargs
) -> Union[pd.DataFrame, torch.Tensor, List[Any]]:
    """
    Loads and concatenates multiple files from a directory.

    Args:
        input_path (Union[str, Path]): Directory path.
        extension (str, optional): File extension filter (e.g., '.csv').
        columns (List[str], optional): Columns to load.
        as_tensor (bool): If True, concatenates into a single Tensor.
        recursive (bool): If True, searches subdirectories.
        verbose (bool): If True, shows a progress bar.

    Returns:
        Combined DataFrame, Tensor, or List of objects.
    """
    input_path = Path(input_path)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Invalid directory: {input_path}")

    pattern = f"*{extension}" if extension else "*"
    if recursive:
        files = sorted(list(input_path.rglob(pattern)))
    else:
        files = sorted(list(input_path.glob(pattern)))

    # Filter valid extensions
    valid_exts = {'.csv', '.tsv', '.txt', '.xlsx', '.root', '.parquet', '.pt', '.pth'}
    files = [f for f in files if f.suffix.lower() in valid_exts and f.is_file()]

    if not files:
        raise FileNotFoundError(f"No valid files found in {input_path}")

    loaded_data = []
    
    # Progress Bar Setup
    iterator = tqdm(files, desc="[IO] Loading Dataset", unit="file", disable=not verbose)

    # State tracking for concatenation strategy
    all_tensors = True
    all_dfs = True
    
    for f in iterator:
        try:
            data = load_file(
                f, 
                columns=columns, 
                as_tensor=as_tensor, 
                device=device, 
                verbose=False, # Suppress individual file logs
                **kwargs
            )
            
            if not isinstance(data, torch.Tensor): all_tensors = False
            if not isinstance(data, (pd.DataFrame, pd.Series)): all_dfs = False
            
            # Attach source metadata for DataFrames
            if isinstance(data, pd.DataFrame):
                data.attrs['source_file'] = f.name
                
            loaded_data.append(data)
        except Exception as e:
            if verbose:
                # Use tqdm.write to avoid breaking the progress bar
                if HAS_TQDM:
                    tqdm.write(f"[WARN] Skipping {f.name}: {e}")
                else:
                    print(f"[WARN] Skipping {f.name}: {e}")

    if not loaded_data:
        raise RuntimeError("Failed to load any files successfully.")

    # --- Concatenation Strategy ---

    # 1. Tensor Concatenation
    if all_tensors and len(loaded_data) > 0:
        try:
            return torch.cat(loaded_data, dim=0)
        except Exception as e:
            if verbose: print(f"[INFO] Tensor concatenation failed (mismatched shapes?), returning list. Error: {e}")
            return loaded_data

    # 2. DataFrame Concatenation
    if all_dfs and len(loaded_data) > 0:
        return pd.concat(loaded_data, ignore_index=True, copy=False)
    
    # 3. Mixed/Other -> Return List
    return loaded_data

# ==============================================================================
# 2. GENERIC SAVERS
# ==============================================================================

def _prepare_save_path(output_path: Union[str, Path], filename: str, ext: str) -> Path:
    """Internal helper to ensure directory existence and file extension."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if not filename.lower().endswith(ext):
        filename = f"{filename}{ext}"
    return output_path / filename

def save_data(
    data: Union[pd.DataFrame, torch.Tensor, Any], 
    output_path: Union[str, Path], 
    filename: str,
    format: str = "csv",
    verbose: bool = True,
    **kwargs
) -> Path:
    """
    Saves data to disk in the specified format with a progress indicator.

    Args:
        data: Data to save (DataFrame or Tensor).
        output_path: Target directory.
        filename: Target filename.
        format: Target format ('csv', 'parquet', 'pt', etc.).
        verbose: If True, prints status.

    Returns:
        Path: The absolute path to the saved file.
    """
    format = format.lower().replace('.', '')
    save_path = _prepare_save_path(output_path, filename, f".{format}")
    
    # Convert Tensor to DataFrame for tabular formats if requested
    if isinstance(data, torch.Tensor) and format in ["csv", "xlsx", "parquet", "root"]:
        data = pd.DataFrame(data.detach().cpu().numpy())

    try:
        # Wrap the saving process in a progress bar (acts as a spinner/timer for large files)
        with tqdm(total=1, desc=f"[IO] Saving {filename}", unit="file", disable=not verbose) as pbar:
            
            if isinstance(data, pd.DataFrame):
                # Extract 'index' to avoid duplication in kwargs
                include_index = kwargs.pop('index', False)

                if format == "csv":
                    data.to_csv(save_path, index=include_index, **kwargs)
                elif format == "xlsx":
                    data.to_excel(save_path, index=include_index, **kwargs)
                elif format == "parquet":
                    data.to_parquet(save_path, index=include_index, **kwargs)
                elif format == "root":
                    if not HAS_UPROOT: 
                        raise ImportError("Package 'uproot' is required to save .root files.")
                    tree_name = kwargs.get("tree_name", "tree")
                    kwargs.pop("tree_name", None) 
                    
                    # Fix for uproot: Convert DataFrame to Dict of Arrays
                    clean_dict = {col: data[col].values for col in data.columns}
                    
                    with uproot.recreate(save_path) as f:
                        f[tree_name] = clean_dict
                else:
                    raise ValueError(f"Unsupported format '{format}' for DataFrame.")
            else:
                # Binary save for Tensors/Objects
                if format in ["pt", "pth"]:
                    torch.save(data, save_path)
                else:
                    raise ValueError(f"For non-tabular data, use format='pt'.")
            
            pbar.update(1)
            
    except Exception as e:
        raise RuntimeError(f"Error saving to {save_path}: {str(e)}") from e
    
    return save_path

# ==============================================================================
# 3. MODEL I/O
# ==============================================================================

def save_model(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[optim.Optimizer] = None,
    config: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    filename: str = "model_checkpoint",
    verbose: bool = True
) -> Path:
    """
    Saves a complete neural network checkpoint (weights, optimizer, config).
    Uses a progress bar to indicate activity.
    """
    save_path = _prepare_save_path(path, filename, ".pt")
    
    # Auto-detect internal optimizer (e.g., from NeuralModel)
    if optimizer is None and hasattr(model, 'optimizer'):
        optimizer = model.optimizer

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config or {},
        'metadata': metadata or {}
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    try:
        with tqdm(total=1, desc=f"[IO] Saving Model", unit="chkpt", disable=not verbose) as pbar:
            torch.save(checkpoint, save_path)
            pbar.update(1)
    except Exception as e:
        raise e
        
    return save_path

def load_model(
    path: Union[str, Path],
    model_instance: Optional[nn.Module] = None,
    optimizer_instance: Optional[optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Loads a checkpoint. Can restore state into a model instance or return metadata.

    Args:
        path: Path to .pt file.
        model_instance: Model to inject weights into.
        optimizer_instance: Optimizer to inject state into.
        device: 'cpu' or 'cuda'.
        strict: Enforce strict state_dict matching.
        verbose: Show progress/status.

    Returns:
        Dict containing 'config', 'metadata', and 'full_checkpoint'.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
        
    try:
        with tqdm(total=1, desc=f"[IO] Loading Model", unit="chkpt", disable=not verbose) as pbar:
            checkpoint = torch.load(path, map_location=device)
            pbar.update(1)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e
    
    # 1. Restore Model Weights
    if model_instance is not None:
        if 'model_state_dict' in checkpoint:
            model_instance.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            if verbose: tqdm.write("   [+] Model weights restored.")
        else:
            try:
                # Legacy support for files containing only state_dict
                model_instance.load_state_dict(checkpoint, strict=strict)
                if verbose: tqdm.write("   [!] Legacy format detected. Weights restored.")
            except:
                if verbose: tqdm.write("   [ERR] Could not find 'model_state_dict' in checkpoint.")

    # 2. Restore Optimizer State
    if optimizer_instance is None and hasattr(model_instance, 'optimizer'):
        optimizer_instance = model_instance.optimizer

    if optimizer_instance is not None and 'optimizer_state_dict' in checkpoint:
        optimizer_instance.load_state_dict(checkpoint['optimizer_state_dict'])
        if verbose: tqdm.write("   [+] Optimizer state restored.")

    return {
        'config': checkpoint.get('config', {}),
        'metadata': checkpoint.get('metadata', {}),
        'full_checkpoint': checkpoint
    }