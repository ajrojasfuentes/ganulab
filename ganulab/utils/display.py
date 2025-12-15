# ganulab/utils/display.py

"""
GANU-Lab Display Utilities.

This module provides professional-grade visualization tools for data analysis (EDA),
model training monitoring, regression results, and generative outputs. It leverages
Seaborn and Matplotlib, stylized for high-quality, publication-ready figures.

Features:
- Automatic tensor-to-numpy conversion.
- Uncertainty visualization (Intervals, Reliability Curves).
- Comparative distribution plotting (Targets vs Predictions).
- Generative image grids.
"""

import math
from pathlib import Path
from typing import List, Optional, Union, Dict, Tuple, Any, Sequence

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 0. CONFIGURATION & HELPERS
# ==============================================================================

def set_style(style: str = "darkgrid", context: str = "notebook", font_scale: float = 1.1):
    """
    Configures global Matplotlib/Seaborn styles for a professional aesthetic.

    Args:
        style (str): Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
        context (str): Scaling context ('paper', 'notebook', 'talk', 'poster').
        font_scale (float): Font scaling factor.
    """
    sns.set_theme(style=style, context=context, font_scale=font_scale)
    
    # Custom overrides for the "GANU-Lab" look
    plt.rcParams['figure.figsize'] = (12, 7)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['savefig.transparent'] = False 
    plt.rcParams['axes.axisbelow'] = True # Ensure grid is behind plot elements

def _to_numpy(data: Any) -> np.ndarray:
    """Internal helper to robustly convert various inputs (Tensor, List) to Numpy."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data)
    if isinstance(data, pd.Series):
        return data.values
    return data

def _save_and_show(save_path: Optional[str] = None):
    """Internal helper to handle saving and displaying plots."""
    if save_path:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"💾 Plot saved to: {save_path}")
    plt.show()

# ==============================================================================
# 1. TRAINING & LOSS VISUALIZATION
# ==============================================================================

def plot_training_runs(
    losses_data: Union[Sequence[float], Sequence[Sequence[float]], torch.Tensor],
    title: str = "Training Loss History",
    smoothing_window: int = 50,
    log_scale_y: bool = False,
    save_path: Optional[str] = None
):
    """
    Plots loss curves for one or multiple training runs on the same chart.
    Visualizes raw noisy data faintly behind a smoothed trend line.

    Args:
        losses_data: Single list/tensor `[l1, l2...]` or list of lists `[[run1...], [run2...]]`.
        title (str): Plot title.
        smoothing_window (int): Window size for the moving average trend line.
        log_scale_y (bool): If True, uses a logarithmic scale for the Y axis.
        save_path (str, optional): Path to save the figure.
    """
    # 1. Input Normalization
    runs: List[np.ndarray] = []
    
    def extract_run(item): return _to_numpy(item).flatten()

    if isinstance(losses_data, torch.Tensor):
        losses_data = losses_data.detach().cpu().numpy()

    if isinstance(losses_data, np.ndarray):
        if losses_data.ndim == 1:
             runs.append(losses_data)
        else:
             runs = [row for row in losses_data]
    elif isinstance(losses_data, list):
        if len(losses_data) > 0 and isinstance(losses_data[0], (list, np.ndarray, torch.Tensor)):
             for item in losses_data:
                  runs.append(extract_run(item))
        else:
             runs.append(extract_run(losses_data))
    else:
         raise ValueError("Unsupported format for losses_data.")
    
    # 2. Plotting
    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("husl", len(runs)) if len(runs) > 1 else ["teal"]
    
    for i, run_data in enumerate(runs):
        color = palette[i]
        label = f"Run {i+1}" if len(runs) > 1 else "Loss Trend"
        steps = np.arange(len(run_data))
        
        # Raw Data (Background)
        sns.lineplot(
            x=steps, y=run_data, color=color, 
            alpha=0.25, linewidth=1, label=None, zorder=1
        )
        
        # Smoothed Data (Foreground)
        if len(run_data) >= smoothing_window and smoothing_window > 1:
            smoothed = pd.Series(run_data).rolling(window=smoothing_window, min_periods=1, center=True).mean()
            sns.lineplot(
                x=steps, y=smoothed.values, color=color, 
                linewidth=2.5, label=label, zorder=2
            )
        else:
            sns.lineplot(
                x=steps, y=run_data, color=color, 
                linewidth=2.5, label=label, zorder=2
            )

    if log_scale_y:
        plt.yscale('log')
        plt.ylabel("Loss (Log Scale)")
    else:
        plt.ylabel("Loss Value")
        
    plt.xlabel("Iterations / Steps")
    plt.title(title, fontweight='bold', pad=15)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9)
    plt.tight_layout()

    _save_and_show(save_path)

def plot_training_curves(
    history: Dict[str, List[float]], 
    title: str = "Metric Comparison during Training",
    smoothing_window: int = 5,
    log_scale_y: bool = False,
    save_path: Optional[str] = None
):
    """
    Plots multiple distinct metrics from a single training run (e.g., Train Loss vs Val Loss).

    Args:
        history (Dict): Dictionary keys are metric names, values are lists of floats.
        title (str): Plot title.
        smoothing_window (int): Smoothing window size.
        log_scale_y (bool): Logarithmic Y axis.
        save_path (str, optional): Path to save the figure.
    """
    plt.figure(figsize=(12, 7))
    colors = sns.color_palette("Set2", len(history))
    
    for i, (metric_name, values) in enumerate(history.items()):
        color = colors[i]
        values_np = _to_numpy(values).flatten()
        steps = np.arange(len(values_np))

        if len(values_np) > smoothing_window * 2:
            plt.plot(steps, values_np, alpha=0.2, color=color)
            values_smooth = pd.Series(values_np).rolling(window=smoothing_window, min_periods=1).mean()
            plt.plot(steps, values_smooth, label=metric_name, color=color, linewidth=2.5)
        else:
            plt.plot(steps, values_np, label=metric_name, marker='o', markersize=4, color=color, linewidth=2)

    if log_scale_y:
        plt.yscale('log')
        plt.ylabel("Metric Value (Log Scale)")
    else:
        plt.ylabel("Metric Value")
        
    plt.xlabel("Epochs / Steps")
    plt.title(title, fontweight='bold')
    plt.legend(frameon=True)
    plt.tight_layout()
    
    _save_and_show(save_path)

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def plot_distributions(
    data: Union[pd.DataFrame, torch.Tensor],
    columns: Optional[List[str]] = None,
    kind: str = "hist", 
    bins: Union[int, str] = 50,
    title: str = "Feature Distributions",
    color: str = "teal",
    save_path: Optional[str] = None
):
    """
    Plots the distribution of multiple columns in a grid layout.

    Args:
        data: Input data (DataFrame or Tensor).
        columns: List of column names to plot. If None, plots first 12 numeric columns.
        kind (str): Type of plot ('hist', 'kde', 'box').
        bins (int): Number of bins for histograms.
        title (str): Title of the figure.
        color (str): Main color of the plots.
        save_path (str, optional): Path to save the figure.
    """
    if isinstance(data, torch.Tensor):
        data = pd.DataFrame(_to_numpy(data))
        if columns: data.columns = columns

    if columns is None:
        data = data.select_dtypes(include=np.number)
        columns = data.columns[:12]

    if len(columns) == 0: return

    n_cols = 3
    n_rows = math.ceil(len(columns) / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        plot_kwargs = {'data': data, 'x': col, 'ax': ax, 'color': color}
        
        if kind == "hist":
            sns.histplot(**plot_kwargs, kde=True, alpha=0.5, bins=bins, edgecolor=None,
                         line_kws={'color': 'darkslategray', 'linewidth': 1.5})
        elif kind == "kde":
            sns.kdeplot(**plot_kwargs, fill=True, alpha=0.5, warn_singular=False)
        elif kind == "box":
            sns.boxplot(**plot_kwargs, width=0.5, linewidth=1.5)
        
        ax.set_title(col, fontweight='bold')
        ax.set_xlabel("")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    plt.suptitle(title, y=1.02, fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    _save_and_show(save_path)

def plot_heatmap(
    data: pd.DataFrame,
    title: str = "Unnamed Heatmap",
    columns: tuple[str, str] | None = None,
    hist_kwargs: dict | None = None,
    fig_kwargs: dict | None = None,
):
    # Defaults
    if columns is None:
        if data.shape[1] < 2:
            raise ValueError("DataFrame needs at least 2 columns to plot a heatmap.")
        columns = (data.columns[0], data.columns[1])

    if hist_kwargs is None:
        hist_kwargs = {}
    if fig_kwargs is None:
        fig_kwargs = {}

    for col in columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Figura
    default_fig_kwargs = dict(figsize=(8, 6), dpi=480)
    default_fig_kwargs.update(fig_kwargs)
    fig, ax = plt.subplots(**default_fig_kwargs)

    # Datos y limpieza
    x = data[columns[0]].to_numpy()
    y = data[columns[1]].to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Parámetros de hexbin
    default_hexbin_kwargs = dict(
        gridsize=80,
        cmap="viridis",
        mincnt=1,
        alpha=0.7,
    )
    default_hexbin_kwargs.update(hist_kwargs)

    hb = ax.hexbin(x, y, **default_hexbin_kwargs)

    cbar = fig.colorbar(hb, ax=ax, label="Density")
    ax.set_title(title)
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])

    fig.tight_layout()
    plt.show()

    return fig, ax, hb

def plot_correlations(
    data: pd.DataFrame, 
    method: str = "pearson", 
    title: str = "Correlation Matrix",
    cmap: str = "RdBu_r", 
    annot: bool = True,
    threshold: float = 0.0,
    save_path: Optional[str] = None
):
    """
    Plots a correlation heatmap.

    Args:
        data: Input DataFrame.
        method: Correlation method ('pearson', 'kendall', 'spearman').
        threshold: Minimum correlation value to display (filters weak correlations).
        save_path (str, optional): Path to save the figure.
    """
    numeric_data = data.select_dtypes(include=np.number)
    if numeric_data.empty: return
         
    corr = numeric_data.corr(method=method)
    if threshold > 0: corr = corr[corr.abs() > threshold]

    plt.figure(figsize=(10, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
        annot=annot, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .7}
    )
    plt.title(title, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    _save_and_show(save_path)

# ==============================================================================
# 3. REGRESSION & INTERVAL VISUALIZATION
# ==============================================================================

def compare_distributions(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    columns: Optional[List[str]] = None,
    bins: int = 100,
    title: str = "Targets vs Predictions Distribution",
    save_path: Optional[str] = None
):
    """
    Plots overlaid histograms of True Targets vs Model Predictions.
    
    Args:
        y_true: Ground truth values [N, Features].
        y_pred: Predicted values [N, Features].
        columns: Names of the features.
        bins: Number of histogram bins.
        save_path (str, optional): Path to save the figure.
    """
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    
    if y_true_np.ndim == 1: y_true_np = y_true_np[:, np.newaxis]
    if y_pred_np.ndim == 1: y_pred_np = y_pred_np[:, np.newaxis]

    num_features = y_true_np.shape[1]
    if columns is None: columns = [f"Feature {i}" for i in range(num_features)]
        
    n_cols = 3
    n_rows = math.ceil(num_features / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i in range(num_features):
        ax = axes[i]
        col_name = columns[i] if i < len(columns) else f"Feature {i}"
        
        t_data = y_true_np[:, i]
        p_data = y_pred_np[:, i]
        
        min_val = min(t_data.min(), p_data.min())
        max_val = max(t_data.max(), p_data.max())
        
        sns.histplot(
            x=t_data, ax=ax, bins=bins, 
            color='black', alpha=0.3, label='Targets',
            binrange=(min_val, max_val), stat='density', element="step",
            fill=True, lw=0
        )
        
        sns.histplot(
            x=p_data, ax=ax, bins=bins, 
            color='teal', alpha=0.5, label='Predictions',
            binrange=(min_val, max_val), stat='density', element="step",
            fill=True, lw=0
        )
        
        ax.set_title(col_name, fontweight='bold')
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)

    for j in range(i + 1, len(axes)): axes[j].axis('off')

    plt.suptitle(title, y=1.02, fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    _save_and_show(save_path)

def plot_regression_intervals(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_lower: Optional[Union[np.ndarray, torch.Tensor]] = None,
    y_upper: Optional[Union[np.ndarray, torch.Tensor]] = None,
    samples: int = 150,
    title: str = "Regression Results with Uncertainty",
    sort_by_true: bool = True,
    save_path: Optional[str] = None
):
    """
    Plots true vs. predicted values with uncertainty intervals (shaded area).

    Args:
        y_true: Ground truth.
        y_pred: Prediction.
        y_lower: Lower bound of uncertainty.
        y_upper: Upper bound of uncertainty.
        samples: Number of points to subsample for cleaner visualization.
        sort_by_true: Sort X-axis by ground truth value for readability.
        save_path (str, optional): Path to save the figure.
    """
    y_true_np = _to_numpy(y_true).flatten()
    y_pred_np = _to_numpy(y_pred).flatten()
    
    N = len(y_true_np)
    if N > samples:
        indices = np.random.choice(N, samples, replace=False)
    else:
        indices = np.arange(N)

    if sort_by_true:
        sort_idx = np.argsort(y_true_np[indices])
        indices = indices[sort_idx]
        
    y_t_sub = y_true_np[indices]
    y_p_sub = y_pred_np[indices]
    x = np.arange(len(indices))
    
    plt.figure(figsize=(14, 7))
    
    if y_lower is not None and y_upper is not None:
        y_l_sub = _to_numpy(y_lower).flatten()[indices]
        y_u_sub = _to_numpy(y_upper).flatten()[indices]
        
        plt.fill_between(
            x, y_l_sub, y_u_sub, 
            color='teal', alpha=0.2, 
            label='Uncertainty Interval', zorder=1
        )
        
    plt.plot(x, y_t_sub, 'k--', label='Ground Truth', alpha=0.7, linewidth=1.5, zorder=2)
    plt.plot(x, y_p_sub, color='teal', label='Prediction', linewidth=2.5, zorder=3)
    plt.scatter(x, y_t_sub, color='black', s=15, alpha=0.5, zorder=2)
    plt.scatter(x, y_p_sub, color='teal', s=25, zorder=3)

    plt.title(title, fontweight='bold')
    plt.ylabel("Value")
    plt.xlabel("Sample Index")
    plt.legend(frameon=True)
    plt.tight_layout()
    
    _save_and_show(save_path)

def plot_reliability_curves(
    x_data: Union[np.ndarray, torch.Tensor],
    y_data: Union[np.ndarray, torch.Tensor],
    columns: Optional[List[str]] = None,
    bins: int = 10,
    quantile_bins: bool = True,
    title: str = "Reliability / Calibration Curves",
    xlabel: str = "Predicted Value / Uncertainty",
    ylabel: str = "Observed Value / Error",
    save_path: Optional[str] = None
):
    """
    Plots Reliability (Calibration) Curves for multiple features on a single chart.
    Useful for checking if predicted uncertainty matches empirical error.

    Args:
        x_data: Predictor (e.g., Predicted Probability or Uncertainty).
        y_data: Target (e.g., Observed Frequency or Actual Error).
        columns: Names for the curves/features.
        bins: Number of bins to calculate means.
        quantile_bins: If True, uses equal frequency bins (quantiles).
        save_path (str, optional): Path to save the figure.
    """
    x_np = _to_numpy(x_data)
    y_np = _to_numpy(y_data)

    if x_np.ndim == 1: x_np = x_np[:, np.newaxis]
    if y_np.ndim == 1: y_np = y_np[:, np.newaxis]
    
    if x_np.shape != y_np.shape:
        raise ValueError(f"Shape mismatch: x={x_np.shape}, y={y_np.shape}")

    num_features = x_np.shape[1]
    if columns is None:
        columns = [f"Feature {i}" for i in range(num_features)]

    plt.figure(figsize=(10, 8))
    
    # 1. Plot Identity Line
    global_min = min(x_np.min(), y_np.min())
    global_max = max(x_np.max(), y_np.max())
    plt.plot([global_min, global_max], [global_min, global_max], 
             "k--", label="Perfect Calibration", alpha=0.4, linewidth=1.5)

    # 2. Calculate and Plot Curves
    colors = sns.color_palette("husl", num_features) if num_features > 1 else ["teal"]

    for i in range(num_features):
        x_col = x_np[:, i]
        y_col = y_np[:, i]
        
        mask = np.isfinite(x_col) & np.isfinite(y_col)
        x_col = x_col[mask]
        y_col = y_col[mask]

        if quantile_bins:
            try:
                bin_edges = np.quantile(x_col, np.linspace(0, 1, bins + 1))
            except Exception:
                bin_edges = np.linspace(x_col.min(), x_col.max(), bins + 1)
        else:
            bin_edges = np.linspace(x_col.min(), x_col.max(), bins + 1)
            
        bin_indices = np.digitize(x_col, bin_edges)
        bin_indices = np.clip(bin_indices, 1, bins) - 1

        curve_x = []
        curve_y = []

        for b in range(bins):
            mask_b = (bin_indices == b)
            if np.any(mask_b):
                curve_x.append(x_col[mask_b].mean())
                curve_y.append(y_col[mask_b].mean())
        
        plt.plot(
            curve_x, curve_y, 
            marker='o', linestyle='-', linewidth=2, markersize=6,
            label=columns[i], color=colors[i], alpha=0.8
        )

    plt.title(title, fontweight='bold', pad=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    _save_and_show(save_path)

# ==============================================================================
# 4. GENERATIVE OUTPUTS (Images)
# ==============================================================================

def plot_image_grid(
    tensor_images: torch.Tensor, 
    n_row: Optional[int] = None,
    normalize: bool = True,
    title: str = "Generated Samples",
    save_path: Optional[str] = None
):
    """
    Displays a grid of images from a batch tensor [B, C, H, W].
    
    Args:
        tensor_images: Batch of images.
        n_row: Number of rows in grid.
        normalize: Normalize pixel values to [0, 1].
        save_path (str, optional): Path to save the figure.
    """
    try:
        from torchvision.utils import make_grid
    except ImportError:
        print("[ERR] Torchvision is not installed.")
        return

    tensor_images = tensor_images.detach().cpu()
    
    if n_row is None:
        n_row = int(math.sqrt(tensor_images.shape[0]))
    
    grid = make_grid(tensor_images, nrow=n_row, normalize=normalize, padding=2, pad_value=1)
    ndarr = grid.numpy().transpose((1, 2, 0))
    
    plt.figure(figsize=(10, 10))
    plt.imshow(ndarr, interpolation='nearest')
    plt.axis('off')
    plt.title(title, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    _save_and_show(save_path)