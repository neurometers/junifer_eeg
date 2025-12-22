"""
Aggregators - Data aggregation utilities for EEG marker tensors.

This module provides functions for aggregating marker data across
different dimensions (channels, epochs, time, etc.).
"""

from typing import Callable, Dict, List, Optional

import numpy as np
from scipy import stats


def trim_mean80(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute 80% trimmed mean (removes top/bottom 10%)."""
    return stats.trim_mean(data, proportiontocut=0.1, axis=axis)


def trim_mean90(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute 90% trimmed mean (removes top/bottom 5%)."""
    return stats.trim_mean(data, proportiontocut=0.05, axis=axis)


def nanmean(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute mean ignoring NaN values."""
    return np.nanmean(data, axis=axis)


def nanmedian(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute median ignoring NaN values."""
    return np.nanmedian(data, axis=axis)


def nanstd(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute standard deviation ignoring NaN values."""
    return np.nanstd(data, axis=axis)


def percentile_25(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute 25th percentile."""
    return np.nanpercentile(data, 25, axis=axis)


def percentile_75(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute 75th percentile."""
    return np.nanpercentile(data, 75, axis=axis)


def iqr(data: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """Compute interquartile range (Q3 - Q1)."""
    return stats.iqr(data, axis=axis, nan_policy="omit")


# Registry of aggregation functions
AGGREGATION_FUNCTIONS: Dict[str, Callable] = {
    # Basic statistics
    "mean": lambda d, axis=None: np.mean(d, axis=axis),
    "median": lambda d, axis=None: np.median(d, axis=axis),
    "std": lambda d, axis=None: np.std(d, axis=axis),
    "var": lambda d, axis=None: np.var(d, axis=axis),
    "min": lambda d, axis=None: np.min(d, axis=axis),
    "max": lambda d, axis=None: np.max(d, axis=axis),
    "sum": lambda d, axis=None: np.sum(d, axis=axis),
    # NaN-safe versions
    "nanmean": nanmean,
    "nanmedian": nanmedian,
    "nanstd": nanstd,
    # Trimmed means (NICE-compatible)
    "trim_mean80": trim_mean80,
    "trim_mean90": trim_mean90,
    # Percentiles
    "percentile_25": percentile_25,
    "percentile_75": percentile_75,
    "iqr": iqr,
    # Range
    "range": lambda d, axis=None: np.ptp(d, axis=axis),
    # Count
    "count": lambda d, axis=None: np.sum(~np.isnan(d), axis=axis)
    if np.any(np.isnan(d))
    else d.shape[axis]
    if axis is not None
    else d.size,
}


def aggregate(
    data: np.ndarray, method: str, axis: Optional[int] = None
) -> np.ndarray:
    """Aggregate data using the specified method.

    Parameters
    ----------
    data : np.ndarray
        Input data to aggregate.
    method : str
        Aggregation method name (e.g., 'mean', 'std', 'trim_mean80').
    axis : int, optional
        Axis along which to aggregate. If None, aggregates over all elements.

    Returns
    -------
    np.ndarray
        Aggregated data.

    Raises
    ------
    ValueError
        If method is not recognized.
    """
    if method not in AGGREGATION_FUNCTIONS:
        available = list(AGGREGATION_FUNCTIONS.keys())
        raise ValueError(
            f"Unknown aggregation method: '{method}'. Available: {available}"
        )

    return AGGREGATION_FUNCTIONS[method](data, axis=axis)


def aggregate_channels(
    data: np.ndarray, method: str, channel_axis: int = -1
) -> np.ndarray:
    """Aggregate data across channel dimension.

    Parameters
    ----------
    data : np.ndarray
        Input data with channel dimension.
    method : str
        Aggregation method.
    channel_axis : int
        Axis index for channels (default: -1, last axis).

    Returns
    -------
    np.ndarray
        Data with channel dimension removed.
    """
    return aggregate(data, method, axis=channel_axis)


def aggregate_epochs(
    data: np.ndarray, method: str, epoch_axis: int = 0
) -> np.ndarray:
    """Aggregate data across epoch dimension.

    Parameters
    ----------
    data : np.ndarray
        Input data with epoch dimension.
    method : str
        Aggregation method.
    epoch_axis : int
        Axis index for epochs (default: 0, first axis).

    Returns
    -------
    np.ndarray
        Data with epoch dimension removed.
    """
    return aggregate(data, method, axis=epoch_axis)


def aggregate_time(
    data: np.ndarray, method: str = "mean", time_axis: int = -1
) -> np.ndarray:
    """Aggregate data across time dimension.

    Parameters
    ----------
    data : np.ndarray
        Input data with time dimension.
    method : str
        Aggregation method (default: 'mean').
    time_axis : int
        Axis index for time (default: -1, last axis).

    Returns
    -------
    np.ndarray
        Data with time dimension removed.
    """
    return aggregate(data, method, axis=time_axis)


def aggregate_connectivity(
    data: np.ndarray, method: str, connectivity_axis: int = 1
) -> np.ndarray:
    """Aggregate connectivity matrix across one channel dimension.

    Parameters
    ----------
    data : np.ndarray
        Input connectivity data (epochs, channels_i, channels_j).
    method : str
        Aggregation method.
    connectivity_axis : int
        Axis to aggregate over (default: 1).

    Returns
    -------
    np.ndarray
        Data with one connectivity dimension removed.
    """
    return aggregate(data, method, axis=connectivity_axis)


class DataAggregator:
    """Helper class for applying multiple aggregations to tensor data."""

    def __init__(self, data: np.ndarray):
        """Initialize with data.

        Parameters
        ----------
        data : np.ndarray
            Input tensor data.
        """
        self.data = data
        self._original_shape = data.shape

    def aggregate(
        self, method: str, axis: int, inplace: bool = True
    ) -> np.ndarray:
        """Apply aggregation along an axis.

        Parameters
        ----------
        method : str
            Aggregation method.
        axis : int
            Axis to aggregate.
        inplace : bool
            If True, update self.data. If False, return copy.

        Returns
        -------
        np.ndarray
            Aggregated data.
        """
        result = aggregate(self.data, method, axis=axis)
        if inplace:
            self.data = result
        return result

    def reset(self):
        """Reset to original data (not implemented - would need copy)."""
        raise NotImplementedError("Reset not supported - create new instance")

    @property
    def shape(self) -> tuple:
        """Current data shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Current number of dimensions."""
        return self.data.ndim


def flatten_connectivity_matrix(
    matrix: np.ndarray, include_diagonal: bool = False
) -> np.ndarray:
    """Flatten a connectivity matrix to upper triangular values.

    Parameters
    ----------
    matrix : np.ndarray
        Square connectivity matrix (n_channels, n_channels) or
        3D tensor (n_epochs, n_channels, n_channels).
    include_diagonal : bool
        Whether to include diagonal values.

    Returns
    -------
    np.ndarray
        Flattened upper triangular values.
    """
    if matrix.ndim == 2:
        n = matrix.shape[0]
        k = 0 if include_diagonal else 1
        indices = np.triu_indices(n, k=k)
        return matrix[indices]
    elif matrix.ndim == 3:
        n_epochs = matrix.shape[0]
        n = matrix.shape[1]
        k = 0 if include_diagonal else 1
        indices = np.triu_indices(n, k=k)

        result = []
        for i in range(n_epochs):
            result.append(matrix[i][indices])
        return np.array(result)
    else:
        raise ValueError(f"Expected 2D or 3D matrix, got {matrix.ndim}D")


def unflatten_connectivity_matrix(
    values: np.ndarray,
    n_channels: int,
    include_diagonal: bool = False,
    symmetric: bool = True,
) -> np.ndarray:
    """Reconstruct connectivity matrix from flattened values.

    Parameters
    ----------
    values : np.ndarray
        Flattened upper triangular values (1D) or
        (n_epochs, n_pairs) for multiple epochs.
    n_channels : int
        Number of channels in the original matrix.
    include_diagonal : bool
        Whether diagonal values are included.
    symmetric : bool
        Whether to make the matrix symmetric.

    Returns
    -------
    np.ndarray
        Reconstructed connectivity matrix.
    """
    k = 0 if include_diagonal else 1
    indices = np.triu_indices(n_channels, k=k)

    if values.ndim == 1:
        matrix = np.zeros((n_channels, n_channels))
        matrix[indices] = values
        if symmetric:
            matrix = matrix + matrix.T
            if include_diagonal:
                np.fill_diagonal(matrix, matrix.diagonal() / 2)
        return matrix
    elif values.ndim == 2:
        n_epochs = values.shape[0]
        matrices = np.zeros((n_epochs, n_channels, n_channels))
        for i in range(n_epochs):
            matrices[i][indices] = values[i]
            if symmetric:
                matrices[i] = matrices[i] + matrices[i].T
                if include_diagonal:
                    np.fill_diagonal(matrices[i], matrices[i].diagonal() / 2)
        return matrices
    else:
        raise ValueError(f"Expected 1D or 2D values, got {values.ndim}D")


def generate_pair_names(
    channel_names: List[str], include_diagonal: bool = False
) -> List[str]:
    """Generate channel pair names for flattened connectivity.

    Parameters
    ----------
    channel_names : list of str
        List of channel names.
    include_diagonal : bool
        Whether to include self-connections.

    Returns
    -------
    list of str
        List of pair names in format "ch1-ch2".
    """
    n = len(channel_names)
    k = 0 if include_diagonal else 1
    indices = np.triu_indices(n, k=k)

    pair_names = []
    for i, j in zip(indices[0], indices[1]):
        pair_names.append(f"{channel_names[i]}-{channel_names[j]}")

    return pair_names
