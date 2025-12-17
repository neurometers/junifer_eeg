"""Symbolic mutual information with ROI aggregation."""

from typing import Any, ClassVar, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker
from .symbolic_mutual_information import SymbolicMutualInformation

__all__ = ["SymbolicMutualInformationROIs"]


@register_marker
class SymbolicMutualInformationROIs(EEGEpochsMarker):
    """Symbolic mutual information with ROI/channel/trial aggregation.

    Combines SymbolicMutualInformation with aggregation pipeline for:
    - Connectivity dimension aggregation (across channel_y)
    - Channel dimension aggregation (across channel_x)
    - Trial/epoch aggregation

    Parameters
    ----------
    kernel : int, default=3
        Length of ordinal patterns.
    taus : int or list of int, default=8
        Time delay(s) for ordinal patterns. Can be single integer or list of integers
        for multiple temporal scales.
    weighted : bool, default=True
        Whether to compute weighted SMI.
    csd : bool, default=True
        Whether to apply Current Source Density preprocessing.
    rois : list of str or int, optional
        Channel specifications for filtering BEFORE connectivity computation.
        Each item can be:
        - int: channel index (e.g., 0, 1, 223)
        - str: channel name (e.g., 'E1', 'E224') OR semantic ROI (e.g., 'scalp')
        If None, uses all channels.
        **CRITICAL**: ROI filtering is applied BEFORE connectivity computation,
        filtering BOTH dimensions of the connectivity matrix.
    tmin : float, optional
        Start time for analysis window.
    tmax : float, optional
        End time for analysis window.
    connectivity_method : str, optional
        Method to aggregate across connectivity dimension (channel_y).
        Options: 'median', 'mean', 'std', 'trim_mean80', 'trim_mean90', etc.
    channel_method : str, optional
        Method to aggregate across channel dimension (channel_x).
        Options: 'median', 'mean', 'std', 'trim_mean80', 'trim_mean90', etc.
    trial_method : str, optional
        Method to aggregate across trial/epoch dimension.
        Options: 'median', 'mean', 'std', 'trim_mean80', 'trim_mean90', etc.
    equipment : str, default="egi256"
        Equipment configuration.
    on : str, optional
        Data type (default "EEG").
    name : str, optional
        Marker name.

    """

    _DEPENDENCIES: ClassVar = {
        "mne",
        "numpy",
        "scipy",
        "numba",
        "mne-connectivity",
    }
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"symbolicmutualinformation": "vector"}
    }

    def __init__(
        self,
        kernel: int = 3,
        taus: int | list[int] = 8,
        weighted: bool = True,
        csd: bool = True,
        rois: Optional[list[str] | list[int]] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        connectivity_method: Optional[str] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize SMI with ROI aggregation."""
        super().__init__(
            tmin=tmin, tmax=tmax, equipment=equipment, on=on, name=name
        )

        self.kernel = kernel
        self.taus = taus
        self.weighted = weighted
        self.csd = csd
        self.rois = rois
        self.connectivity_method = connectivity_method
        self.channel_method = channel_method
        self.trial_method = trial_method

    def get_output_type(self, input_type: str, output_feature: str) -> str:
        """Get output type based on aggregation settings.

        Returns
        -------
        str
            - 'timeseries': 2D/3D tensor (0-1 aggregations)
            - 'vector': 1D array (2 aggregations)
            - 'scalar_table': scalar (all 3 aggregations)

        """
        # Count active aggregations
        agg_count = sum(
            [
                self.connectivity_method is not None,
                self.channel_method is not None,
                self.trial_method is not None,
            ]
        )

        if agg_count <= 1:
            return "timeseries"
        elif agg_count == 3:
            return "scalar_table"
        else:
            return "vector"

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute SMI with aggregation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Aggregated SMI results.

        """
        logger.debug("Computing SMI with ROI aggregation")

        # Compute base SMI connectivity
        smi_marker = SymbolicMutualInformation(
            kernel=self.kernel,
            taus=self.taus,
            weighted=self.weighted,
            csd=self.csd,
            rois=self.rois,
            tmin=self.tmin,
            tmax=self.tmax,
            equipment=self.equipment,
            on=self._on,
        )
        result = smi_marker.compute(input, extra_input)

        # Get connectivity data (pairs format) from base marker
        connectivity_data = result["symbolicmutualinformation"]["data"]

        # Handle single vs multiple taus
        if connectivity_data.ndim == 2:
            # Single tau: (n_epochs, n_pairs)
            output_data = self._aggregate_single_tau(connectivity_data)
        else:
            # Multiple taus: (n_taus, n_epochs, n_pairs)
            output_data = self._aggregate_multiple_taus(connectivity_data)

        # Format output
        return {"symbolicmutualinformation": {"data": output_data}}

    def _aggregate_single_tau(
        self, connectivity_data: np.ndarray
    ) -> np.ndarray:
        """Aggregate connectivity data for single tau.

        Follows the same 3-step aggregation as the test helper:
        1. Reconstruct full matrix and apply connectivity_method (→ per-channel)
        2. Apply channel_method across channels
        3. Apply trial_method across epochs
        """
        from ..utils import aggregate_data

        n_epochs, n_pairs = connectivity_data.shape

        # If no aggregation requested, return raw connectivity pairs
        if (
            self.connectivity_method is None
            and self.channel_method is None
            and self.trial_method is None
        ):
            return connectivity_data

        # Step 1: Reconstruct full matrix and apply connectivity_method
        # This converts (n_epochs, n_pairs) → (n_epochs, n_channels)
        # by reconstructing the matrix and averaging across connections

        # Determine number of channels from n_pairs
        n_channels = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)

        # Reconstruct full connectivity matrices
        indices_use = np.triu_indices(n_channels, k=1)
        connectivity_tensor = np.zeros((n_epochs, n_channels, n_channels))

        for epoch_idx in range(n_epochs):
            connectivity_tensor[epoch_idx][indices_use] = connectivity_data[
                epoch_idx
            ]
            # Symmetrize
            connectivity_tensor[epoch_idx] = (
                connectivity_tensor[epoch_idx]
                + connectivity_tensor[epoch_idx].T
            )

        # Apply connectivity_method: aggregate across connections (axis=2)
        # (n_epochs, n_channels, n_channels) → (n_epochs, n_channels)
        if self.connectivity_method is not None:
            current_data = aggregate_data(
                connectivity_tensor, self.connectivity_method, axis=2
            )
        else:
            # Default: mean across connections (like test helper)
            current_data = np.mean(connectivity_tensor, axis=2)

        # Step 2: Apply channel_method (same as SpectralPowerBands)
        if self.channel_method is not None:
            # (n_epochs, n_channels) → (n_epochs,)
            current_data = aggregate_data(
                current_data, self.channel_method, axis=1
            )

        # Step 3: Apply trial_method (same as SpectralPowerBands)
        if self.trial_method is not None:
            if current_data.ndim == 2:
                # (n_epochs, n_channels) → (n_channels,)
                current_data = aggregate_data(
                    current_data, self.trial_method, axis=0
                )
            elif current_data.ndim == 1:
                # (n_epochs,) → scalar
                current_data = aggregate_data(
                    current_data, self.trial_method, axis=None
                )

        return current_data

    def _aggregate_multiple_taus(
        self, connectivity_data: np.ndarray
    ) -> np.ndarray:
        """Aggregate connectivity data for multiple taus.

        Follows SpectralPowerBands pattern: process each tau, then stack.
        """
        n_taus = connectivity_data.shape[0]

        # Process each tau through single tau aggregation
        tau_results = []
        for tau_idx in range(n_taus):
            tau_data = connectivity_data[tau_idx]  # (n_epochs, n_pairs)
            tau_result = self._aggregate_single_tau(tau_data)
            tau_results.append(tau_result)

        # Stack results - shape depends on aggregation
        first_result = tau_results[0]
        if np.isscalar(first_result) or (
            hasattr(first_result, "ndim") and first_result.ndim == 0
        ):
            # Each tau → scalar, stack to (n_taus,)
            return np.array([float(r) for r in tau_results])
        else:
            # Each tau → array, stack to (n_taus, ...)
            return np.stack(tau_results, axis=0)
