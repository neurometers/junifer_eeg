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
    connectivity_method : str or list of str, optional
        Method(s) to aggregate across connectivity dimension (channel_y).
        Options: 'median', 'mean', 'std', 'trim_mean', etc.
    channel_method : str or list of str, optional
        Method(s) to aggregate across channel dimension (channel_x).
    channel_method_params : dict, optional
        Parameters for channel aggregation method.
    trial_method : str or list of str, optional
        Method(s) to aggregate across trial/epoch dimension.
    trial_method_params : dict, optional
        Parameters for trial aggregation method.
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
        connectivity_method: Optional[str | list[str]] = None,
        channel_method: Optional[str | list[str]] = None,
        channel_method_params: Optional[dict] = None,
        trial_method: Optional[str | list[str]] = None,
        trial_method_params: Optional[dict] = None,
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
        self.channel_method_params = channel_method_params or {}
        self.trial_method = trial_method
        self.trial_method_params = trial_method_params or {}

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

        # Get connectivity data
        connectivity_data = result["symbolicmutualinformation"]["data"]

        # Handle single vs multiple taus
        if connectivity_data.ndim == 2:
            # Single tau - original logic
            return self._aggregate_single_tau(connectivity_data)
        else:
            # Multiple taus - loop through each tau
            return self._aggregate_multiple_taus(connectivity_data)

    def _aggregate_single_tau(
        self, connectivity_data: np.ndarray
    ) -> dict[str, Any]:
        """Aggregate connectivity data for single tau."""
        # If no aggregation requested, return raw connectivity
        if (
            self.connectivity_method is None
            and self.channel_method is None
            and self.trial_method is None
        ):
            return {"symbolicmutualinformation": {"data": connectivity_data}}

        # Apply aggregation pipeline (original logic)
        from ..utils import aggregate_data

        n_epochs, n_pairs = connectivity_data.shape

        # Determine number of channels from n_pairs
        import numpy as np

        n_channels = int((1 + np.sqrt(1 + 8 * n_pairs)) / 2)

        # Reconstruct full connectivity matrices for aggregation
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

        # Transpose to (n_channels, n_channels, n_epochs) for aggregation
        current_data = connectivity_tensor.transpose(1, 2, 0)

        # Step 1: Aggregate across connectivity dimension (channel_y, axis=1)
        if self.connectivity_method is not None:
            current_data = aggregate_data(
                current_data, self.connectivity_method, axis=1
            )

        # Step 2: Aggregate across channels (channel_x, axis=0)
        if self.channel_method is not None:
            if self.channel_method_params:
                pass  # TODO: Support method params
            current_data = aggregate_data(
                current_data, self.channel_method, axis=0
            )
        else:
            if current_data.ndim == 2:
                current_data = current_data.T  # (n_epochs, n_channels)

        # Step 3: Aggregate across trials (epochs)
        if self.trial_method is not None:
            if self.trial_method_params:
                pass  # TODO: Support method params
            if current_data.ndim > 0:
                epoch_axis = 2 if current_data.ndim == 3 else 0
                current_data = aggregate_data(
                    current_data, self.trial_method, axis=epoch_axis
                )

        # Format output
        if np.isscalar(current_data):
            return {"symbolicmutualinformation": {"data": float(current_data)}}
        elif hasattr(current_data, "size") and current_data.size == 1:
            return {
                "symbolicmutualinformation": {
                    "data": float(current_data.item())
                }
            }
        else:
            return {"symbolicmutualinformation": {"data": current_data}}

    def _aggregate_multiple_taus(
        self, connectivity_data: np.ndarray
    ) -> dict[str, Any]:
        """Aggregate connectivity data for multiple taus."""
        n_taus, n_epochs, n_pairs = connectivity_data.shape

        # Convert taus to list for iteration
        taus_list = (
            [self.taus] if isinstance(self.taus, int) else list(self.taus)
        )

        # Process each tau separately
        results = {}
        for tau_idx, tau in enumerate(taus_list):
            tau_data = connectivity_data[tau_idx]  # (n_epochs, n_pairs)
            tau_result = self._aggregate_single_tau(tau_data)
            results[f"tau_{tau}"] = tau_result["symbolicmutualinformation"][
                "data"
            ]

        return {"symbolicmutualinformation": results}
