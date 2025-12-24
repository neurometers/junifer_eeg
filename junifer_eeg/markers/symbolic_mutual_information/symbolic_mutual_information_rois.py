"""Symbolic mutual information with ROI aggregation."""

from typing import Any, ClassVar, Optional

from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker, format_marker_result
from ..utils import apply_aggregation_preserve_dims
from .symbolic_mutual_information import SymbolicMutualInformation

__all__ = ["SymbolicMutualInformationROIs"]


@register_marker
class SymbolicMutualInformationROIs(EEGEpochsMarker):
    """Symbolic mutual information with ROI aggregation.

    Always returns a 4D tensor with shape (n_taus, n_epochs, n_channels, n_channels).
    Aggregation parameters control which dimensions have size 1, but the structure
    is preserved for consistency.

    Examples:
    - No aggregation: (n_taus, n_epochs, n_channels, n_channels)
    - Channel aggregation: (n_taus, n_epochs, 1, n_channels)
    - Epoch aggregation: (n_taus, 1, n_channels, n_channels)
    - Full aggregation: (1, 1, 1, 1) for scalar values

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
        "EEG": {"symbolicmutualinformation": "timeseries"}
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

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute SMI with aggregation.

        Always returns a 4D tensor with shape (n_taus, n_epochs, n_channels, n_channels).
        Aggregation methods control which dimensions have size 1.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Aggregated SMI results as 4D tensor.

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

        # Get the 4D tensor from base marker: (n_taus, n_epochs, n_channels, n_channels)
        connectivity_data = result["symbolicmutualinformation"]["data"]

        # Apply aggregation while preserving 4D structure
        output_data = apply_aggregation_preserve_dims(
            connectivity_data,
            channel_method=self.channel_method,
            trial_method=self.trial_method,
            connectivity_method=self.connectivity_method,
        )

        # Format output using centralized format_marker_result
        # Use channel names for the full matrix representation
        return format_marker_result(
            feature_name="symbolicmutualinformation",
            data=output_data,
            col_names=result["symbolicmutualinformation"]["col_names"],
            channel_aggregated=True,  # Some form of aggregation may be applied
        )
