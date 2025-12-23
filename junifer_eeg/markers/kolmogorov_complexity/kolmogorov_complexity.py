"""Kolmogorov Complexity marker implementation.

Structure (2-file, matching permutation_entropy_new):
- _kolmogorov_complexity_base.py: Singleton with caching + compute logic
- kolmogorov_complexity.py: Main marker extending EEGEpochsMarker (this file)
"""

from typing import Any, ClassVar, Optional, Union

from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker, format_marker_result
from ..utils import filter_to_eeg_channels
from ._kolmogorov_complexity_base import KolmogorovComplexityBase

__all__ = ["KolmogorovComplexity"]


@register_marker
class KolmogorovComplexity(EEGEpochsMarker):
    """Kolmogorov Complexity marker with flexible ROI and trial aggregation.

    This marker computes the Kolmogorov complexity of EEG signals using
    compression-based approximation. The signal is first symbolized into
    discrete bins, then compressed using zlib, and the complexity is
    measured as the compression ratio.

    Supports channel-wise computation with flexible ROI and trial aggregation.

    Uses singleton pattern with caching - multiple KC markers with different
    aggregation methods on the same epochs will reuse cached KC values.

    Parameters
    ----------
    nbins : int, default=16
        Number of bins for signal symbolization.
    rois : list of str or int, optional
        Channel specifications for filtering BEFORE computation.
        Each item can be:
        - int: channel index (e.g., 0, 1, 223)
        - str: channel name (e.g., 'E1', 'E224') OR semantic ROI (e.g., 'scalp')
        If None, uses all channels.
    channel_method : str, optional
        Method for channel aggregation: 'mean', 'median', 'std', 'trim_mean80',
        'trim_mean90', etc. If None, no channel aggregation.
    trial_method : str, optional
        Method for trial aggregation: 'mean', 'median', 'std', 'trim_mean80',
        'trim_mean90', etc. If None, no trial aggregation.
    tmin : float, optional
        Start time for analysis. If None, use start of epochs.
    tmax : float, optional
        End time for analysis. If None, use end of epochs.
    equipment : str, default="egi256"
        Equipment configuration.
    on : str, optional
        Data type (default "EEG").
    name : str, optional
        Marker name.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"kolmogorovcomplexity": "timeseries"}
    }

    def __init__(
        self,
        nbins: int = 16,
        rois: Union[list[str], list[int], None] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize Kolmogorov Complexity marker."""
        super().__init__(
            tmin=tmin, tmax=tmax, equipment=equipment, on=on, name=name
        )

        self.nbins = nbins
        self.rois = rois
        self.channel_method = channel_method
        self.trial_method = trial_method

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute Kolmogorov complexity with flexible aggregation.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed Kolmogorov complexity features with keys:
            - 'data': array of shape (n_epochs, n_channels)
            - 'col_names': channel names
        """
        logger.debug("Computing Kolmogorov complexity")

        # Get and validate epochs
        data_obj = input["data"]
        self._validate_input(data_obj)

        # Filter to EEG channels only
        data_obj, _, _ = filter_to_eeg_channels(data_obj)
        ch_names = data_obj.ch_names

        # Apply ROI filtering BEFORE computation if specified
        if self.rois is not None:
            from ..utils import apply_roi_filtering_to_epochs

            data_obj, ch_names = apply_roi_filtering_to_epochs(
                data_obj,
                self.rois,
                self.equipment,
                marker_name="KolmogorovComplexity",
            )

        # Compute KC using singleton (with caching)
        kc_base = KolmogorovComplexityBase()
        kc_values = kc_base.compute(
            data_obj,
            self.nbins,
            self.tmin,
            self.tmax,
        )

        output_data = kc_values
        output_ch_names = ch_names

        # Apply aggregation using common helper
        from ..utils import apply_channel_trial_aggregation

        output_data = apply_channel_trial_aggregation(
            output_data, self.channel_method, self.trial_method
        )

        # Use centralized format_marker_result to ensure consistent col_names storage
        return format_marker_result(
            feature_name="kolmogorovcomplexity",
            data=output_data,
            col_names=list(output_ch_names),
            channel_aggregated=(self.channel_method is not None),
        )
