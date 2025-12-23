"""Permutation entropy at multiple temporal scales."""

from typing import Any, ClassVar, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker, format_marker_result
from ..utils import filter_to_eeg_channels
from ._permutation_entropy_base import PermutationEntropyBase

__all__ = ["PermutationEntropy"]


@register_marker
class PermutationEntropy(EEGEpochsMarker):
    """Permutation entropy at multiple temporal scales with optional aggregation.

    Computes permutation entropy at different temporal scales (tau values).
    Uses adaptive lowpass filtering: filter_freq = sfreq / (kernel * tau).

    Parameters
    ----------
    taus : int or list of int, default=8
        Time delay(s) for ordinal patterns. Different tau values represent
        different temporal scales. Can be single int or list for multiple scales.
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
    kernel : int, default=3
        Length of ordinal patterns.
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

    _DEPENDENCIES: ClassVar = {"mne", "numpy", "scipy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"permutationentropy": "timeseries"}
    }

    def __init__(
        self,
        taus: int | list[int] = 8,
        rois: Union[list[str], list[int], None] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        kernel: int = 3,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize permutation entropy marker."""
        super().__init__(
            tmin=tmin, tmax=tmax, equipment=equipment, on=on, name=name
        )

        # Convert single tau to list
        self.taus = [taus] if isinstance(taus, int) else list(taus)
        self.rois = rois
        self.channel_method = channel_method
        self.trial_method = trial_method
        self.kernel = kernel

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute permutation entropy at multiple temporal scales.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Permutation entropy with keys:
            - 'data': array of shape (n_epochs, n_channels) for single tau
                      or (n_taus, n_epochs, n_channels) for multiple taus
            - 'col_names': channel names

        """
        logger.debug("Computing permutation entropy")

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
                marker_name="PermutationEntropy",
            )

        # Compute PE using base (with caching potential)
        pe_base = PermutationEntropyBase()

        # Compute PE for each tau value (different temporal scales)
        # Always use adaptive lowpass filtering: filter_freq = sfreq / (kernel * tau)
        all_tau_pe = {}
        for tau in self.taus:
            logger.debug(f"Computing PE for tau={tau}")

            pe_values = pe_base.compute(
                data_obj,
                self.kernel,
                tau,
                None,  # fmin=None triggers adaptive filter
                None,  # fmax=None triggers adaptive filter
                6,  # filter_order=6 (NICE default)
                self.tmin if self.tmin is not None else data_obj.tmin,
                self.tmax if self.tmax is not None else data_obj.tmax,
            )

            all_tau_pe[f"tau_{tau}"] = pe_values

        # Format output - unify single and multiple taus then apply aggregation
        if len(all_tau_pe) == 1:
            # Single tau - shape (n_epochs, n_channels)
            tau_name = next(iter(all_tau_pe.keys()))
            output_data = all_tau_pe[tau_name]
        else:
            # Multiple taus - stack into tensor (n_taus, n_epochs, n_channels)
            tau_order = [f"tau_{tau}" for tau in self.taus]
            output_data = np.stack(
                [all_tau_pe[tau_name] for tau_name in tau_order],
                axis=0,
            )

        output_ch_names = ch_names

        # Apply aggregation using common helper
        from ..utils import apply_channel_trial_aggregation

        output_data = apply_channel_trial_aggregation(
            output_data, self.channel_method, self.trial_method
        )

        # Use centralized format_marker_result to ensure consistent col_names storage
        return format_marker_result(
            feature_name="permutationentropy",
            data=output_data,
            col_names=output_ch_names,
            channel_aggregated=(self.channel_method is not None),
        )
