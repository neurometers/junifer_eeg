"""Permutation entropy at multiple temporal scales."""

from typing import Any, ClassVar, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.utils import logger

from ..base import EEGEpochsMarker
from ..utils import filter_to_eeg_channels
from ._permutation_entropy_base import PermutationEntropyBase

__all__ = ["PermutationEntropy"]


@register_marker
class PermutationEntropy(EEGEpochsMarker):
    """Permutation entropy at multiple temporal scales.

    Computes permutation entropy at different temporal scales (tau values).
    Uses adaptive lowpass filtering: filter_freq = sfreq / (kernel * tau).
    Does not perform channel or trial aggregation - use PermutationEntropyROIs for that.

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
        rois: Optional[list[str] | list[int]] = None,
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
        sfreq = data_obj.info["sfreq"]

        # Apply ROI filtering BEFORE computation if specified
        if self.rois is not None:
            import mne

            from ..utils import get_data_for_rois

            # Get equipment from data metadata
            description = data_obj.info.get("description") or ""
            if "equipment=" in description:
                equipment = description.replace("equipment=", "")
            else:
                equipment = self.equipment

            # Get data as (n_epochs, n_channels, n_times)
            data_array = data_obj.get_data()

            # Transpose to (n_channels, n_epochs, n_times)
            data_transposed = data_array.transpose(1, 0, 2)

            # Get ROI-filtered data
            roi_data_dict = get_data_for_rois(
                data_transposed,
                ch_names,
                self.rois,
                equipment,
            )

            # Extract the filtered data
            if "selected_channels" in roi_data_dict:
                data_filtered = roi_data_dict["selected_channels"]
                # Transpose back to (n_epochs, n_channels, n_times)
                data_filtered = data_filtered.transpose(1, 0, 2)

                # Create minimal info for filtered channels
                n_channels_filtered = data_filtered.shape[1]
                # Validate that ROI count matches filtered data shape
                if len(self.rois) != n_channels_filtered:
                    raise ValueError(
                        f"ROI count mismatch: {len(self.rois)} != {n_channels_filtered} in PermutationEntropy"
                    )
                # Use actual ROI channel names instead of default 'ch_0' names to preserve channel identity
                info = mne.create_info(
                    ch_names=self.rois,
                    sfreq=sfreq,
                    ch_types="eeg",
                )

                # Create new Epochs object with filtered channels
                data_obj = mne.EpochsArray(
                    data_filtered,
                    info,
                    events=data_obj.events,
                    tmin=data_obj.tmin,
                    verbose=False,
                )
                ch_names = data_obj.ch_names

        # Compute PE using base (with caching potential)
        pe_base = PermutationEntropyBase()

        # Compute PE for each tau value (different temporal scales)
        # Always use adaptive lowpass filtering: filter_freq = sfreq / (kernel * tau)
        all_tau_pe = {}
        for tau in self.taus:
            logger.debug(f"Computing PE for tau={tau}")

            pe_values = pe_base._compute_pe(
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

        # Format output
        if len(all_tau_pe) == 1:
            # Single tau - return 2D array (n_epochs, n_channels)
            tau_name = next(iter(all_tau_pe.keys()))
            return {
                "permutationentropy": {
                    "data": all_tau_pe[tau_name],
                    "col_names": ch_names,
                }
            }

        # Multiple taus - stack into 3D tensor (n_taus, n_epochs, n_channels)
        tau_order = [f"tau_{tau}" for tau in self.taus]
        tensor = np.stack(
            [all_tau_pe[tau_name] for tau_name in tau_order],
            axis=0,
        )

        return {
            "permutationentropy": {
                "data": tensor,
                "col_names": ch_names,
            }
        }
