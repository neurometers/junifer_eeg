"""Symbolic mutual information marker (no aggregation)."""

from typing import Any, ClassVar, Optional

import numpy as np
from junifer.api.decorators import register_marker
from junifer.utils import logger
from mne.utils import _time_mask
from scipy.signal import butter, filtfilt

from ..base import EEGEpochsMarker
from ..utils import (
    create_connectivity_pair_column_names,
    filter_to_eeg_channels,
)
from ._symbolic_mutual_information_base import SymbolicMutualInformationBase

__all__ = ["SymbolicMutualInformation"]


@register_marker
class SymbolicMutualInformation(EEGEpochsMarker):
    """Symbolic mutual information marker for connectivity analysis.

    Computes symbolic mutual information between EEG channels without aggregation.
    Use SymbolicMutualInformationROIs for channel/trial aggregation.

    This implementation matches NICE's WSMI algorithm with:
    - Optional CSD preprocessing
    - Adaptive or custom frequency filtering
    - Symbolic transformation
    - Weighted or unweighted mutual information

    Parameters
    ----------
    kernel : int, default=3
        Length of ordinal patterns.
    taus : int or list of int, default=8
        Time delay(s) for ordinal patterns. Can be single integer or list of integers
        for multiple temporal scales.
    weighted : bool, default=True
        Whether to compute weighted SMI (True) or unweighted SMI (False).
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
        equipment: str = "egi256",
        on: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Initialize SMI marker."""
        super().__init__(
            tmin=tmin, tmax=tmax, equipment=equipment, on=on, name=name
        )

        # Convert single tau to list and validate
        taus_list = [taus] if isinstance(taus, int) else list(taus)
        if len(taus_list) == 0:
            raise ValueError("taus parameter cannot be empty")
        self.taus = taus_list
        self.kernel = kernel
        self.weighted = weighted
        self.csd = csd
        self.rois = rois

    def compute(
        self,
        input: dict[str, Any],
        extra_input: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Compute symbolic mutual information.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Epochs object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            SMI connectivity with keys:
            - 'data': array of shape (n_epochs, n_channel_pairs)
            - 'col_names': channel pair names

        """
        logger.debug("Computing symbolic mutual information")

        # Get and validate epochs
        data_obj = input["data"]
        self._validate_input(data_obj)

        # Filter to EEG channels only (E1-E256)
        data_obj, eeg_ch_names, eeg_indices = filter_to_eeg_channels(data_obj)

        # Validate parameters
        if self.kernel <= 1:
            raise ValueError(
                f"kernel (pattern length) must be > 1, got {self.kernel}"
            )
        for tau in self.taus:
            if tau <= 0:
                raise ValueError(f"tau (delay) must be > 0, got {tau}")

        epochs = data_obj
        sfreq = epochs.info["sfreq"]

        # Apply CSD preprocessing if requested
        if self.csd:
            from mne import pick_types
            from mne.preprocessing import compute_current_source_density

            if (
                "eeg" in epochs
                and pick_types(epochs.info, meg=False, eeg=True).size > 0
            ):
                epochs_temp = epochs.copy()
                if epochs_temp.info["bads"]:
                    epochs_temp.interpolate_bads(reset_bads=True)

                epochs_csd = compute_current_source_density(
                    epochs_temp, lambda2=1e-5
                )

                csd_picks = pick_types(epochs_csd.info, csd=True)
                if len(csd_picks) > 0:
                    epochs = epochs_csd

        # Pick data channels
        from mne import pick_types

        picks = pick_types(
            epochs.info,
            meg=True,
            eeg=True,
            csd=True,
            seeg=True,
            ecog=True,
            ref_meg=False,
            exclude="bads",
        )

        if len(picks) == 0:
            raise ValueError(
                "No suitable channels found after picking. Check channel types."
            )

        data_for_comp = epochs.get_data(picks=picks)
        picked_ch_names = [epochs.ch_names[i] for i in picks]
        n_epochs, n_channels, n_times = data_for_comp.shape

        # Apply ROI filtering BEFORE connectivity computation if specified
        if self.rois is not None:
            from ..utils import get_data_for_rois

            # Get equipment from data metadata
            description = epochs.info.get("description") or ""
            if "equipment=" in description:
                equipment = description.replace("equipment=", "")
            else:
                equipment = self.equipment

            # Transpose to (n_channels, n_epochs, n_times)
            data_transposed = data_for_comp.transpose(1, 0, 2)

            # Get ROI-filtered data
            roi_data_dict = get_data_for_rois(
                data_transposed,
                picked_ch_names,
                self.rois,
                equipment,
            )

            # Extract the filtered data
            if "selected_channels" in roi_data_dict:
                data_filtered = roi_data_dict["selected_channels"]
                # Transpose back to (n_epochs, n_channels, n_times)
                data_for_comp = data_filtered.transpose(1, 0, 2)
                n_channels = data_for_comp.shape[1]
                # Update channel names
                picked_ch_names = [f"ch_{i}" for i in range(n_channels)]

        if n_channels < 2:
            raise ValueError(
                f"At least 2 channels required for connectivity, got {n_channels}"
            )

        # Compute SMI for each tau value (different temporal scales)
        # Always use adaptive lowpass filtering: filter_freq = sfreq / (kernel * tau)
        all_tau_smi = {}
        smi_base = SymbolicMutualInformationBase()
        col_names = create_connectivity_pair_column_names(picked_ch_names)

        for tau in self.taus:
            logger.debug(f"Computing SMI for tau={tau}")

            # Apply adaptive lowpass filtering based on tau
            # filter_freq = sfreq / (kernel * tau)
            filter_freq = np.double(sfreq) / self.kernel / tau
            b, a = butter(
                6,  # filter_order=6 (NICE default)
                2.0 * filter_freq / np.double(sfreq),
                "lowpass",
            )

            # Concatenate, filter, split back (NICE approach)
            data_concatenated = np.hstack(data_for_comp)
            fdata_concatenated = filtfilt(b, a, data_concatenated)
            fdata = np.transpose(
                np.array(np.split(fdata_concatenated, n_epochs, axis=1)),
                [1, 2, 0],
            )  # (channels, times, epochs)

            # Apply time mask
            time_mask = _time_mask(epochs.times, self.tmin, self.tmax)
            fdata_masked = fdata[:, time_mask, :]

            # Check if enough samples for symbolization
            min_samples = tau * (self.kernel - 1) + 1
            if fdata_masked.shape[1] < min_samples:
                raise ValueError(
                    f"After time masking, {fdata_masked.shape[1]} samples available, "
                    f"but {min_samples} needed for kernel={self.kernel}, tau={tau}"
                )

            # Compute SMI using base class
            connectivity_matrix = smi_base._compute_smi(
                fdata_masked, self.kernel, tau, self.weighted
            )

            # Convert to per-epoch format: (n_epochs, n_channel_pairs)
            indices_use = np.triu_indices(n_channels, k=1)

            # Extract upper triangular values for each epoch
            epoch_data = []
            for epoch_idx in range(n_epochs):
                epoch_matrix = connectivity_matrix[:, :, epoch_idx]
                upper_tri_values = epoch_matrix[indices_use]
                epoch_data.append(upper_tri_values)

            all_tau_smi[f"tau_{tau}"] = np.array(epoch_data)

        # Format output
        if len(all_tau_smi) == 1:
            # Single tau - return 2D array (n_epochs, n_channel_pairs)
            tau_name = next(iter(all_tau_smi.keys()))
            return {
                "symbolicmutualinformation": {
                    "data": all_tau_smi[tau_name],
                    "col_names": col_names,
                }
            }

        # Multiple taus - stack into 3D tensor (n_taus, n_epochs, n_channel_pairs)
        tau_order = [f"tau_{tau}" for tau in self.taus]
        tensor = np.stack(
            [all_tau_smi[tau_name] for tau_name in tau_order],
            axis=0,
        )

        return {
            "symbolicmutualinformation": {
                "data": tensor,
                "col_names": col_names,
            }
        }
