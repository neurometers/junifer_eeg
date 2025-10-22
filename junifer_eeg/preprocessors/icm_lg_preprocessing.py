"""ICM Local-Global Equipment-Specific Preprocessing for EEG data.

This module provides preprocessing components that closely follow the original
next_icm/lg/preprocessing.py implementation, reusing the same functions and
logic wherever possible.
"""

import pickle
from pathlib import Path
from typing import Any, Optional

import mne
import numpy as np
from junifer.api.decorators import register_preprocessor
from junifer.preprocess.base import BasePreprocessor
from mne.utils import logger

from .utils import (
    EQUIPMENT_FILTER_PARAMS,
    ICM_LG_EVENT_ID,
    _adaptive_egi,
    _check_min_channels,
    _check_min_events,
)


def _dump_data(data, element, stage, step_num, dump_location):
    """Helper function to dump preprocessing data."""
    if not dump_location:
        return

    dump_path = Path(dump_location) / element
    dump_path.mkdir(parents=True, exist_ok=True)

    # Dump EEG data in native MNE format
    if isinstance(data.get("data"), (mne.io.BaseRaw, mne.BaseEpochs)):
        eeg_file = dump_path / f"{step_num:02d}_{stage}_eeg.fif"
        data["data"].save(eeg_file, overwrite=True)
        logger.info(f"Dumped EEG data: {eeg_file}")

    # Dump metadata
    metadata = {k: v for k, v in data.items() if k != "data"}
    if metadata:
        meta_file = dump_path / f"{step_num:02d}_{stage}_metadata.pkl"
        with open(meta_file, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Dumped metadata: {meta_file}")


@register_preprocessor
class ICMEquipmentFilter(BasePreprocessor):
    """ICM Equipment-specific EEG filtering with self-contained implementation.

    This preprocessor applies the exact same filtering parameters and logic as the
    original next_icm implementation, but uses a self-contained MNE-based approach
    without any external dependencies on nice_ext.

    Parameters
    ----------
    equipment_type : str, optional
        Equipment type ('egi', 'brainvision', 'ant', 'biosemi'). Default: 'egi'
    config_params : dict, optional
        Additional configuration parameters passed to the filtering functions.
    n_jobs : int, optional
        Number of parallel jobs for filtering. Default: 1
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        equipment_type: str = "egi",
        config_params: Optional[dict[str, Any]] = None,
        n_jobs: int = 1,
        dump_location: Optional[str] = None,
        dump_granularity: str = "full",
        on: Optional[list[str]] = None,
    ):
        self.equipment_type = equipment_type
        self.config_params = config_params or {}
        self.n_jobs = n_jobs
        self.dump_location = dump_location
        self.dump_granularity = dump_granularity
        super().__init__(on=on)

    def get_valid_inputs(self) -> list[str]:
        """Get valid input types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Apply equipment-specific filtering using original next_icm functions."""
        raw = input["data"]

        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("Input data must be mne.io.BaseRaw")

        # Debug: Check annotations received by preprocessor
        from collections import Counter

        desc_counts = Counter(raw.annotations.description)
        icm_conditions = {
            desc: count
            for desc, count in desc_counts.items()
            if desc in {"HSTD", "HDVT", "LSGS", "LSGD", "LDGD", "LDGS"}
        }
        logger.info(
            f"[PREPROCESSOR DEBUG] Received raw with {sum(icm_conditions.values())} ICM events"
        )
        logger.info(f"[PREPROCESSOR DEBUG] ICM conditions: {icm_conditions}")

        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Debug: Check annotations after copy
        desc_counts_after = Counter(raw.annotations.description)
        icm_after = {
            desc: count
            for desc, count in desc_counts_after.items()
            if desc in {"HSTD", "HDVT", "LSGS", "LSGD", "LDGD", "LDGS"}
        }
        logger.info(
            f"[PREPROCESSOR DEBUG] After copy: {sum(icm_after.values())} ICM events"
        )

        # Re-apply montage if needed (Junifer loses dig points in data transfer)
        # This is critical for bad channel interpolation later
        if self.equipment_type == "egi":
            if (
                raw.info.get("dig") is None
                or len(raw.info.get("dig", [])) == 0
            ):
                logger.info(
                    "Re-applying montage (dig points lost in data transfer)"
                )
                montage = mne.channels.make_standard_montage(
                    "GSN-HydroCel-256"
                )
                raw.set_montage(montage, on_missing="ignore")
                logger.info(
                    f"Montage applied: {len(raw.info['dig'])} dig points"
                )

        # Get equipment-specific parameters
        if self.equipment_type not in EQUIPMENT_FILTER_PARAMS:
            raise ValueError(
                f"Unsupported equipment type: {self.equipment_type}"
            )

        params = EQUIPMENT_FILTER_PARAMS[self.equipment_type].copy()
        # Override with any user-provided parameters
        params.update(self.config_params)

        # Apply equipment-specific filtering using self-contained implementation
        self._apply_equipment_filtering(raw, params)

        output = input.copy()
        output["data"] = raw

        # Dump data if requested
        if self.dump_location:
            element = input.get("meta", {}).get("element", "unknown_element")
            if isinstance(element, dict):
                element = "unknown_element"
            _dump_data(
                output, element, "equipment_filtered", 1, self.dump_location
            )

        return output, None

    def _apply_equipment_filtering(self, raw, params):
        """Apply equipment-specific filtering using self-contained MNE implementation."""
        # Get parameters with defaults
        lpass = params.get("lpass", 40.0)
        hpass = params.get("hpass", 0.5)
        notches = params.get("notches", [50, 100])
        hp_order = params.get("hp_order", 4)
        lp_order = params.get("lp_order", 8)
        l_trans_bandwidth = params.get("l_trans_bandwidth", 0.1)

        # Pick EEG channels
        picks = mne.pick_types(
            raw.info, eeg=True, meg=True, ecg=True, exclude=[]
        )

        # Apply high-pass filter (Butterworth)
        if hpass is not None:
            logger.info(
                f"Applying high-pass filter at {hpass} Hz (order {hp_order})"
            )
            raw.filter(
                l_freq=hpass,
                h_freq=None,
                picks=picks,
                method="iir",
                iir_params={"ftype": "butter", "order": hp_order},
                l_trans_bandwidth=l_trans_bandwidth,
                n_jobs=self.n_jobs,
            )

        # Apply low-pass filter (Butterworth)
        if lpass is not None:
            logger.info(
                f"Applying low-pass filter at {lpass} Hz (order {lp_order})"
            )
            raw.filter(
                l_freq=None,
                h_freq=lpass,
                picks=picks,
                method="iir",
                iir_params={"ftype": "butter", "order": lp_order},
                n_jobs=self.n_jobs,
            )

        # Apply notch filters
        if notches:
            # For BrainVision, add 200Hz notch if sampling rate > 400Hz
            if (
                self.equipment_type == "brainvision"
                and raw.info["sfreq"] > 400
            ):
                notches = [*notches, 200]

            # Filter out notch frequencies that are above current Nyquist frequency
            current_nyquist = raw.info["sfreq"] / 2.0
            valid_notches = [f for f in notches if f < current_nyquist]

            if (
                valid_notches
            ):  # Only apply if there are valid notch frequencies
                logger.info(f"Applying notch filters at {valid_notches} Hz")
                raw.notch_filter(
                    valid_notches, method="fft", n_jobs=self.n_jobs
                )

        # Resample to target frequency (EGI always resamples to 250Hz)
        resample_freq = params.get("resample_freq")
        if resample_freq and raw.info["sfreq"] != resample_freq:
            logger.info(f"Resampling to {resample_freq} Hz")
            # Modern MNE preserves digitization points automatically
            raw.resample(resample_freq, npad="auto")


@register_preprocessor
class ICMAdaptiveArtifactRejection(BasePreprocessor):
    """ICM Adaptive artifact rejection using original next_icm _adaptive_egi function.

    This preprocessor uses the exact same _adaptive_egi function from nice_ext
    as the original next_icm implementation.

    Parameters
    ----------
    reject : dict, optional
        Rejection criteria. Default: {'eeg': 100e-6}
    n_epochs_bad_ch : float, optional
        Fraction of epochs for bad channel detection. Default: 0.5
    n_channels_bad_epoch : float, optional
        Fraction of channels for bad epoch detection. Default: 0.1
    zscore_thresh : float, optional
        Z-score threshold for artifact detection. Default: 4
    max_iter : int, optional
        Maximum iterations for adaptive algorithm. Default: 4
    min_channels : float, optional
        Minimum fraction of good channels required. Default: 0.7
    min_events : float, optional
        Minimum fraction of good events required. Default: 0.3
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        reject: Optional[dict[str, float]] = None,
        n_epochs_bad_ch: float = 0.5,
        n_channels_bad_epoch: float = 0.1,
        zscore_thresh: float = 4,
        max_iter: int = 4,
        min_channels: float = 0.7,
        min_events: float = 0.3,
        interpolate_bads: bool = True,
        dump_location: Optional[str] = None,
        dump_granularity: str = "full",
        on: Optional[list[str]] = None,
    ):
        self.reject = reject or {"eeg": 100e-6}
        self.n_epochs_bad_ch = n_epochs_bad_ch
        self.n_channels_bad_epoch = n_channels_bad_epoch
        self.zscore_thresh = zscore_thresh
        self.max_iter = max_iter
        self.min_channels = min_channels
        self.min_events = min_events
        self.interpolate_bads = interpolate_bads
        self.dump_location = dump_location
        self.dump_granularity = dump_granularity
        super().__init__(on=on)

    def get_valid_inputs(self) -> list[str]:
        """Get valid input types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Apply adaptive artifact rejection using original next_icm functions."""
        epochs = input["data"]

        if not isinstance(epochs, mne.BaseEpochs):
            raise ValueError("Input data must be mne.BaseEpochs")

        # Check if we have any epochs to process
        if len(epochs) == 0:
            logger.warning(
                "No epochs available for artifact rejection. Returning empty epochs."
            )
            output = input.copy()
            output["data"] = epochs
            return output, None

        # Store original epoch indices before any dropping
        original_epoch_indices = list(range(len(epochs)))

        bad_channels, bad_epochs = _adaptive_egi(
            epochs,
            self.reject,
            n_epochs_bad_ch=self.n_epochs_bad_ch,
            n_channels_bad_epoch=self.n_channels_bad_epoch,
            zscore_thresh=self.zscore_thresh,
            max_iter=self.max_iter,
        )

        # Store bad epochs/channels info in epochs metadata for report generation
        epochs.info["description"] = "ICM LG preprocessed data"

        # Store bad channels info (before interpolation)
        all_bad_channels = list(set(epochs.info["bads"] + bad_channels))
        epochs.info["bads"].extend(bad_channels)

        # Store preprocessing info in epochs.info for later access
        # Store bad epochs and bad channels info in epochs metadata
        # Use MNE's approved 'temp' key for custom info storage
        if "temp" not in epochs.info:
            epochs.info["temp"] = {}
        epochs.info["temp"]["preprocessing_info"] = {}

        epochs.info["temp"]["preprocessing_info"]["bad_channels_detected"] = (
            bad_channels
        )
        epochs.info["temp"]["preprocessing_info"]["bad_channels_total"] = (
            all_bad_channels
        )
        epochs.info["temp"]["preprocessing_info"]["bad_epochs_detected"] = (
            bad_epochs
        )
        epochs.info["temp"]["preprocessing_info"][
            "n_epochs_before_rejection"
        ] = len(original_epoch_indices)
        epochs.info["temp"]["preprocessing_info"][
            "n_epochs_after_rejection"
        ] = len(epochs)
        epochs.info["temp"]["preprocessing_info"][
            "n_channels_interpolated"
        ] = len(all_bad_channels)

        logger.info(
            f"found bad channels: {len(bad_channels)} {bad_channels!s}"
        )
        logger.info(f"found bad epochs: {len(bad_epochs)} epochs")

        _check_min_events(epochs, self.min_events)
        _check_min_channels(epochs, bad_channels, self.min_channels)

        # Apply average reference
        # This must be done BEFORE interpolation for correct results
        # Modern MNE equivalent of: ref_proj = mne.proj.make_eeg_average_ref_proj(epochs.info)
        epochs.set_eeg_reference("average", projection=True)
        epochs.apply_proj()

        # Interpolate bad channels
        if self.interpolate_bads and len(epochs.info["bads"]) > 0:
            logger.info(
                f"Interpolating {len(epochs.info['bads'])} bad channels"
            )
            epochs.interpolate_bads(reset_bads=True)
        elif not self.interpolate_bads and len(epochs.info["bads"]) > 0:
            logger.info(
                f"Skipping interpolation for {len(epochs.info['bads'])} bad channels (disabled)"
            )

        output = input.copy()
        output["data"] = epochs

        # Dump data if requested - save only essential metadata for visualization
        if self.dump_location:
            element = input.get("meta", {}).get("element", "unknown_element")
            if isinstance(element, dict):
                element = "unknown_element"

            # Save only essential metadata for visualization (no large files)
            dump_path = Path(self.dump_location) / element
            dump_path.mkdir(parents=True, exist_ok=True)

            # Save only preprocessing metadata (bad channels list)
            preprocessing_info = {
                "bad_channels_detected": bad_channels,
                "bad_epochs_detected": bad_epochs,
                "n_epochs_before_rejection": len(original_epoch_indices),
                "n_epochs_after_rejection": len(epochs),
                "n_channels_interpolated": len(all_bad_channels),
            }

            metadata = {"preprocessing_info": preprocessing_info}

            meta_file = dump_path / "03_bad_channels_metadata.pkl"
            with open(meta_file, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Dumped bad channels metadata: {meta_file}")
            logger.info(
                f"Bad channels: {len(bad_channels)}, Bad epochs: {len(bad_epochs)}"
            )

            # Save final clean epochs after artifact rejection (stage 03)
            eeg_file = dump_path / "03_artifact_rejected_eeg.fif"
            epochs.save(eeg_file, overwrite=True)
            logger.info(
                f"Dumped clean epochs after artifact rejection: {eeg_file}"
            )
            logger.info(
                f"Clean epochs: {len(epochs)}, Projections: {len(epochs.info['projs'])}"
            )

        return output, None


@register_preprocessor
class ICMLGEpoching(BasePreprocessor):
    """ICM Local-Global epoching preprocessor using original next_icm approach.

    This preprocessor follows the exact same epoching logic as the original
    next_icm implementation, including event detection and channel handling.

    Parameters
    ----------
    tmin : float, optional
        Start time before event. Default: -0.2
    tmax : float, optional
        End time after event. Default: 1.34
    baseline : tuple of float, optional
        Baseline correction period. Default: (None, 0)
    event_id : dict, optional
        Event ID mapping. If None, uses ICM LG default mapping.
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 1.34,
        baseline: tuple[Optional[float], float] = (None, 0),
        event_id: Optional[dict[str, int]] = None,
        dump_location: Optional[str] = None,
        dump_granularity: str = "full",
        on: Optional[list[str]] = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        self.event_id = event_id or ICM_LG_EVENT_ID
        self.dump_location = dump_location
        self.dump_granularity = dump_granularity
        super().__init__(on=on)

    def get_valid_inputs(self) -> list[str]:
        """Get valid input types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Create epochs from continuous data using original next_icm approach."""
        raw = input["data"]

        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("Input data must be mne.io.BaseRaw")

        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Find events (matching original next_icm approach)
        # For EDF files with annotations, use events_from_annotations instead
        try:
            events = mne.find_events(raw, shortest_event=1)
            found_id = np.unique(events[:, 2])
            this_id = {k: v for k, v in self.event_id.items() if v in found_id}
        except ValueError:
            # No STI channel found - try reading from annotations (EDF files)
            logger.info(
                "No STI channel found, reading events from annotations"
            )
            events, event_id_from_annot = mne.events_from_annotations(raw)
            # Filter to only include ICM LG condition events
            # The datareader already mapped MARQUEUR codes to condition names (HSTD, HDVT, LSGS, etc.)
            icm_conditions = {"HSTD", "HDVT", "LSGS", "LSGD", "LDGD", "LDGS"}
            this_id = {
                k: v
                for k, v in event_id_from_annot.items()
                if k in icm_conditions
            }

            if not this_id:
                raise ValueError(
                    f"No ICM LG events found in annotations. Available: {list(event_id_from_annot.keys())}"
                ) from None

            logger.info(f"Found ICM conditions: {sorted(this_id.keys())}")
            found_id = np.unique(events[:, 2])

        # Create epochs (matching original next_icm parameters)
        epochs = mne.Epochs(
            raw,
            events,
            this_id,
            tmin=self.tmin,
            tmax=self.tmax,
            preload=True,
            reject=None,  # No rejection at epoching stage (done later in artifact rejection)
            picks=None,
            baseline=self.baseline,
            verbose=False,
        )

        # Filter to keep only E1-E256 EEG channels and STI 014 (matching NICE)
        channels_to_keep = [f"E{i}" for i in range(1, 257)] + ["STI 014"]
        channels_in_epochs = [
            ch for ch in channels_to_keep if ch in epochs.ch_names
        ]
        if len(channels_in_epochs) < len(epochs.ch_names):
            epochs.pick_channels(channels_in_epochs)
            logger.info(
                f"Filtered to {len(channels_in_epochs)} channels (E1-E256 + STI 014)"
            )

        # Handle concatenation events (matching original next_icm)
        # Look for STI 014 channel and remove concatenation events
        if "STI 014" in epochs.ch_names:
            ch_idx = epochs.ch_names.index("STI 014")
            concat_idx = []

            # ICM LG concatenation event constant (from next_icm/lg/constants.py)
            icm_lg_concatenation_event = 2014.0

            for ii, e in enumerate(epochs):
                if icm_lg_concatenation_event in e[ch_idx]:
                    concat_idx.append(ii)

            if concat_idx:
                epochs.drop(concat_idx, reason="concatenation")
                logger.info(f"Dropped {len(concat_idx)} concatenation epochs")

        # Drop stimulus channels after epoching (matching original next_icm)
        if "STI 014" in epochs.ch_names:
            epochs.drop_channels(["STI 014"])

        output = input.copy()
        output["data"] = epochs

        # Note: Epoch file saving moved to ICMAdaptiveArtifactRejection
        # so that projections are included in the saved file

        return output, None
