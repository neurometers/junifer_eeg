"""ICM Local-Global Equipment-Specific Preprocessing for EEG data."""

from typing import Any, Dict, List, Optional, Tuple

import mne
import numpy as np
from junifer.api.decorators import register_preprocessor
from junifer.preprocess.base import BasePreprocessor

# Equipment-specific filtering parameters
EQUIPMENT_FILTER_PARAMS = {
    "egi": {
        "l_freq": 0.1,
        "h_freq": 40.0,
        "notch_freq": 60.0,
        "resample_freq": 250,
    },
    "brainvision": {
        "l_freq": 0.1,
        "h_freq": 30.0,
        "notch_freq": 50.0,
        "resample_freq": 250,
    },
    "ant": {
        "l_freq": 0.1,
        "h_freq": 30.0,
        "notch_freq": 50.0,
        "resample_freq": 250,
    },
    "biosemi": {
        "l_freq": 0.1,
        "h_freq": 30.0,
        "notch_freq": 50.0,
        "resample_freq": 250,
    },
}


@register_preprocessor
class ICMEquipmentFilter(BasePreprocessor):
    """ICM Equipment-specific EEG filtering.

    Parameters
    ----------
    equipment_type : str, optional
        Equipment type for filtering parameters. Default: 'egi'
    l_freq : float, optional
        Low-pass filter frequency.
    h_freq : float, optional
        High-pass filter frequency.
    notch_freq : float, optional
        Notch filter frequency.
    resample_freq : float, optional
        Resampling frequency.
    """

    def __init__(
        self,
        equipment_type: str = "egi",
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        notch_freq: Optional[float] = None,
        resample_freq: Optional[float] = None,
        on: Optional[List[str]] = None,
    ):
        self.equipment_type = equipment_type
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.resample_freq = resample_freq

        super().__init__(on=on)

    def get_valid_inputs(self) -> List[str]:
        """Get valid input types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: Dict[str, Any],
        extra_input: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
        """Apply equipment-specific filtering."""
        raw = input["data"]

        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("Input data must be mne.io.BaseRaw")

        # Get equipment-specific parameters
        params = EQUIPMENT_FILTER_PARAMS.get(
            self.equipment_type,
            EQUIPMENT_FILTER_PARAMS["egi"],
        )

        # Use custom parameters if provided
        l_freq = self.l_freq or params["l_freq"]
        h_freq = self.h_freq or params["h_freq"]
        notch_freq = self.notch_freq or params["notch_freq"]
        resample_freq = self.resample_freq or params["resample_freq"]

        # Apply filtering
        raw = raw.copy()

        # Resample if necessary
        if raw.info["sfreq"] != resample_freq:
            raw.resample(resample_freq)

        # Apply bandpass filter
        raw.filter(l_freq, h_freq)

        # Apply notch filter
        raw.notch_filter(notch_freq)

        input["data"] = raw
        return input, None


@register_preprocessor
class ICMAdaptiveArtifactRejection(BasePreprocessor):
    """ICM Adaptive artifact rejection.

    Parameters
    ----------
    z_threshold : float, optional
        Z-score threshold for artifact detection. Default: 3.0
    max_bad_channels : float, optional
        Maximum proportion of bad channels to reject. Default: 0.1
    max_bad_epochs : float, optional
        Maximum proportion of bad epochs to reject. Default: 0.2
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        max_bad_channels: float = 0.1,
        max_bad_epochs: float = 0.2,
        on: Optional[List[str]] = None,
    ):
        self.z_threshold = z_threshold
        self.max_bad_channels = max_bad_channels
        self.max_bad_epochs = max_bad_epochs

        super().__init__(on=on)

    def get_valid_inputs(self) -> List[str]:
        """Get valid input types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: Dict[str, Any],
        extra_input: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
        """Apply adaptive artifact rejection."""
        data = input["data"]

        if isinstance(data, mne.io.BaseRaw):
            # For continuous data, create epochs for artifact detection
            events = mne.find_events(data)
            if len(events) == 0:
                output = input.copy()
                output["data"] = data
                return output, None

            # Create temporary epochs
            epochs = mne.Epochs(
                data,
                events,
                tmin=-0.1,
                tmax=0.8,
                baseline=None,
                preload=True,
                verbose=False,
            )

            # Apply artifact rejection
            epochs_clean = self._apply_adaptive_rejection(epochs)

            # Return cleaned epochs with preserved meta
            output = input.copy()
            output["data"] = epochs_clean
            return output, None

        if isinstance(data, mne.BaseEpochs):
            # Apply artifact rejection to epochs
            epochs_clean = self._apply_adaptive_rejection(data)
            output = input.copy()
            output["data"] = epochs_clean
            return output, None

        raise ValueError(
            "Input data must be mne.io.BaseRaw or mne.BaseEpochs",
        )

    def _apply_adaptive_rejection(
        self,
        epochs: mne.BaseEpochs,
    ) -> mne.BaseEpochs:
        """Apply adaptive artifact rejection algorithm."""
        epochs = epochs.copy()

        # Get data matrix
        data = epochs.get_data()
        n_epochs, n_channels, n_times = data.shape

        # Calculate channel-wise statistics
        channel_stats = np.zeros(n_channels)
        for ch in range(n_channels):
            ch_data = data[:, ch, :].flatten()
            channel_stats[ch] = np.std(ch_data)

        # Find bad channels
        channel_z_scores = np.abs(
            (channel_stats - np.mean(channel_stats)) / np.std(channel_stats),
        )
        bad_channels = np.where(channel_z_scores > self.z_threshold)[0]

        # Limit number of bad channels
        max_bad_ch = int(self.max_bad_channels * n_channels)
        if len(bad_channels) > max_bad_ch:
            bad_channels = bad_channels[
                np.argsort(channel_z_scores[bad_channels])[-max_bad_ch:]
            ]

        # Calculate epoch-wise statistics
        epoch_stats = np.zeros(n_epochs)
        for ep in range(n_epochs):
            ep_data = data[ep, :, :].flatten()
            epoch_stats[ep] = np.std(ep_data)

        # Find bad epochs
        epoch_z_scores = np.abs(
            (epoch_stats - np.mean(epoch_stats)) / np.std(epoch_stats),
        )
        bad_epochs = np.where(epoch_z_scores > self.z_threshold)[0]

        # Limit number of bad epochs
        max_bad_ep = int(self.max_bad_epochs * n_epochs)
        if max_bad_ep == 0:
            bad_epochs = []  # do not drop any epochs
        elif len(bad_epochs) > max_bad_ep:
            bad_epochs = bad_epochs[
                np.argsort(epoch_z_scores[bad_epochs])[-max_bad_ep:]
            ]

        # Update bad channels and epochs
        bad_ch_names = [epochs.ch_names[ch] for ch in bad_channels]
        if bad_ch_names:
            epochs.info["bads"] = list(set(epochs.info["bads"] + bad_ch_names))

        if len(bad_epochs) > 0:
            epochs.drop(bad_epochs)

        # Apply average referencing (matching NICE behavior)
        epochs.set_eeg_reference("average", projection=True)

        # Interpolate bad channels (only if digitization info is available)
        if len(epochs.info["bads"]) > 0:
            try:
                epochs.interpolate_bads(reset_bads=True)
            except RuntimeError as e:
                if "Cannot fit headshape without digitization" in str(e):
                    # Skip interpolation for synthetic data without electrode positions
                    print(
                        "Warning: Skipping channel interpolation - no digitization info available"
                    )
                    print(
                        f"Bad channels marked but not interpolated: {epochs.info['bads']}"
                    )
                else:
                    raise e

        return epochs


@register_preprocessor
class ICMLGEpoching(BasePreprocessor):
    """ICM Local-Global epoching preprocessor.

    Parameters
    ----------
    tmin : float, optional
        Start time before event. Default: -0.2
    tmax : float, optional
        End time after event. Default: 1.34
    baseline : tuple, optional
        Baseline period. Default: (-0.2, 0.0)
    reject_criteria : dict, optional
        Rejection criteria for epochs. Default: {'eeg': 100e-6}
    """

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 1.34,
        baseline: Tuple[float, float] = (-0.2, 0.0),
        reject_criteria: Optional[Dict[str, float]] = None,
        event_id: Optional[Dict[str, int]] = None,
        on: Optional[List[str]] = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
        self.baseline = baseline
        # Handle reject_criteria: None means no rejection, empty dict means no rejection, otherwise use provided criteria
        if reject_criteria is None:
            self.reject_criteria = None  # No rejection
        elif isinstance(reject_criteria, dict) and len(reject_criteria) == 0:
            self.reject_criteria = None  # Empty dict means no rejection
        else:
            self.reject_criteria = reject_criteria or {"eeg": 100e-6}
        # Allow user-provided event_id mapping (e.g. to match custom trigger codes)
        self.event_id = event_id
        super().__init__(on=on)

    def get_valid_inputs(self) -> List[str]:
        """Get valid input types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: Dict[str, Any],
        extra_input: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any] | None]:
        """Create epochs from continuous data."""
        raw = input["data"]

        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("Input data must be mne.io.BaseRaw")

        # Keep EEG channels and stimulus channels, drop everything else
        # Get EEG channels using MNE's channel type detection
        eeg_picks = mne.pick_types(raw.info, eeg=True)
        eeg_channels = [raw.ch_names[i] for i in eeg_picks]
        # Get all stimulus channels
        stim_picks = mne.pick_types(raw.info, stim=True)
        stim_channels = [raw.ch_names[i] for i in stim_picks]

        # Define good channels: all EEG + all stimulus channels
        good_channels = eeg_channels + stim_channels

        # Find channels to drop (everything not in good_channels)
        to_drop = [x for x in raw.ch_names if x not in good_channels]

        if len(to_drop) > 0:
            print(
                f"Dropping {len(to_drop)} non-EEG channels: {to_drop[:5]}{'...' if len(to_drop) > 5 else ''}"
            )
            raw = raw.copy()  # Make a copy to avoid modifying original
            raw.drop_channels(to_drop)

        # Find events - try stim channels first, then annotations
        try:
            events = mne.find_events(raw)
        except ValueError as e:
            if "No stim channels found" in str(e) and raw.annotations:
                # Convert annotations to events
                events, event_id_from_annot = mne.events_from_annotations(raw)
                print(f"Extracted {len(events)} events from annotations")
            else:
                raise e

        if len(events) == 0:
            raise ValueError("No events found in the data")

        # Use provided event_id mapping or default ICM LG codes
        event_id = self.event_id or {
            "HSTD": 10,
            "HDVT": 20,
            "LSGS": 30,
            "LSGD": 40,
            "LDGS": 60,
            "LDGD": 50,
        }

        # Filter events to ICM LG events only
        icm_events = []
        # Keep only the events matching mapping values (or all if mapping empty)
        if event_id == "auto":
            # Build deterministic mapping based on sorted unique codes
            codes = sorted(np.unique(events[:, 2]))
            if len(codes) >= 6:
                event_id = dict(
                    zip(
                        ["HSTD", "HDVT", "LSGS", "LSGD", "LDGD", "LDGS"],
                        codes[:6],
                    )
                )
            else:
                event_id = {}  # accept all events if not enough codes

        if isinstance(event_id, dict) and event_id:
            for event in events:
                if event[2] in event_id.values():
                    icm_events.append(event)
            icm_events = np.array(icm_events)
            if len(icm_events) == 0:
                raise ValueError(
                    "No matching ICM LG events found in the data. Check event_id mapping or raw triggers."
                )
        else:
            # event_id empty dict (or None) means accept all events
            icm_events = events

        # Create epochs
        epochs = mne.Epochs(
            raw,
            icm_events,
            event_id=event_id,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            reject=self.reject_criteria,
            preload=True,
            verbose=False,
            on_missing="ignore",
        )

        # Drop stimulus channels after epoching (matching NICE behavior)
        stim_channels_to_drop = [
            ch for ch in epochs.ch_names if ch.startswith("STI")
        ]
        if stim_channels_to_drop:
            epochs.drop_channels(stim_channels_to_drop)

        output = input.copy()
        output["data"] = epochs
        return output, None
