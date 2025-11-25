"""Unified EEG epoching preprocessor."""

from typing import Any, Optional

import mne
import numpy as np
from junifer.api.decorators import register_preprocessor
from junifer.preprocess.base import BasePreprocessor
from mne.utils import logger


@register_preprocessor
class EEGEpoching(BasePreprocessor):
    """Unified EEG epoching preprocessor.

    Creates epochs from continuous data either:
    1. Event-based: Uses events/triggers from the recording
    2. Fixed-length: Creates epochs of fixed duration (when event_id=None and duration is set)

    Parameters
    ----------
    tmin : float, optional
        Start time before event (event-based) or epoch start (fixed-length). Default: -0.2
    tmax : float, optional
        End time after event (event-based) or epoch end (fixed-length). Default: 1.0
    duration : float, optional
        Duration of fixed-length epochs. If set, creates fixed-length epochs instead of event-based.
        Overrides tmax. Default: None
    overlap : float, optional
        Overlap between fixed-length epochs in seconds. Only used with duration. Default: 0.0
    baseline : tuple of float, optional
        Baseline correction period. Default: (None, 0)
    event_id : dict, optional
        Event ID mapping for event-based epoching. If None with duration set, creates fixed-length epochs.
    exclude_channels : list of str, optional
        Additional channels to exclude (beyond non-EEG channels).
    dump_path : str, optional
        Full path (including filename) for dumping. If None, no dumping.
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 1.0,
        duration: Optional[float] = None,
        overlap: float = 0.0,
        baseline: tuple[Optional[float], float] = (None, 0),
        event_id: Optional[dict[str, int]] = None,
        exclude_channels: Optional[list[str]] = None,
        dump_path: Optional[str] = None,
        on: Optional[list[str]] = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
        self.duration = duration
        self.overlap = overlap
        self.baseline = baseline
        self.event_id = event_id
        self.exclude_channels = exclude_channels or []
        self.dump_path = dump_path
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
        """Create epochs from continuous data (event-based or fixed-length)."""
        raw = input["data"]

        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("Input data must be mne.io.BaseRaw")

        raw = raw.copy()

        # Choose epoching method
        if self.duration is not None:
            # Fixed-length epoching
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=self.duration,
                overlap=self.overlap,
                preload=True,
                verbose=False,
            )
            if self.baseline is not None:
                epochs.apply_baseline(self.baseline)
        else:
            # Event-based epoching
            try:
                events = mne.find_events(raw, shortest_event=1)
                found_id = np.unique(events[:, 2])
                this_id = (
                    {str(v): v for v in found_id}
                    if self.event_id is None
                    else {
                        k: v for k, v in self.event_id.items() if v in found_id
                    }
                )
            except ValueError:
                logger.info("Reading events from annotations")
                events, event_id_from_annot = mne.events_from_annotations(raw)
                this_id = (
                    event_id_from_annot
                    if self.event_id is None
                    else {
                        k: v
                        for k, v in self.event_id.items()
                        if k in event_id_from_annot
                    }
                )
                if not this_id:
                    raise ValueError(
                        f"No matching events. Available: {list(event_id_from_annot.keys())}"
                    ) from None

            epochs = mne.Epochs(
                raw,
                events,
                this_id,
                tmin=self.tmin,
                tmax=self.tmax,
                preload=True,
                baseline=self.baseline,
                verbose=False,
            )

        # Filter to EEG channels only
        eeg_picks = mne.pick_types(epochs.info, eeg=True, meg=False)
        eeg_channels = [epochs.ch_names[i] for i in eeg_picks]
        channels_to_keep = [
            ch for ch in eeg_channels if ch not in self.exclude_channels
        ]

        if len(channels_to_keep) < len(epochs.ch_names):
            epochs.pick_channels(channels_to_keep)

        output = input.copy()
        output["data"] = epochs
        if "meta" not in output:
            output["meta"] = {}
        output["meta"]["n_epochs_original"] = len(epochs)

        if self.dump_path:
            from .dump_utils import dump_preprocessing_data

            dump_preprocessing_data(output, self.dump_path)

        return output, None
