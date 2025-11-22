"""General EEG epoching preprocessor."""

from typing import Any, Optional

import mne
import numpy as np
from junifer.api.decorators import register_preprocessor
from junifer.preprocess.base import BasePreprocessor
from mne.utils import logger


@register_preprocessor
class EEGEpoching(BasePreprocessor):
    """General EEG epoching preprocessor.

    Creates epochs from continuous data by detecting events and excluding
    non-EEG channels. Uses equipment information stored in the data to
    identify which channels are EEG vs stimulus/other channels.

    Parameters
    ----------
    tmin : float, optional
        Start time before event. Default: -0.2
    tmax : float, optional
        End time after event. Default: 1.0
    baseline : tuple of float, optional
        Baseline correction period. Default: (None, 0)
    event_id : dict, optional
        Event ID mapping. If None, uses all events found.
    exclude_channels : list of str, optional
        Additional channels to exclude (beyond non-EEG channels).
    dump_path : str, optional
        Full path (including filename) for dumping. If None, no dumping.
        If it doesn't end with .fif, the extension will be added automatically.
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        tmin: float = -0.2,
        tmax: float = 1.0,
        baseline: tuple[Optional[float], float] = (None, 0),
        event_id: Optional[dict[str, int]] = None,
        exclude_channels: Optional[list[str]] = None,
        dump_path: Optional[str] = None,
        on: Optional[list[str]] = None,
    ):
        self.tmin = tmin
        self.tmax = tmax
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
        """Create epochs from continuous data."""
        raw = input["data"]

        if not isinstance(raw, mne.io.BaseRaw):
            raise ValueError("Input data must be mne.io.BaseRaw")

        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Find events
        try:
            events = mne.find_events(raw, shortest_event=1)
            found_id = np.unique(events[:, 2])
            if self.event_id is None:
                # Use all found events
                this_id = {str(v): v for v in found_id}
            else:
                # Filter to only include requested events
                this_id = {
                    k: v for k, v in self.event_id.items() if v in found_id
                }
        except ValueError:
            # No STI channel found - try reading from annotations
            logger.info(
                "No STI channel found, reading events from annotations"
            )
            events, event_id_from_annot = mne.events_from_annotations(raw)
            if self.event_id is None:
                # Use all found events
                this_id = event_id_from_annot
            else:
                # Filter to only include requested events
                this_id = {
                    k: v
                    for k, v in self.event_id.items()
                    if k in event_id_from_annot
                }

            if not this_id:
                raise ValueError(
                    f"No matching events found. Available: {list(event_id_from_annot.keys())}"
                ) from None

        # Create epochs
        epochs = mne.Epochs(
            raw,
            events,
            this_id,
            tmin=self.tmin,
            tmax=self.tmax,
            preload=True,
            reject=None,
            picks=None,
            baseline=self.baseline,
            verbose=False,
        )

        # Exclude non-EEG channels
        # Pick only EEG channels
        eeg_picks = mne.pick_types(
            epochs.info, eeg=True, meg=False, exclude=[]
        )
        eeg_channels = [epochs.ch_names[i] for i in eeg_picks]

        # Also exclude any user-specified channels
        channels_to_keep = [
            ch for ch in eeg_channels if ch not in self.exclude_channels
        ]

        if len(channels_to_keep) < len(epochs.ch_names):
            epochs.pick_channels(channels_to_keep)
            logger.info(
                f"Filtered to {len(channels_to_keep)} EEG channels (excluded {len(epochs.ch_names) - len(channels_to_keep)} non-EEG/stimulus channels)"
            )

        output = input.copy()
        output["data"] = epochs

        # Store original epoch count in meta for later use
        if "meta" not in output:
            output["meta"] = {}
        output["meta"]["n_epochs_original"] = len(epochs)

        # Dump data if requested
        if self.dump_path:
            from .dump_utils import dump_preprocessing_data

            dump_preprocessing_data(output, self.dump_path)

        return output, None
