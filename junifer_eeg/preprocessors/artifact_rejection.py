"""Modular artifact rejection preprocessors."""

from typing import Any, Optional

import mne
from junifer.api.decorators import register_preprocessor
from junifer.preprocess.base import BasePreprocessor
from mne.utils import logger

from .artifact_utils import (
    check_min_channels,
    check_min_events,
    find_bads_channels_high_frequency,
    find_bads_channels_threshold,
    find_bads_channels_variance,
    find_bads_epochs_threshold,
)


@register_preprocessor
class BadChannelsThreshold(BasePreprocessor):
    """Detect bad channels based on threshold rejection.

    Finds bad channels based on the number of epochs where the channel
    values range (max - min) is over the reject threshold.

    Parameters
    ----------
    reject : dict, optional
        Rejection threshold for each channel type. Default: {'eeg': 100e-6}
    n_epochs_bad_ch : float, optional
        Fraction of epochs over threshold to mark channel as bad. Default: 0.5
    min_channels : float, optional
        Minimum fraction of good channels required. Default: 0.7
    interpolate : bool, optional
        Whether to interpolate bad channels. Default: True
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        reject: Optional[dict[str, float]] = None,
        n_epochs_bad_ch: float = 0.5,
        min_channels: float = 0.7,
        interpolate: bool = True,
        on: Optional[list[str]] = None,
    ):
        self.reject = reject or {"eeg": 100e-6}
        self.n_epochs_bad_ch = n_epochs_bad_ch
        self.min_channels = min_channels
        self.interpolate = interpolate
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
        """Detect and mark bad channels based on threshold."""
        epochs = input["data"]

        if not isinstance(epochs, mne.BaseEpochs):
            raise ValueError("Input data must be mne.BaseEpochs")

        # Get picks excluding already marked bad channels
        picks = mne.pick_types(
            epochs.info, meg=False, eeg=True, exclude="bads"
        )

        # Find bad channels
        bad_channels = find_bads_channels_threshold(
            epochs, picks, self.reject, self.n_epochs_bad_ch
        )

        # Check minimum channels
        all_bad_channels = list(set(epochs.info["bads"] + bad_channels))
        check_min_channels(epochs, all_bad_channels, self.min_channels)

        # Mark bad channels
        epochs.info["bads"].extend(bad_channels)

        # Interpolate if requested
        if self.interpolate and len(epochs.info["bads"]) > 0:
            logger.info(
                f"Interpolating {len(epochs.info['bads'])} bad channels"
            )
            epochs.interpolate_bads(reset_bads=True)

        output = input.copy()
        output["data"] = epochs
        return output, None


@register_preprocessor
class BadChannelsVariance(BasePreprocessor):
    """Detect bad channels based on variance z-scoring.

    Finds bad channels based on iterative z-score outlier detection
    over the channel variance.

    Parameters
    ----------
    zscore_thresh : float, optional
        Z-score threshold for outlier detection. Default: 4
    max_iter : int, optional
        Maximum iterations for z-scoring. Default: 2
    min_channels : float, optional
        Minimum fraction of good channels required. Default: 0.7
    interpolate : bool, optional
        Whether to interpolate bad channels. Default: True
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        zscore_thresh: float = 4,
        max_iter: int = 2,
        min_channels: float = 0.7,
        interpolate: bool = True,
        on: Optional[list[str]] = None,
    ):
        self.zscore_thresh = zscore_thresh
        self.max_iter = max_iter
        self.min_channels = min_channels
        self.interpolate = interpolate
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
        """Detect and mark bad channels based on variance."""
        epochs = input["data"]

        if not isinstance(epochs, mne.BaseEpochs):
            raise ValueError("Input data must be mne.BaseEpochs")

        # Get picks excluding already marked bad channels
        picks = mne.pick_types(
            epochs.info, meg=False, eeg=True, exclude="bads"
        )

        # Find bad channels
        bad_channels = find_bads_channels_variance(
            epochs, picks, self.zscore_thresh, self.max_iter
        )

        # Check minimum channels
        all_bad_channels = list(set(epochs.info["bads"] + bad_channels))
        check_min_channels(epochs, all_bad_channels, self.min_channels)

        # Mark bad channels
        epochs.info["bads"].extend(bad_channels)

        # Interpolate if requested
        if self.interpolate and len(epochs.info["bads"]) > 0:
            logger.info(
                f"Interpolating {len(epochs.info['bads'])} bad channels"
            )
            epochs.interpolate_bads(reset_bads=True)

        output = input.copy()
        output["data"] = epochs
        return output, None


@register_preprocessor
class BadEpochsThreshold(BasePreprocessor):
    """Detect bad epochs based on threshold rejection.

    Finds bad epochs based on the number of channels where the values
    range is over the reject threshold.

    Parameters
    ----------
    reject : dict, optional
        Rejection threshold for each channel type. Default: {'eeg': 100e-6}
    n_channels_bad_epoch : float, optional
        Fraction of channels over threshold to mark epoch as bad. Default: 0.1
    min_events : float, optional
        Minimum fraction of good events required. Default: 0.3
    drop_bad_epochs : bool, optional
        Whether to drop bad epochs. Default: True
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        reject: Optional[dict[str, float]] = None,
        n_channels_bad_epoch: float = 0.1,
        min_events: float = 0.3,
        drop_bad_epochs: bool = True,
        on: Optional[list[str]] = None,
    ):
        self.reject = reject or {"eeg": 100e-6}
        self.n_channels_bad_epoch = n_channels_bad_epoch
        self.min_events = min_events
        self.drop_bad_epochs = drop_bad_epochs
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
        """Detect and optionally drop bad epochs based on threshold."""
        epochs = input["data"]

        if not isinstance(epochs, mne.BaseEpochs):
            raise ValueError("Input data must be mne.BaseEpochs")

        # Get picks excluding already marked bad channels
        picks = mne.pick_types(
            epochs.info, meg=False, eeg=True, exclude="bads"
        )

        # Store original epoch count before dropping
        original_epoch_count = len(epochs)

        # Find bad epochs
        bad_epochs = find_bads_epochs_threshold(
            epochs, picks, self.reject, self.n_channels_bad_epoch
        )

        # Store bad epochs info before dropping (for tests)
        bad_epochs_indices = (
            bad_epochs.tolist()
            if hasattr(bad_epochs, "tolist")
            else list(bad_epochs)
        )

        # Drop bad epochs if requested
        if self.drop_bad_epochs and len(bad_epochs) > 0:
            epochs.drop(bad_epochs, reason="artifacted")
            logger.info(f"Dropped {len(bad_epochs)} bad epochs")

        # Check minimum events
        check_min_events(epochs, self.min_events)

        output = input.copy()
        output["data"] = epochs

        # Store bad epochs info in epochs.info["temp"] for backward compatibility with tests
        if "temp" not in epochs.info:
            epochs.info["temp"] = {}
        if "preprocessing_info" not in epochs.info["temp"]:
            epochs.info["temp"]["preprocessing_info"] = {}
        epochs.info["temp"]["preprocessing_info"]["bad_epochs_detected"] = (
            bad_epochs_indices
        )
        epochs.info["temp"]["preprocessing_info"][
            "n_epochs_before_rejection"
        ] = original_epoch_count
        epochs.info["temp"]["preprocessing_info"][
            "n_epochs_after_rejection"
        ] = len(epochs)

        return output, None


@register_preprocessor
class BadChannelsHighFrequency(BasePreprocessor):
    """Detect bad channels based on high frequency variance z-scoring.

    Finds bad channels based on iterative z-score outlier detection
    over the channel high frequency standard deviation.

    Parameters
    ----------
    zscore_thresh : float, optional
        Z-score threshold for outlier detection. Default: 4
    max_iter : int, optional
        Maximum iterations for z-scoring. Default: 2
    min_channels : float, optional
        Minimum fraction of good channels required. Default: 0.7
    interpolate : bool, optional
        Whether to interpolate bad channels. Default: True
    dump_path : str, optional
        Full path (including filename) for dumping. If None, no dumping.
        If it doesn't end with .fif, the extension will be added automatically.
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    def __init__(
        self,
        zscore_thresh: float = 4,
        max_iter: int = 2,
        min_channels: float = 0.7,
        interpolate: bool = True,
        dump_path: Optional[str] = None,
        on: Optional[list[str]] = None,
    ):
        self.zscore_thresh = zscore_thresh
        self.max_iter = max_iter
        self.min_channels = min_channels
        self.interpolate = interpolate
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
        """Detect and mark bad channels based on high frequency variance."""
        epochs = input["data"]

        if not isinstance(epochs, mne.BaseEpochs):
            raise ValueError("Input data must be mne.BaseEpochs")

        # Get picks excluding already marked bad channels
        picks = mne.pick_types(
            epochs.info, meg=False, eeg=True, exclude="bads"
        )

        # Store original epoch count from input (before any preprocessing)
        # Try to get from input meta, otherwise use current count
        original_epoch_count = input.get("meta", {}).get(
            "n_epochs_original", len(epochs)
        )
        if original_epoch_count is None or original_epoch_count == 0:
            original_epoch_count = len(epochs)

        # Find bad channels
        bad_channels = find_bads_channels_high_frequency(
            epochs, picks, self.zscore_thresh, self.max_iter
        )

        # Check minimum channels
        all_bad_channels = list(set(epochs.info["bads"] + bad_channels))
        check_min_channels(epochs, all_bad_channels, self.min_channels)

        # Store bad channels BEFORE interpolation (since reset_bads=True clears them)
        all_bad_channels_before_interp = all_bad_channels.copy()

        # Mark bad channels
        epochs.info["bads"].extend(bad_channels)

        # Interpolate if requested
        if self.interpolate and len(epochs.info["bads"]) > 0:
            logger.info(
                f"Interpolating {len(epochs.info['bads'])} bad channels"
            )
            epochs.interpolate_bads(reset_bads=True)

        output = input.copy()
        output["data"] = epochs

        # Always store preprocessing info (for tests and debugging)
        # Collect all bad channels detected across all steps
        preprocessing_info = {
            "bad_channels_detected": all_bad_channels_before_interp,
            "n_epochs_before_rejection": original_epoch_count,
            "n_epochs_after_rejection": len(epochs),
            "n_channels_interpolated": len(all_bad_channels_before_interp),
        }

        # Store in output meta
        output["meta"] = output.get("meta", {})
        output["meta"]["preprocessing_info"] = preprocessing_info

        # Also store in epochs.info["temp"] for backward compatibility with tests
        # Merge with existing preprocessing_info if present (e.g., from BadEpochsThreshold)
        if "temp" not in epochs.info:
            epochs.info["temp"] = {}
        if "preprocessing_info" not in epochs.info["temp"]:
            epochs.info["temp"]["preprocessing_info"] = {}
        # Merge the new info with existing info (preserve bad_epochs_detected from BadEpochsThreshold)
        epochs.info["temp"]["preprocessing_info"].update(preprocessing_info)

        # Dump data if requested
        if self.dump_path:
            from .dump_utils import dump_preprocessing_data

            # Replace {element} placeholder in dump_path
            element = input.get("meta", {}).get("element", "unknown_element")
            if isinstance(element, dict):
                element = "unknown_element"
            dump_path_resolved = (
                self.dump_path.format(element=element)
                if "{element}" in self.dump_path
                else self.dump_path
            )

            # Dump using the utility function - all metadata will be stored in data.yaml
            dump_preprocessing_data(output, dump_path_resolved)

        return output, None
