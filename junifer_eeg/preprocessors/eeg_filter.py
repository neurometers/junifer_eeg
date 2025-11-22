"""General EEG filter preprocessor."""

from typing import Any, ClassVar, Optional

import mne
from junifer.api.decorators import register_preprocessor
from junifer.preprocess import BasePreprocessor
from mne.utils import logger


@register_preprocessor
class EEGFilter(BasePreprocessor):
    """General EEG filter preprocessor.

    Applies high-pass, low-pass, and notch filters, and optionally resamples.
    All parameters are user-configurable.

    Parameters
    ----------
    low_freq : float, optional
        Low-pass filter frequency (Hz). If None, no low-pass filtering.
    high_freq : float, optional
        High-pass filter frequency (Hz). If None, no high-pass filtering.
    notches : list of float, optional
        Notch filter frequencies (Hz). If None, no notch filtering.
    resample_freq : float, optional
        Target sampling frequency for resampling. If None, no resampling.
    hp_order : int, optional
        High-pass filter order. Default: 4
    lp_order : int, optional
        Low-pass filter order. Default: 8
    l_trans_bandwidth : float, optional
        Low-frequency transition bandwidth. Default: 0.1
    filter_method : str, optional
        Filter method ('iir' or 'fir'). Default: 'iir'
    n_jobs : int, optional
        Number of parallel jobs. Default: 1
    dump_path : str, optional
        Full path (including filename) for dumping. If None, no dumping.
        If it doesn't end with .fif, the extension will be added automatically.
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    _DEPENDENCIES: ClassVar = {"mne"}

    def __init__(
        self,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        notches: Optional[list[float]] = None,
        resample_freq: Optional[float] = None,
        hp_order: int = 4,
        lp_order: int = 8,
        l_trans_bandwidth: float = 0.1,
        filter_method: str = "iir",
        n_jobs: int = 1,
        dump_path: Optional[str] = None,
        on: Optional[str] = None,
    ) -> None:
        """Initialize EEGFilter."""
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notches = notches
        self.resample_freq = resample_freq
        self.hp_order = hp_order
        self.lp_order = lp_order
        self.l_trans_bandwidth = l_trans_bandwidth
        self.filter_method = filter_method
        self.n_jobs = n_jobs
        self.dump_path = dump_path
        super().__init__(on=on)

    def get_valid_inputs(self) -> list[str]:
        """Get valid data types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Filter EEG data."""
        raw_or_epochs = input["data"]

        if not isinstance(raw_or_epochs, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise ValueError(
                "Input data must be mne.io.BaseRaw or mne.BaseEpochs"
            )

        # Make a copy to avoid modifying original
        raw_or_epochs = raw_or_epochs.copy()

        # Pick EEG channels
        picks = mne.pick_types(
            raw_or_epochs.info, eeg=True, meg=True, ecg=True, exclude=[]
        )

        # Apply high-pass filter
        if self.high_freq is not None:
            logger.info(
                f"Applying high-pass filter at {self.high_freq} Hz (order {self.hp_order})"
            )
            raw_or_epochs.filter(
                l_freq=self.high_freq,
                h_freq=None,
                picks=picks,
                method=self.filter_method,
                iir_params={"ftype": "butter", "order": self.hp_order}
                if self.filter_method == "iir"
                else None,
                l_trans_bandwidth=self.l_trans_bandwidth,
                n_jobs=self.n_jobs,
            )

        # Apply low-pass filter
        if self.low_freq is not None:
            logger.info(
                f"Applying low-pass filter at {self.low_freq} Hz (order {self.lp_order})"
            )
            raw_or_epochs.filter(
                l_freq=None,
                h_freq=self.low_freq,
                picks=picks,
                method=self.filter_method,
                iir_params={"ftype": "butter", "order": self.lp_order}
                if self.filter_method == "iir"
                else None,
                n_jobs=self.n_jobs,
            )

        # Apply notch filters
        if self.notches:
            # Filter out notch frequencies that are above current Nyquist frequency
            current_nyquist = raw_or_epochs.info["sfreq"] / 2.0
            valid_notches = [f for f in self.notches if f < current_nyquist]

            if valid_notches:
                logger.info(f"Applying notch filters at {valid_notches} Hz")
                raw_or_epochs.notch_filter(
                    valid_notches, method="fft", n_jobs=self.n_jobs
                )

        # Resample if requested
        if (
            self.resample_freq is not None
            and raw_or_epochs.info["sfreq"] != self.resample_freq
        ):
            logger.info(f"Resampling to {self.resample_freq} Hz")
            raw_or_epochs.resample(self.resample_freq, npad="auto")

        output = input.copy()
        output["data"] = raw_or_epochs

        # Dump data if requested
        if self.dump_path:
            from .dump_utils import dump_preprocessing_data

            dump_preprocessing_data(output, self.dump_path)

        return output, None
