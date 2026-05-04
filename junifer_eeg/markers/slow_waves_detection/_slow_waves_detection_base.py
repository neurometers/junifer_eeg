"""Base slow waves detection computation with caching (singleton pattern).

Structure (2-file, matching kolmogorov_complexity_new):
- _slow_waves_detection_base.py: Singleton with caching + compute logic (this file)
- slow_waves_detection.py: Main marker extending EEGEpochsMarker
"""

from typing import TYPE_CHECKING, ClassVar, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from junifer.utils import logger

from ...utils.singleton import Singleton

if TYPE_CHECKING:
    import mne


__all__ = ["SlowWavesDetectionBase"]


# Valid features that can be computed
SLOW_WAVE_FEATURES = ("Duration", "PTP", "Frequency", "Slope", "Density")


class SlowWavesDetectionBase(metaclass=Singleton):
    """Slow waves detection computation with caching.

    Singleton class that detects slow waves and computes ALL features with
    internal caching for efficient reuse across multiple markers.

    The caching uses (epochs_id, params) as key - valid while the
    epochs object exists in memory. This follows the same pattern
    as KolmogorovComplexityBase.

    Features computed:
    - Duration: Mean slow wave duration per epoch/channel
    - PTP: Mean peak-to-peak amplitude per epoch/channel
    - Frequency: Mean slow wave frequency per epoch/channel
    - Slope: Mean slow wave slope per epoch/channel
    - Density: Count of slow waves per epoch/channel
    """

    _DEPENDENCIES: ClassVar = {"numpy", "yasa", "pandas", "mne", "scipy"}

    # Internal cache: {(epochs_id, params_tuple): {feature_name: array}}
    _cache: ClassVar[dict] = {}

    def __init__(self) -> None:
        """Initialize slow waves detection base."""
        pass

    @staticmethod
    def _normalize_subject_id(subject_id: str | int | float | None) -> str | None:
        """Normalize subject IDs for CSV matching.

        Makes matching tolerant to common variants like ``03`` vs ``3`` or
        ``sub-03`` vs ``03``.
        """
        if subject_id is None:
            return None
        value = str(subject_id).strip()
        if value.lower().startswith("sub-"):
            value = value[4:]
        if value.isdigit():
            value = str(int(value))
        return value

    def __del__(self) -> None:  # pragma: no cover
        """Terminate and clear cache."""
        logger.debug("Clearing cache for slow waves detection computation")
        SlowWavesDetectionBase._cache.clear()

    def compute(
        self,
        epochs: "mne.Epochs",
        detection_method: Literal["yasa", "custom"],
        freq_sw: Tuple[float, float],
        amp_ptp_initial: float,
        freq_threshold: float,
        artifact_threshold: float,
        ptp_threshold_mode: Literal["adaptive", "fixed"],
        ptp_percentile: float,
        max_ptp_amplitude: float,
        ptp_thresholds_path: str | None,
        subject_id: str | None,
        reference_channels: Tuple[str, ...],
        proximity_amplitude: float | None = None,
        proximity_window: float = 1.0,
        ptp_thresholds_strict: bool = True,
        filter_design: Literal["chebyshev2", "fir"] = "chebyshev2",
        slope_uv_per_s_range: Tuple[float, float] | None = None,
        downsample_to: float | None = None,
    ) -> Dict[str, np.ndarray]:
        """Compute slow wave detection with caching.

        Returns ALL features in a dictionary. The caller selects which
        feature to return based on user request.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object (already filtered to desired channels).
        detection_method : {"yasa", "custom"}
            Detection backend to use.
        freq_sw : tuple of float
            Slow wave frequency range in Hz (fmin, fmax).
        amp_ptp_initial : float
            Initial peak-to-peak amplitude threshold in µV.
        freq_threshold : float
            Maximum frequency threshold for filtering (Hz).
        artifact_threshold : float
            Positive peak amplitude threshold for artifact removal (µV).
        reference_channels : tuple of str
            Reference channel names for re-referencing.

        Returns
        -------
        features : dict of np.ndarray
            Dictionary with keys 'Duration', 'PTP', 'Frequency', 'Slope', 'Density'.
            Each value is an array of shape (n_epochs, n_channels).
        """
        # Build cache key from hashable parameters
        cache_key = (
            id(epochs),
            detection_method,
            freq_sw,
            amp_ptp_initial,
            freq_threshold,
            artifact_threshold,
            ptp_threshold_mode,
            ptp_percentile,
            max_ptp_amplitude,
            ptp_thresholds_path,
            subject_id,
            reference_channels,
            proximity_amplitude,
            proximity_window,
            ptp_thresholds_strict,
            filter_design,
            slope_uv_per_s_range,
            downsample_to,
        )

        # Check cache first
        if cache_key in self._cache:
            logger.debug(f"SlowWaves cache hit: epochs_id={id(epochs)}")
            return self._cache[cache_key]

        logger.debug(
            f"SlowWaves cache miss: computing freq_sw={freq_sw}, "
            f"amp_ptp={amp_ptp_initial}, method={detection_method}"
        )

        # Perform actual computation
        features = self._detect_slow_waves(
            epochs,
            detection_method,
            freq_sw,
            amp_ptp_initial,
            freq_threshold,
            artifact_threshold,
            ptp_threshold_mode,
            ptp_percentile,
            max_ptp_amplitude,
            ptp_thresholds_path,
            subject_id,
            reference_channels,
            proximity_amplitude,
            proximity_window,
            ptp_thresholds_strict,
            filter_design,
            slope_uv_per_s_range,
            downsample_to,
        )

        # Store in cache
        self._cache[cache_key] = features

        return features

    def _detect_slow_waves(
        self,
        epochs: "mne.Epochs",
        detection_method: Literal["yasa", "custom"],
        freq_sw: Tuple[float, float],
        amp_ptp_initial: float,
        freq_threshold: float,
        artifact_threshold: float,
        ptp_threshold_mode: Literal["adaptive", "fixed"],
        ptp_percentile: float,
        max_ptp_amplitude: float,
        ptp_thresholds_path: str | None,
        subject_id: str | None,
        reference_channels: Tuple[str, ...],
        proximity_amplitude: float | None = None,
        proximity_window: float = 1.0,
        ptp_thresholds_strict: bool = True,
        filter_design: Literal["chebyshev2", "fir"] = "chebyshev2",
        slope_uv_per_s_range: Tuple[float, float] | None = None,
        downsample_to: float | None = None,
    ) -> Dict[str, np.ndarray]:
        """Perform slow wave detection on epochs.

        Parameters
        ----------
        epochs : mne.Epochs
            MNE Epochs object.
        freq_sw : tuple of float
            Slow wave frequency range in Hz.
        amp_ptp_initial : float
            Initial peak-to-peak amplitude threshold in µV.
        freq_threshold : float
            Maximum frequency threshold for filtering.
        artifact_threshold : float
            Positive peak amplitude threshold for artifact removal.
        reference_channels : tuple of str
            Reference channel names.

        Returns
        -------
        features : dict of np.ndarray
            Dictionary with all computed features.
        """
        # Make a copy to avoid modifying original data
        epochs_copy = epochs.copy()

        # Re-reference to specified reference channels
        try:
            # Check if reference channels exist
            missing_channels = [
                ch
                for ch in reference_channels
                if ch not in epochs_copy.ch_names
            ]
            if missing_channels:
                logger.warning(
                    f"Reference channels {missing_channels} not found. "
                    "Using original data."
                )
            else:
                # Check if reference channels have valid data
                valid_ref = True
                for ch in reference_channels:
                    ch_data = epochs_copy.get_data(picks=ch)
                    if np.allclose(ch_data, 0):
                        logger.warning(
                            f"Reference channel {ch} contains all zeros. "
                            "Using original data."
                        )
                        valid_ref = False
                        break

                if valid_ref:
                    # Apply custom referencing
                    ref_data = epochs_copy.get_data(
                        picks=list(reference_channels)
                    ).mean(axis=1)
                    ref_data = ref_data[:, np.newaxis, :]
                    epochs_copy._data -= ref_data

        except Exception as e:
            logger.warning(f"Re-referencing failed ({e}), using original data")

        # Optional downsampling (Le Coz 2025) — opt-in, default disabled
        if downsample_to is not None and downsample_to > 0:
            current_sf = float(epochs_copy.info["sfreq"])
            if downsample_to < current_sf:
                epochs_copy.resample(downsample_to, npad="auto", verbose=False)

        # Get sampling frequency and channel info
        sf = epochs_copy.info["sfreq"]
        ch_names = epochs_copy.ch_names
        chan2idx = {ch: i for i, ch in enumerate(ch_names)}

        if detection_method == "yasa":
            all_events, filtered_per_epoch = self._detect_with_yasa(
                epochs_copy=epochs_copy,
                sf=sf,
                ch_names=ch_names,
                chan2idx=chan2idx,
                freq_sw=freq_sw,
                amp_ptp_initial=amp_ptp_initial,
                filter_design=filter_design,
                need_filtered=proximity_amplitude is not None,
            )
        else:
            all_events, filtered_per_epoch = self._detect_with_custom_method(
                epochs_copy=epochs_copy,
                sf=sf,
                ch_names=ch_names,
                chan2idx=chan2idx,
                freq_sw=freq_sw,
                amp_ptp_initial=amp_ptp_initial,
                filter_design=filter_design,
            )

        # Combine all events from all epochs
        events_df = (
            pd.concat(all_events, ignore_index=True)
            if all_events
            else pd.DataFrame()
        )

        # Apply post-processing filtering (Andrillon & Pinggal criteria)
        if not events_df.empty:
            events_df = self._apply_dynamic_threshold(
                events_df,
                freq_threshold=freq_threshold,
                artifact_threshold=artifact_threshold,
                ptp_threshold_mode=ptp_threshold_mode,
                ptp_percentile=ptp_percentile,
                max_ptp_amplitude=max_ptp_amplitude,
                ptp_thresholds_path=ptp_thresholds_path,
                subject_id=subject_id,
                proximity_amplitude=proximity_amplitude,
                proximity_window=proximity_window,
                ptp_thresholds_strict=ptp_thresholds_strict,
                slope_uv_per_s_range=slope_uv_per_s_range,
                filtered_per_epoch=filtered_per_epoch,
                sf=sf,
            )

        # Get dimensions
        n_epochs = len(epochs_copy)
        n_channels = len(ch_names)

        # Initialize feature arrays with NaN
        features = {
            "Duration": np.full((n_epochs, n_channels), np.nan),
            "PTP": np.full((n_epochs, n_channels), np.nan),
            "Frequency": np.full((n_epochs, n_channels), np.nan),
            "Slope": np.full((n_epochs, n_channels), np.nan),
            "Density": np.full((n_epochs, n_channels), np.nan),
        }

        # Fill feature arrays
        if not events_df.empty:
            grouped = events_df.groupby(["Epoch", "ChanIdx"])

            for (epoch_idx, chan_idx), group in grouped:
                if epoch_idx < n_epochs and chan_idx < n_channels:
                    features["Duration"][epoch_idx, chan_idx] = group[
                        "Duration"
                    ].mean()
                    features["PTP"][epoch_idx, chan_idx] = group["PTP"].mean()
                    features["Frequency"][epoch_idx, chan_idx] = group[
                        "Frequency"
                    ].mean()
                    features["Slope"][epoch_idx, chan_idx] = group[
                        "Slope"
                    ].mean()
                    features["Density"][epoch_idx, chan_idx] = len(group)

        return features

    def _detect_with_yasa(
        self,
        epochs_copy: "mne.Epochs",
        sf: float,
        ch_names: list[str],
        chan2idx: Dict[str, int],
        freq_sw: Tuple[float, float],
        amp_ptp_initial: float,
        filter_design: Literal["chebyshev2", "fir"] = "chebyshev2",
        need_filtered: bool = False,
    ) -> Tuple[list[pd.DataFrame], Dict[int, np.ndarray]]:
        """Run the YASA backend epoch by epoch.

        Returns
        -------
        all_events : list of pd.DataFrame
            One DataFrame per epoch with detected slow waves.
        filtered_per_epoch : dict[int, np.ndarray]
            Mapping epoch_idx → filtered signal in µV with shape
            (n_channels, n_samples). Empty when ``need_filtered`` is False.
        """
        import yasa

        all_events: list[pd.DataFrame] = []
        filtered_per_epoch: Dict[int, np.ndarray] = {}
        for epoch_idx in range(len(epochs_copy)):
            epoch_data = epochs_copy[epoch_idx].get_data()[0]
            epoch_data_uV = epoch_data * 1e6
            if need_filtered:
                filtered_per_epoch[epoch_idx] = self._bandpass_uv(
                    epoch_data_uV, sf, freq_sw, filter_design
                )
            res = yasa.sw_detect(
                data=epoch_data_uV,
                sf=sf,
                ch_names=ch_names,
                freq_sw=freq_sw,
                amp_ptp=(amp_ptp_initial, np.inf),
                coupling=False,
                remove_outliers=False,
                verbose=False,
            )
            df = res.summary() if res is not None else None
            if df is not None and len(df):
                df = df.copy()
                df["Epoch"] = epoch_idx
                df["ChanIdx"] = df["Channel"].map(chan2idx)
                # Le Coz 2025 separable slopes — derived from YASA columns
                # so the schema stays additive.
                if {"Start", "NegPeak", "MidCrossing", "ValNegPeak"}.issubset(
                    df.columns
                ):
                    desc_dt = (df["NegPeak"] - df["Start"]).replace(
                        0.0, np.nan
                    )
                    asc_dt = (df["MidCrossing"] - df["NegPeak"]).replace(
                        0.0, np.nan
                    )
                    df["DescendingSlope"] = (
                        df["ValNegPeak"].abs() / desc_dt
                    )
                    df["AscendingSlope"] = df["ValNegPeak"].abs() / asc_dt
                all_events.append(df)
        return all_events, filtered_per_epoch

    def _detect_with_custom_method(
        self,
        epochs_copy: "mne.Epochs",
        sf: float,
        ch_names: list[str],
        chan2idx: Dict[str, int],
        freq_sw: Tuple[float, float],
        amp_ptp_initial: float,
        filter_design: Literal["chebyshev2", "fir"] = "chebyshev2",
    ) -> Tuple[list[pd.DataFrame], Dict[int, np.ndarray]]:
        """Run a zero-crossing detector inspired by Andrillon et al. 2021.

        Returns
        -------
        all_events : list of pd.DataFrame
        filtered_per_epoch : dict[int, np.ndarray]
            Filtered signal in µV per epoch (n_channels, n_samples). Used
            both for detection and for the optional proximity-to-artifact
            rule downstream.
        """
        all_events: list[pd.DataFrame] = []
        filtered_per_epoch: Dict[int, np.ndarray] = {}

        for epoch_idx in range(len(epochs_copy)):
            epoch_data = epochs_copy[epoch_idx].get_data()[0]
            epoch_data_uV = epoch_data * 1e6
            filtered = self._bandpass_uv(
                epoch_data_uV, sf, freq_sw, filter_design
            )
            filtered_per_epoch[epoch_idx] = filtered
            events = []

            for ch_idx, ch_name in enumerate(ch_names):
                ch_events = self._detect_channel_zero_crossing(
                    signal_uV=filtered[ch_idx],
                    sf=sf,
                    amp_ptp_initial=amp_ptp_initial,
                    epoch_idx=epoch_idx,
                    ch_name=ch_name,
                    chan_idx=ch_idx,
                )
                if ch_events:
                    events.extend(ch_events)

            if events:
                all_events.append(pd.DataFrame(events))

        return all_events, filtered_per_epoch

    @staticmethod
    def _bandpass_uv(
        data_uV: np.ndarray,
        sf: float,
        freq_sw: Tuple[float, float],
        filter_design: Literal["chebyshev2", "fir"],
    ) -> np.ndarray:
        """Bandpass-filter µV data in the slow-wave band.

        ``"chebyshev2"`` (default) reproduces the previous behaviour
        (scipy ``iirdesign`` Chebyshev II IIR + ``sosfiltfilt``).
        ``"fir"`` uses ``mne.filter.filter_data`` zero-phase FIR
        (Le Coz 2025).
        """
        wp = np.asarray(freq_sw, dtype=float)
        if wp.shape != (2,):
            raise ValueError(
                f"freq_sw must be a length-2 tuple, got {freq_sw!r}"
            )

        if filter_design == "fir":
            from mne.filter import filter_data

            return filter_data(
                data_uV.astype(np.float64, copy=False),
                sfreq=sf,
                l_freq=float(wp[0]),
                h_freq=float(wp[1]),
                method="fir",
                phase="zero",
                verbose=False,
            )

        # Default: Chebyshev II IIR (preserves pre-existing behaviour)
        from scipy.signal import iirdesign, sosfiltfilt

        ws = np.asarray(
            [
                max(0.01, wp[0] * 0.1),
                min((sf / 2.0) - 0.5, max(wp[1] + 5.0, wp[1] * 1.5)),
            ],
            dtype=float,
        )
        if not (0 < ws[0] < wp[0] < wp[1] < ws[1] < (sf / 2.0)):
            raise ValueError(
                "Invalid custom slow-wave filter bounds derived from "
                f"freq_sw={freq_sw} and sf={sf}"
            )
        sos = iirdesign(
            wp=wp,
            ws=ws,
            gpass=3,
            gstop=25,
            ftype="cheby2",
            output="sos",
            fs=sf,
        )
        return sosfiltfilt(sos, data_uV, axis=-1)

    def _detect_channel_zero_crossing(
        self,
        signal_uV: np.ndarray,
        sf: float,
        amp_ptp_initial: float,
        epoch_idx: int,
        ch_name: str,
        chan_idx: int,
    ) -> list[dict]:
        """Detect zero-crossing slow waves on one channel."""
        zero_crossings = np.where(np.diff(np.signbit(signal_uV)))[0]
        if len(zero_crossings) < 3:
            return []

        events = []
        for idx in range(len(zero_crossings) - 2):
            start = zero_crossings[idx] + 1
            mid = zero_crossings[idx + 1] + 1
            end = zero_crossings[idx + 2] + 1
            if not (start < mid < end):
                continue

            neg_segment = signal_uV[start:mid]
            pos_segment = signal_uV[mid:end]
            if neg_segment.size == 0 or pos_segment.size == 0:
                continue

            neg_rel = int(np.argmin(neg_segment))
            neg_val = float(neg_segment[neg_rel])
            if neg_val >= 0:
                continue

            pos_rel = int(np.argmax(pos_segment))
            pos_val = float(pos_segment[pos_rel])
            if pos_val <= 0:
                continue

            neg_idx = start + neg_rel
            pos_idx = mid + pos_rel
            duration = (end - start) / sf
            if duration <= 0:
                continue

            ptp = pos_val - neg_val
            if ptp < amp_ptp_initial:
                continue

            neg_to_pos = max((pos_idx - neg_idx) / sf, np.finfo(float).eps)
            frequency = 1.0 / duration
            slope = ptp / neg_to_pos

            # Le Coz 2025 separable slopes (always emitted; additive columns).
            desc_dt = max((neg_idx - start) / sf, np.finfo(float).eps)
            asc_dt = max((mid - neg_idx) / sf, np.finfo(float).eps)
            descending_slope = abs(neg_val) / desc_dt
            ascending_slope = abs(neg_val) / asc_dt

            events.append(
                {
                    "Channel": ch_name,
                    "Epoch": epoch_idx,
                    "ChanIdx": chan_idx,
                    "Start": start / sf,
                    "End": end / sf,
                    "MidCrossing": mid / sf,
                    "Duration": duration,
                    "ValNegPeak": neg_val,
                    "ValPosPeak": pos_val,
                    "PTP": ptp,
                    "Frequency": frequency,
                    "Slope": slope,
                    "DescendingSlope": descending_slope,
                    "AscendingSlope": ascending_slope,
                }
            )

        return events

    def _apply_dynamic_threshold(
        self,
        sw_df: pd.DataFrame,
        freq_threshold: float,
        artifact_threshold: float,
        ptp_threshold_mode: Literal["adaptive", "fixed"],
        ptp_percentile: float,
        max_ptp_amplitude: float,
        ptp_thresholds_path: str | None,
        subject_id: str | None,
        proximity_amplitude: float | None = None,
        proximity_window: float = 1.0,
        ptp_thresholds_strict: bool = True,
        slope_uv_per_s_range: Tuple[float, float] | None = None,
        filtered_per_epoch: Dict[int, np.ndarray] | None = None,
        sf: float | None = None,
    ) -> pd.DataFrame:
        """Apply dynamic thresholding and filtering to slow waves.

        This implements a combined filtering stage for both backends:
        1. Remove waves with frequency > freq_threshold Hz
        2. Remove waves with PTP >= max_ptp_amplitude µV
        3. Remove artifacts with positive peak >= artifact_threshold µV
        4. If available, remove waves whose positive half-period proxy falls
           outside the expected range used by the legacy custom method
        5. If ``ptp_threshold_mode="adaptive"``, keep only waves above a
           per-channel PTP threshold, either loaded from CSV or computed from
           the current element data.

        Parameters
        ----------
        sw_df : pd.DataFrame
            DataFrame with all detected slow waves.
        freq_threshold : float
            Maximum frequency threshold.
        artifact_threshold : float
            Positive peak amplitude threshold for artifact removal.

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame.
        """
        # Rule 1: Remove high frequency waves
        sw_df = sw_df[sw_df["Frequency"] <= freq_threshold].copy()

        if sw_df.empty:
            return sw_df

        # Rule 2: Remove abnormally large PTP events
        if "PTP" in sw_df.columns:
            sw_df = sw_df[sw_df["PTP"] < max_ptp_amplitude].copy()

        if sw_df.empty:
            return sw_df

        # Rule 3: Positive-peak artifact rejection when the backend exposes it
        if "ValPosPeak" in sw_df.columns:
            sw_df = sw_df[sw_df["ValPosPeak"] < artifact_threshold].copy()

        if sw_df.empty:
            return sw_df

        # Rule 4: Legacy custom-method half-period constraint, if available.
        slope_col = None
        if "pos_halfway_period" in sw_df.columns:
            slope_col = "pos_halfway_period"
        elif "PosHalfPeriod" in sw_df.columns:
            slope_col = "PosHalfPeriod"

        if slope_col is not None:
            sw_df = sw_df[
                (sw_df[slope_col] >= 0.143) & (sw_df[slope_col] <= 2.0)
            ].copy()

        if sw_df.empty:
            return sw_df

        # Rule 4b (opt-in, Le Coz 2025): drop waves whose ascending OR
        # descending slope falls outside ``slope_uv_per_s_range``.
        if slope_uv_per_s_range is not None and {
            "AscendingSlope",
            "DescendingSlope",
        }.issubset(sw_df.columns):
            lo, hi = float(slope_uv_per_s_range[0]), float(
                slope_uv_per_s_range[1]
            )
            asc = sw_df["AscendingSlope"]
            desc = sw_df["DescendingSlope"]
            keep = (
                asc.between(lo, hi, inclusive="both")
                & desc.between(lo, hi, inclusive="both")
            )
            sw_df = sw_df[keep].copy()

        if sw_df.empty:
            return sw_df

        # Rule 4c (opt-in, Pinggal 2022 / Andrillon 2021): drop waves whose
        # center falls within ``proximity_window`` seconds of any sample where
        # |filtered_signal| > proximity_amplitude on the same channel/epoch.
        if (
            proximity_amplitude is not None
            and filtered_per_epoch
            and sf
            and sf > 0
        ):
            sw_df = self._apply_proximity_filter(
                sw_df,
                filtered_per_epoch=filtered_per_epoch,
                sf=float(sf),
                proximity_amplitude=float(proximity_amplitude),
                proximity_window=float(proximity_window),
            )

        if sw_df.empty or ptp_threshold_mode == "fixed":
            return sw_df

        if ptp_thresholds_path is not None:
            if subject_id is None:
                raise ValueError(
                    "ptp_thresholds_path was provided but no subject metadata "
                    "was found in extra_input."
                )
            thr_df = pd.read_csv(ptp_thresholds_path)
            normalized_subject = self._normalize_subject_id(subject_id)
            thr_df = thr_df[
                thr_df["subject"]
                .map(self._normalize_subject_id)
                .eq(normalized_subject)
            ]
            if thr_df.empty:
                if ptp_thresholds_strict:
                    raise ValueError(
                        f"No PTP thresholds found for subject={subject_id} in "
                        f"{ptp_thresholds_path}"
                    )
                logger.warning(
                    f"No PTP thresholds found for subject={subject_id} in "
                    f"{ptp_thresholds_path}; emitting empty slow-wave results "
                    "for this element (ptp_thresholds_strict=False)."
                )
                return sw_df.iloc[0:0]
            channel_thresholds = dict(
                zip(thr_df["channel"], thr_df["ptp_threshold"])
            )
        else:
            channel_thresholds = {}
            for channel in sw_df["Channel"].unique():
                channel_sw = sw_df[sw_df["Channel"] == channel]
                channel_thresholds[channel] = np.percentile(
                    channel_sw["PTP"], ptp_percentile
                )

        keep_mask = sw_df.apply(
            lambda row: row["PTP"]
            > channel_thresholds.get(row["Channel"], np.inf),
            axis=1,
        )
        sw_df = sw_df[keep_mask].copy()

        return sw_df

    @staticmethod
    def _apply_proximity_filter(
        sw_df: pd.DataFrame,
        filtered_per_epoch: Dict[int, np.ndarray],
        sf: float,
        proximity_amplitude: float,
        proximity_window: float,
    ) -> pd.DataFrame:
        """Drop waves close to high-amplitude artifacts (Pinggal 2022).

        For each (Epoch, ChanIdx), build an artifact mask where the absolute
        value of the post-filter / post-reference signal in µV exceeds
        ``proximity_amplitude``, dilate it by ``proximity_window`` seconds
        on each side, and reject any wave whose center falls inside.
        """
        if sw_df.empty:
            return sw_df

        # Center used for proximity check: prefer MidCrossing when available,
        # otherwise fall back to the midpoint of (Start, End).
        if "MidCrossing" in sw_df.columns:
            centers = sw_df["MidCrossing"].to_numpy()
        else:
            centers = (
                sw_df["Start"].to_numpy() + sw_df["End"].to_numpy()
            ) / 2.0

        win_samples = max(1, int(round(proximity_window * sf)))
        keep = np.ones(len(sw_df), dtype=bool)
        epochs = sw_df["Epoch"].to_numpy()
        chan_idxs = sw_df["ChanIdx"].to_numpy()

        # Cache per-(epoch, channel) the dilated artifact mask
        cache: Dict[tuple[int, int], np.ndarray] = {}
        for i in range(len(sw_df)):
            ep = int(epochs[i])
            ch = int(chan_idxs[i])
            sig = filtered_per_epoch.get(ep)
            if sig is None or ch >= sig.shape[0]:
                continue
            mask_key = (ep, ch)
            mask = cache.get(mask_key)
            if mask is None:
                base = np.abs(sig[ch]) > proximity_amplitude
                if not base.any():
                    cache[mask_key] = base
                    continue
                # Dilate ±win_samples via uniform_filter1d (boolean → bool)
                from scipy.ndimage import maximum_filter1d

                mask = maximum_filter1d(
                    base.astype(np.uint8),
                    size=2 * win_samples + 1,
                    mode="nearest",
                ).astype(bool)
                cache[mask_key] = mask
            n_samples = mask.shape[0]
            sample = int(round(centers[i] * sf))
            if 0 <= sample < n_samples and mask[sample]:
                keep[i] = False

        return sw_df[keep].copy()
