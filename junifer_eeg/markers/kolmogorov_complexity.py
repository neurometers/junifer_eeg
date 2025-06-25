"""Kolmogorov Complexity marker for junifer_eeg."""

from typing import Any, ClassVar

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker


@register_marker
class KolmogorovComplexity(BaseMarker):
    """Kolmogorov Complexity marker.

    This marker computes the Kolmogorov complexity of EEG signals using
    compression-based approximation. The signal is first symbolized into
    discrete bins, then compressed using zlib, and the complexity is
    measured as the compression ratio.

    Adapted from the NICE package implementation.
    """

    _DEPENDENCIES: ClassVar = {"mne", "numpy"}
    _MARKER_INOUT_MAPPINGS: ClassVar = {
        "EEG": {"kolmogorov_complexity": "vector"}
    }

    def __init__(
        self,
        tmin: float | None = None,
        tmax: float | None = None,
        nbins: int = 32,
        on: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the KolmogorovComplexity marker.

        Parameters
        ----------
        tmin : float, optional
            Start time for analysis in seconds.
        tmax : float, optional
            End time for analysis in seconds.
        nbins : int, default=32
            Number of bins for signal symbolization.
        on : str, optional
            Data type to compute on.
        name : str, optional
            Name of the marker.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.nbins = nbins
        super().__init__(on=on, name=name)

    def compute(
        self, input: dict[str, Any], extra_input: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Compute Kolmogorov complexity.

        Parameters
        ----------
        input : dict
            Input data containing 'data' with MNE Raw object.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        dict
            Computed Kolmogorov complexity features.
        """
        import zlib

        from mne.utils import _time_mask

        # Get the MNE Raw object
        raw = input["data"]

        # Get data and apply time mask if specified
        data = raw.get_data()  # Shape: (n_channels, n_times)

        if self.tmin is not None or self.tmax is not None:
            time_mask = _time_mask(raw.times, self.tmin, self.tmax)
            data = data[:, time_mask]

        n_channels, n_samples = data.shape
        k_values = np.zeros(n_channels, dtype=np.float64)

        for channel in range(n_channels):
            signal = data[channel, :]

            # Symbolic transformation - match original algorithm exactly
            ssignal = np.sort(signal)
            items = signal.shape[0]
            first = int(items / 10)
            last = items - first if first > 1 else items - 1
            lower = ssignal[first]
            upper = ssignal[last]
            bsize = (upper - lower) / self.nbins

            # Create symbolic representation
            osignal = np.zeros(signal.shape, dtype=np.uint8)
            maxbin = self.nbins - 1

            for i in range(items):
                tbin = int((signal[i] - lower) / bsize) if bsize > 0 else 0
                osignal[i] = (
                    0 if tbin < 0 else maxbin if tbin > maxbin else tbin
                ) + ord("A")

            # Use zlib for compression
            string = osignal.tobytes()
            cstring = zlib.compress(string)
            k_values[channel] = float(len(cstring)) / float(len(string))

        # Return data in junifer format
        return {
            "kolmogorov_complexity": {
                "data": k_values.reshape(1, -1),  # Shape: (1, n_channels)
                "col_names": [f"{ch}_kolmogorov" for ch in raw.ch_names],
            }
        }
