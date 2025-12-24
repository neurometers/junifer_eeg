"""Simplified HDF5 reader for junifer_eeg markers.

Now that markers have fixed tensor structures, dimension inference is trivial:
just map marker class → dimension order.

This is a thin wrapper around junifer's HDF5FeatureStorage that adds
dimension name mapping based on marker class.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from junifer.storage import HDF5FeatureStorage

# Simple mapping: marker class → dimension order (axis 0, 1, 2, ...)
MARKER_DIMENSIONS = {
    # 3D markers: (bands/taus, epochs, channels)
    "SpectralPowerBands": ["bands", "epochs", "channels"],
    "PermutationEntropy": ["taus", "epochs", "channels"],
    # 2D markers: (epochs, channels)
    "KolmogorovComplexity": ["epochs", "channels"],
    "TimeLockedTopography": ["epochs", "channels"],
    "TimeLockedContrast": ["epochs", "channels"],
    "ContingentNegativeVariation": ["epochs", "channels"],
    "PowerSpectralDensitySummary": ["epochs", "channels"],
    "SlowWavesDetection": ["epochs", "channels"],
    "SpindlesDetection": ["epochs", "channels"],
    # 4D markers: (taus, epochs, channels_i, channels_j)
    "SymbolicMutualInformation": [
        "taus",
        "epochs",
        "channels_i",
        "channels_j",
    ],
    "SymbolicMutualInformationROIs": [
        "taus",
        "epochs",
        "channels_i",
        "channels_j",
    ],
    # 2D markers with special dims: (channels, freqs)
    "PowerSpectralDensityEstimator": ["channels", "frequencies"],
    # Scalar markers
    "WindowDecoding": [],  # Scalar
    "TimeDecoding": ["times"],  # 1D time series
}


class MarkerData:
    """Container for marker data with simple dimension info."""

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        dim_names: List[str],
        col_names: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.name = name
        self.data = data
        self.dim_names = dim_names
        self.col_names = col_names or []
        self.metadata = metadata or {}

    @property
    def shape(self) -> tuple:
        """Data shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    def dump_to_pkl(self, filepath: Union[str, Path]) -> None:
        """Save marker data to pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to save the pickle file.
        """
        import pickle

        filepath = Path(filepath)
        data_dict = {
            "name": self.name,
            "data": self.data,
            "dim_names": self.dim_names,
            "col_names": self.col_names,
            "metadata": self.metadata,
            "shape": self.shape,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data_dict, f)

    def __repr__(self) -> str:
        dims_str = " x ".join(
            f"{name}({size})" for name, size in zip(self.dim_names, self.shape)
        )
        return f"MarkerData('{self.name}': {dims_str})"


class SimplifiedH5Reader:
    """Simplified HDF5 reader for junifer_eeg markers.

    Now that markers have fixed dimensions, we just need:
    1. Read marker class from HDF5 metadata
    2. Map class → dimension names
    3. Done!

    This is a thin wrapper around junifer's HDF5FeatureStorage.

    Examples
    --------
    >>> reader = SimplifiedH5Reader("output.h5")
    >>> data = reader.get("psd_alpha")
    >>> print(data)
    MarkerData('psd_alpha': bands(5) x epochs(100) x channels(64))
    >>> print(data.dim_names)
    ['bands', 'epochs', 'channels']
    """

    def __init__(self, filepath: Union[str, Path]):
        """Initialize reader.

        Parameters
        ----------
        filepath : str or Path
            Path to HDF5 file.
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.filepath}")

        # Use junifer's HDF5FeatureStorage
        self._storage = HDF5FeatureStorage(uri=self.filepath)
        self._features = self._storage.list_features()

    def list_markers(self) -> List[str]:
        """List all available markers in the file.

        Returns
        -------
        list of str
            Marker names.
        """
        return sorted([meta["name"] for meta in self._features.values()])

    def get(self, marker_name: str) -> MarkerData:
        """Load marker data.

        Parameters
        ----------
        marker_name : str
            Marker name (e.g., 'psd_alpha', 'pe_theta').

        Returns
        -------
        MarkerData
            Loaded marker data with dimension info.
        """
        # Read data using junifer's storage
        hdf_data = self._storage.read(feature_name=marker_name)

        # Extract data array
        data = hdf_data["data"]

        # Handle timeseries storage (list of arrays) - convert to single array
        if isinstance(data, list):
            # Timeseries storage: list of arrays, one per element
            # For single element, just extract the array
            if len(data) == 1:
                data = data[0]
            else:
                # Multiple elements - concatenate
                data = np.concatenate(data, axis=0)

        # Get column headers
        col_names = hdf_data.get("column_headers", [])

        # Get marker class from metadata
        marker_class = self._get_marker_class(marker_name)

        # Get dimension names from MARKER_DIMENSIONS dict
        dim_names = MARKER_DIMENSIONS.get(
            marker_class,
            [
                f"dim{i}" for i in range(data.ndim)
            ],  # Fallback if unknown marker
        )

        # Get full metadata
        metadata = {
            "marker_class": marker_class,
            "filepath": str(self.filepath),
            "kind": hdf_data.get("kind"),
        }

        return MarkerData(
            name=marker_name,
            data=data,
            dim_names=dim_names,
            col_names=col_names,
            metadata=metadata,
        )

    def _get_marker_class(self, marker_name: str) -> Optional[str]:
        """Get marker class from metadata.

        Parameters
        ----------
        marker_name : str
            Marker name.

        Returns
        -------
        str or None
            Marker class name (e.g., 'SpectralPowerBands').
        """
        for meta in self._features.values():
            if meta.get("name") == marker_name:
                marker_meta = meta.get("marker", {})
                if isinstance(marker_meta, dict):
                    return marker_meta.get("class")
                # Fallback for simple string storage
                return marker_meta if isinstance(marker_meta, str) else None
        return None

    def dump_all_to_pkl(self, filepath: Union[str, Path]) -> None:
        """Dump all markers to a single pickle file.

        The pickle structure is a dictionary where:
        - Keys are marker names
        - Values are dictionaries containing:
          - "name": marker name
          - "data": numpy array
          - "dim_names": list of dimension names
          - "col_names": list of column names
          - "metadata": metadata dict
          - "shape": tuple of shape

        Parameters
        ----------
        filepath : str or Path
            Path to save the pickle file.

        Examples
        --------
        >>> reader = read_h5("output.h5")
        >>> reader.dump_all_to_pkl("all_markers.pkl")
        >>> # Load back
        >>> import pickle
        >>> with open("all_markers.pkl", "rb") as f:
        ...     all_data = pickle.load(f)
        >>> print(all_data.keys())  # marker names
        >>> print(all_data["cnv"]["shape"])  # individual marker info
        """
        import pickle

        filepath = Path(filepath)

        # Load all markers
        all_markers = {}
        for marker_name in self.list_markers():
            marker_data = self.get(marker_name)

            # Store in same structure as individual dump_to_pkl
            all_markers[marker_name] = {
                "name": marker_data.name,
                "data": marker_data.data,
                "dim_names": marker_data.dim_names,
                "col_names": marker_data.col_names,
                "metadata": marker_data.metadata,
                "shape": marker_data.shape,
            }

        # Save to pickle
        with open(filepath, "wb") as f:
            pickle.dump(all_markers, f)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, *args):
        """Context manager exit."""
        pass

    def __repr__(self) -> str:
        markers = self.list_markers()
        return (
            f"SimplifiedH5Reader({self.filepath.name}, {len(markers)} markers)"
        )


def read_h5(filepath: Union[str, Path]) -> SimplifiedH5Reader:
    """Open an HDF5 file for reading.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file.

    Returns
    -------
    SimplifiedH5Reader
        Reader object.

    Examples
    --------
    >>> reader = read_h5("output.h5")
    >>> print(reader.list_markers())
    ['psd_alpha', 'psd_beta', 'pe_theta', ...]
    >>> data = reader.get("psd_alpha")
    >>> print(data.shape)
    (5, 100, 64)  # bands x epochs x channels
    """
    return SimplifiedH5Reader(filepath)
