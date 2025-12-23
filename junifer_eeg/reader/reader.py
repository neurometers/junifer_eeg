"""
JuniferH5Reader - Main reader class with dimension-aware loading.

This module provides the primary interface for reading junifer HDF5 files
with full support for dimension metadata and schema inference.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .aggregators import aggregate
from .dimension_schema import (
    AggregationInfo,
    DimensionInfo,
    DimensionType,
    TensorSchema,
)
from .marker_registry import (
    MarkerRegistry,
    get_registry,
)


class MarkerData:
    """Container for loaded marker data with schema information.

    Attributes
    ----------
    name : str
        Marker name.
    data : np.ndarray
        The marker data tensor.
    schema : TensorSchema
        Schema describing tensor dimensions.
    raw_metadata : dict
        Original HDF5 metadata.
    """

    def __init__(
        self,
        name: str,
        data: np.ndarray,
        schema: TensorSchema,
        raw_metadata: Optional[Dict] = None,
    ):
        self.name = name
        self.data = data
        self.schema = schema
        self.raw_metadata = raw_metadata or {}

    @property
    def shape(self) -> tuple:
        """Data shape."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.data.ndim

    @property
    def marker_type(self) -> str:
        """Marker type string."""
        return self.schema.marker_type

    @property
    def dim_types(self) -> List[DimensionType]:
        """List of dimension types."""
        return self.schema.dim_types

    def has_dimension(self, dim_type: DimensionType) -> bool:
        """Check if data has a specific dimension type."""
        return self.schema.has_dimension(dim_type)

    def get_axis(self, dim_type: DimensionType) -> Optional[int]:
        """Get axis index for a dimension type."""
        return self.schema.get_axis(dim_type)

    def get_labels(self, dim_type: DimensionType) -> Optional[List[str]]:
        """Get labels for a dimension type."""
        dim = self.schema.get_dim(dim_type)
        return dim.labels if dim else None

    def aggregate(
        self, method: str, dim_type: DimensionType, inplace: bool = False
    ) -> "MarkerData":
        """Aggregate data along a dimension.

        Parameters
        ----------
        method : str
            Aggregation method (e.g., 'mean', 'std').
        dim_type : DimensionType
            Dimension type to aggregate over.
        inplace : bool
            If True, modify this object. If False, return new object.

        Returns
        -------
        MarkerData
            Aggregated data (self if inplace, new object otherwise).
        """
        axis = self.get_axis(dim_type)
        if axis is None:
            raise ValueError(f"Dimension {dim_type.name} not found in data")

        new_data = aggregate(self.data, method, axis=axis)

        # Update schema - remove the aggregated dimension
        new_dims = [
            d for i, d in enumerate(self.schema.dimensions) if i != axis
        ]

        # Update aggregation info
        new_agg = AggregationInfo(
            channel_method=self.schema.aggregation.channel_method,
            trial_method=self.schema.aggregation.trial_method,
            connectivity_method=self.schema.aggregation.connectivity_method,
            time_method=self.schema.aggregation.time_method,
            band_method=self.schema.aggregation.band_method,
        )

        # Set appropriate aggregation field
        if dim_type == DimensionType.CHANNELS:
            new_agg.channel_method = method
        elif dim_type == DimensionType.EPOCHS:
            new_agg.trial_method = method
        elif dim_type in (
            DimensionType.CHANNELS_I,
            DimensionType.CHANNELS_J,
            DimensionType.CHANNEL_PAIRS,
        ):
            new_agg.connectivity_method = method
        elif dim_type == DimensionType.TIMES:
            new_agg.time_method = method
        elif dim_type == DimensionType.BANDS:
            new_agg.band_method = method

        new_schema = TensorSchema(
            marker_type=self.schema.marker_type,
            marker_name=self.schema.marker_name,
            dimensions=new_dims,
            aggregation=new_agg,
            parameters=self.schema.parameters,
        )

        if inplace:
            self.data = new_data
            self.schema = new_schema
            return self
        else:
            return MarkerData(
                name=self.name,
                data=new_data,
                schema=new_schema,
                raw_metadata=self.raw_metadata,
            )

    def select(
        self,
        dim_type: DimensionType,
        indices: Union[int, List[int], slice, str, List[str]],
    ) -> "MarkerData":
        """Select subset of data along a dimension.

        Parameters
        ----------
        dim_type : DimensionType
            Dimension type to select from.
        indices : int, list, slice, or str
            Indices or labels to select.

        Returns
        -------
        MarkerData
            New MarkerData with selected subset.
        """
        axis = self.get_axis(dim_type)
        if axis is None:
            raise ValueError(f"Dimension {dim_type.name} not found in data")

        dim = self.schema.get_dim(dim_type)

        # Convert string labels to indices
        if isinstance(indices, str):
            if dim.labels is None:
                raise ValueError(f"No labels available for {dim_type.name}")
            indices = dim.labels.index(indices)
        elif (
            isinstance(indices, list)
            and indices
            and isinstance(indices[0], str)
        ):
            if dim.labels is None:
                raise ValueError(f"No labels available for {dim_type.name}")
            indices = [dim.labels.index(label) for label in indices]

        # Apply selection
        slices = [slice(None)] * self.ndim
        slices[axis] = indices
        new_data = self.data[tuple(slices)]

        # Update dimension info
        if isinstance(indices, int):
            # Single index - dimension is removed
            new_dims = [
                d for i, d in enumerate(self.schema.dimensions) if i != axis
            ]
        else:
            # Multiple indices - dimension is reduced
            new_dims = list(self.schema.dimensions)
            if isinstance(indices, slice):
                new_size = len(range(*indices.indices(dim.size)))
                new_labels = dim.labels[indices] if dim.labels else None
            else:
                new_size = len(indices)
                new_labels = (
                    [dim.labels[i] for i in indices] if dim.labels else None
                )

            new_dims[axis] = DimensionInfo(
                dim_type=dim.dim_type,
                size=new_size,
                labels=new_labels,
                metadata=dim.metadata,
            )

        new_schema = TensorSchema(
            marker_type=self.schema.marker_type,
            marker_name=self.schema.marker_name,
            dimensions=new_dims,
            aggregation=self.schema.aggregation,
            parameters=self.schema.parameters,
        )

        return MarkerData(
            name=self.name,
            data=new_data,
            schema=new_schema,
            raw_metadata=self.raw_metadata,
        )

    def to_dataframe(self, include_metadata: bool = False):
        """Convert to pandas DataFrame.

        For 2D data, creates DataFrame with appropriate index/columns.
        For higher dimensional data, flattens appropriately.

        Parameters
        ----------
        include_metadata : bool
            Whether to include metadata columns.

        Returns
        -------
        pd.DataFrame
            Data as DataFrame.
        """
        import pandas as pd

        if self.ndim == 1:
            labels = self.schema.dimensions[0].labels
            return pd.DataFrame({self.name: self.data}, index=labels)

        elif self.ndim == 2:
            row_labels = self.schema.dimensions[0].labels
            col_labels = self.schema.dimensions[1].labels

            index = row_labels if row_labels else None
            columns = col_labels if col_labels else None

            return pd.DataFrame(self.data, index=index, columns=columns)

        else:
            # Higher dimensions - need to flatten or reshape
            # For now, just reshape to 2D
            flat_shape = (self.data.shape[0], -1)
            flat_data = self.data.reshape(flat_shape)
            return pd.DataFrame(flat_data)

    def describe(self) -> str:
        """Get human-readable description."""
        return self.schema.describe()

    def __repr__(self) -> str:
        return f"MarkerData('{self.name}', shape={self.shape}, type={self.marker_type})"


class JuniferH5Reader:
    """Dimension-aware reader for junifer HDF5 output files.

    This reader provides:
    - Automatic dimension inference from stored metadata or marker type
    - Schema-aware data access with dimension semantics
    - Easy aggregation along named dimensions
    - Conversion to pandas DataFrames

    Parameters
    ----------
    h5_path : str or Path
        Path to the HDF5 file.
    registry : MarkerRegistry, optional
        Custom marker registry for type detection.

    Examples
    --------
    >>> reader = JuniferH5Reader("output.h5")
    >>> reader.list_markers()
    ['psd_alpha', 'PE_theta', 'wsmi']

    >>> data = reader.get("psd_alpha")
    >>> data.schema.describe()

    >>> # Aggregate across channels
    >>> agg_data = data.aggregate("mean", DimensionType.CHANNELS)
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        registry: Optional[MarkerRegistry] = None,
    ):
        self.path = Path(h5_path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.path}")

        self.registry = registry or get_registry()
        self._storage = None
        self._features = None
        self._cache: Dict[str, MarkerData] = {}

        # Initialize storage
        self._init_storage()

    def _init_storage(self):
        """Initialize HDF5 storage connection."""
        from junifer.storage import HDF5FeatureStorage

        self._storage = HDF5FeatureStorage(str(self.path))
        self._features = self._storage.list_features()
        self._build_marker_index()

    def _build_marker_index(self):
        """Build index mapping short names to full feature names."""
        self.markers: Dict[str, str] = {}
        self._md5_map: Dict[str, str] = {}

        for md5, meta in self._features.items():
            full_name = meta.get("name", md5)
            short_name = self._extract_short_name(full_name)
            self.markers[short_name] = full_name
            self._md5_map[short_name] = md5

    def _extract_short_name(self, full_name: str) -> str:
        """Extract user-friendly short name from full feature name."""
        name = full_name

        # Remove common prefixes
        for prefix in ["EEG_", "eeg_"]:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break

        # Remove common suffixes
        suffixes = [
            "_spectralpower",
            "_permutationentropy",
            "_symbolicmutualinformation",
            "_kolmogorovcomplexity",
            "_timelockedtopo",
            "_timelockedcontrast",
            "_slowwavesdetection",
            "_spindlesdetection",
            "_contingentvariation",
            "_windowdecoding",
            "_timedecoding",
        ]
        for suffix in suffixes:
            if name.lower().endswith(suffix):
                name = name[: -len(suffix)]
                break

        return name

    def list_markers(self) -> List[str]:
        """List all available markers.

        Returns
        -------
        list of str
            Short names of available markers.
        """
        return list(self.markers.keys())

    def get_marker_info(self, marker_name: str) -> Dict[str, Any]:
        """Get metadata about a marker without loading data.

        Parameters
        ----------
        marker_name : str
            Marker name (short or full).

        Returns
        -------
        dict
            Marker metadata.
        """
        marker_name = self._resolve_marker_name(marker_name)
        md5 = self._md5_map[marker_name]
        return self._features[md5]

    def get(
        self,
        marker_name: str,
        use_cache: bool = True,
    ) -> MarkerData:
        """Load marker data with dimension schema.

        Parameters
        ----------
        marker_name : str
            Marker name (short or full).
        use_cache : bool
            Whether to use cached data if available.

        Returns
        -------
        MarkerData
            Loaded data with schema information.
        """
        marker_name = self._resolve_marker_name(marker_name)

        # Check cache
        if use_cache and marker_name in self._cache:
            return self._cache[marker_name]

        # Load from HDF5
        full_name = self.markers[marker_name]
        raw_data = self._storage.read(feature_name=full_name)

        # Extract data array
        data = self._extract_data_array(raw_data)

        # Get or infer schema
        schema = self._get_or_infer_schema(marker_name, raw_data, data)

        # Create MarkerData object
        marker_data = MarkerData(
            name=marker_name,
            data=data,
            schema=schema,
            raw_metadata=raw_data,
        )

        # Cache it
        if use_cache:
            self._cache[marker_name] = marker_data

        return marker_data

    def _resolve_marker_name(self, marker_name: str) -> str:
        """Resolve marker name to canonical short name."""
        if marker_name in self.markers:
            return marker_name

        # Case-insensitive search
        for short in self.markers:
            if short.lower() == marker_name.lower():
                return short

        # Check if it's a full name
        for short, full in self.markers.items():
            if full == marker_name or full.lower() == marker_name.lower():
                return short

        raise KeyError(
            f"Marker '{marker_name}' not found. "
            f"Available: {self.list_markers()}"
        )

    def _extract_data_array(self, raw_data: Dict) -> np.ndarray:
        """Extract numpy array from raw HDF5 data."""
        data = raw_data.get("data", np.array([]))

        # Handle list format
        if isinstance(data, list) and len(data) > 0:
            if len(data) == 1:
                data = np.asarray(data[0])
            else:
                # Check if all arrays have the same shape
                arrays = [np.asarray(d) for d in data]
                shapes = [arr.shape for arr in arrays]
                if all(s == shapes[0] for s in shapes):
                    # Same shapes - can stack
                    data = np.stack(arrays, axis=0)
                else:
                    # Different shapes - concatenate along first axis
                    # This happens when multiple subjects have different epoch counts
                    try:
                        data = np.concatenate(arrays, axis=0)
                    except ValueError:
                        # If concatenation fails, just use the first array
                        data = arrays[0]
        elif not isinstance(data, np.ndarray):
            data = np.asarray(data)

        return data

    def _get_or_infer_schema(
        self, marker_name: str, raw_data: Dict, data: np.ndarray
    ) -> TensorSchema:
        """Get explicit schema or infer from data using marker class metadata."""
        # Check for explicit schema in raw_data
        if "tensor_schema" in raw_data:
            return TensorSchema.from_dict(raw_data["tensor_schema"])

        # Get marker class and params from HDF5 metadata (AUTHORITATIVE)
        md5 = self._md5_map.get(marker_name)
        marker_class = None
        marker_params = None
        if md5 and md5 in self._features:
            marker_meta = self._features[md5].get("marker", {})
            marker_class = marker_meta.get(
                "class"
            )  # e.g., 'SymbolicMutualInformation'
            marker_params = marker_meta  # Full params dict

        # Infer schema using marker class (no name-based guessing)
        col_names = self._extract_col_names(raw_data)
        row_names = raw_data.get("row_headers")
        kind = raw_data.get("kind")

        return self.registry.infer_schema(
            name=marker_name,
            data=data,
            col_names=col_names,
            row_names=row_names,
            kind=kind,
            marker_class=marker_class,
            marker_params=marker_params,
        )

    def _extract_col_names(self, raw_data: Dict) -> Optional[List[str]]:
        """Extract column names from raw data."""
        col_headers = raw_data.get("column_headers")
        if col_headers is not None:
            return list(col_headers)
        return None

    def get_array(
        self,
        marker_name: str,
        band: Optional[str] = None,
        epoch: Optional[Union[int, List[int], slice]] = None,
        channel: Optional[Union[str, List[str], int]] = None,
    ) -> np.ndarray:
        """Get raw numpy array with optional slicing.

        Convenience method for quick data access without full MarkerData.

        Parameters
        ----------
        marker_name : str
            Marker name.
        band : str, optional
            Band name for multi-band markers.
        epoch : int, list, or slice, optional
            Epoch selection.
        channel : str, int, or list, optional
            Channel selection.

        Returns
        -------
        np.ndarray
            Selected data.
        """
        marker_data = self.get(marker_name)

        # Apply selections
        if band is not None and marker_data.has_dimension(DimensionType.BANDS):
            marker_data = marker_data.select(DimensionType.BANDS, band)

        if epoch is not None and marker_data.has_dimension(
            DimensionType.EPOCHS
        ):
            marker_data = marker_data.select(DimensionType.EPOCHS, epoch)

        if channel is not None and marker_data.has_dimension(
            DimensionType.CHANNELS
        ):
            marker_data = marker_data.select(DimensionType.CHANNELS, channel)

        return marker_data.data

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()

    def summary(self) -> str:
        """Get summary of all markers in the file.

        Returns
        -------
        str
            Human-readable summary.
        """
        lines = [
            f"JuniferH5Reader: {self.path.name}",
            "=" * 60,
        ]

        for name in sorted(self.markers.keys()):
            try:
                marker_data = self.get(name)
                mtype = marker_data.marker_type
                shape = marker_data.shape
                dims = " x ".join(d.name for d in marker_data.dim_types)
                lines.append(f"  {name}: {mtype}, shape={shape}")
                lines.append(f"    dims: {dims}")
            except Exception as e:
                lines.append(f"  {name}: ERROR - {e}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"JuniferH5Reader('{self.path}', markers={len(self.markers)})"

    def __str__(self) -> str:
        return self.summary()

    def __getitem__(self, marker_name: str) -> MarkerData:
        """Allow dictionary-style access: reader['marker_name']."""
        return self.get(marker_name)

    def __contains__(self, marker_name: str) -> bool:
        """Allow 'marker_name' in reader."""
        try:
            self._resolve_marker_name(marker_name)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        """Number of markers."""
        return len(self.markers)

    def __iter__(self):
        """Iterate over marker names."""
        return iter(self.markers.keys())

    def dump_to_pkl(self, output_path: Union[str, Path]) -> Dict[str, Any]:
        """Dump all markers and metadata to a pickle file.

        This method exhaustively exports ALL information from the HDF5 file
        to a pickle file that can be easily loaded by any Python user.

        Parameters
        ----------
        output_path : str or Path
            Path to save the pickle file.

        Returns
        -------
        dict
            The dumped data dictionary (same as what's saved to pickle).

        Notes
        -----
        The output pickle contains a dictionary with:
        - 'source_file': Original HDF5 file path
        - 'markers': Dict mapping marker names to marker info dictionaries
        - 'summary': Human-readable summary string

        Each marker info dictionary contains:
        - 'name': Marker name
        - 'full_name': Full feature name in HDF5
        - 'data': numpy array with the marker data
        - 'shape': Tuple with data shape
        - 'dtype': Data type string
        - 'dimensions': List of dimension descriptions
        - 'dimension_labels': Dict mapping dimension names to their labels
        - 'schema_description': Human-readable schema description
        - 'aggregation': Dict with aggregation info
        - 'parameters': Marker-specific parameters (if available)
        - 'raw_metadata': Original HDF5 metadata
        """
        import pickle

        output_path = Path(output_path)

        # Build the dump dictionary
        dump_data = {
            "source_file": str(self.path.absolute()),
            "source_filename": self.path.name,
            "markers": {},
            "marker_list": [],
        }

        # Process each marker
        for marker_name in self.list_markers():
            try:
                marker_data = self.get(marker_name, use_cache=False)

                # Build dimension descriptions
                dimensions = []
                dimension_labels = {}

                for i, dim in enumerate(marker_data.schema.dimensions):
                    dim_desc = {
                        "axis": i,
                        "type": dim.dim_type.name,
                        "size": dim.size,
                        "description": self._get_dimension_description(
                            dim.dim_type
                        ),
                    }
                    dimensions.append(dim_desc)

                    # Store labels if available
                    if dim.labels is not None:
                        dimension_labels[f"axis_{i}_{dim.dim_type.name}"] = (
                            dim.labels
                        )

                # Build marker info
                marker_info = {
                    "name": marker_name,
                    "full_name": self.markers.get(marker_name, marker_name),
                    "data": marker_data.data,
                    "shape": marker_data.shape,
                    "dtype": str(marker_data.data.dtype),
                    "dimensions": dimensions,
                    "dimension_labels": dimension_labels,
                    "schema_description": marker_data.schema.describe(),
                    "aggregation": {
                        "channel_method": marker_data.schema.aggregation.channel_method,
                        "trial_method": marker_data.schema.aggregation.trial_method,
                        "connectivity_method": marker_data.schema.aggregation.connectivity_method,
                        "time_method": marker_data.schema.aggregation.time_method,
                        "band_method": marker_data.schema.aggregation.band_method,
                    },
                    "parameters": marker_data.schema.parameters,
                    "raw_metadata": self._serialize_metadata(
                        marker_data.raw_metadata
                    ),
                }

                dump_data["markers"][marker_name] = marker_info
                dump_data["marker_list"].append(marker_name)

            except Exception as e:
                # Include error info for failed markers
                dump_data["markers"][marker_name] = {
                    "name": marker_name,
                    "error": str(e),
                    "data": None,
                }
                dump_data["marker_list"].append(marker_name)

        # Add summary
        dump_data["summary"] = self._generate_dump_summary(dump_data)

        # Save to pickle
        with open(output_path, "wb") as f:
            pickle.dump(dump_data, f)

        return dump_data

    def _get_dimension_description(self, dim_type: DimensionType) -> str:
        """Get human-readable description for a dimension type."""
        descriptions = {
            DimensionType.EPOCHS: "Trial/epoch index (each epoch is one time window of EEG data)",
            DimensionType.CHANNELS: "EEG channel (electrode position)",
            DimensionType.TIMES: "Time point within epoch (in seconds)",
            DimensionType.FREQUENCIES: "Frequency bin (in Hz)",
            DimensionType.BANDS: "Frequency band (e.g., delta, theta, alpha, beta, gamma)",
            DimensionType.CHANNEL_PAIRS: "Channel pair for connectivity (flattened upper triangle)",
            DimensionType.CHANNELS_I: "First channel in connectivity matrix",
            DimensionType.CHANNELS_J: "Second channel in connectivity matrix",
            DimensionType.FEATURES: "Feature index (marker-specific features)",
            DimensionType.SCALAR: "Scalar value (fully aggregated)",
            DimensionType.ELEMENTS: "Element index (subject/session)",
        }
        return descriptions.get(
            dim_type, f"Unknown dimension type: {dim_type.name}"
        )

    def _serialize_metadata(self, metadata: Dict) -> Dict:
        """Serialize metadata to be pickle-compatible."""
        import pickle

        result = {}
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                result[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in value
                ]
            elif isinstance(value, dict):
                result[key] = self._serialize_metadata(value)
            else:
                try:
                    pickle.dumps(value)
                    result[key] = value
                except Exception:
                    result[key] = str(value)
        return result

    def _generate_dump_summary(self, dump_data: Dict) -> str:
        """Generate human-readable summary of dumped data."""
        lines = [
            "=" * 70,
            "Junifer HDF5 Data Dump",
            f"Source: {dump_data['source_filename']}",
            "=" * 70,
            "",
            f"Total markers: {len(dump_data['marker_list'])}",
            "",
            "MARKERS:",
            "-" * 40,
        ]

        for name in dump_data["marker_list"]:
            info = dump_data["markers"][name]
            if info.get("error"):
                lines.append(f"  {name}: ERROR - {info['error']}")
            else:
                lines.append(f"  {name}:")
                lines.append(f"    Shape: {info['shape']}")
                lines.append(f"    Dtype: {info['dtype']}")
                lines.append("    Dimensions:")
                for dim in info["dimensions"]:
                    lines.append(
                        f"      [{dim['axis']}] {dim['type']}: "
                        f"size={dim['size']}"
                    )
                    lines.append(f"          {dim['description']}")
                lines.append("")

        lines.extend(
            [
                "=" * 70,
                "HOW TO USE THIS FILE:",
                "-" * 40,
                "  import pickle",
                "  with open('output.pkl', 'rb') as f:",
                "      data = pickle.load(f)",
                "",
                "  # List all markers",
                "  print(data['marker_list'])",
                "",
                "  # Get a specific marker's data",
                "  marker = data['markers']['marker_name']",
                "  numpy_array = marker['data']",
                "  print(marker['schema_description'])",
                "=" * 70,
            ]
        )

        return "\n".join(lines)


def read_h5(path: Union[str, Path]) -> JuniferH5Reader:
    """Convenience function to create a JuniferH5Reader.

    Parameters
    ----------
    path : str or Path
        Path to HDF5 file.

    Returns
    -------
    JuniferH5Reader
        Reader instance.
    """
    return JuniferH5Reader(path)
