"""
Modular HDF5 Reader for junifer_eeg markers.

This package provides a dimension-aware reader for junifer HDF5 output files,
with support for all marker types and aggregation configurations.

Modules
-------
dimension_schema : Dimension type definitions and semantics
marker_registry : Per-marker-type dimension expectations
aggregators : Data aggregation utilities
reader : Main reader class with dimension-aware loading
schema_builder : Utilities for markers to build tensor schemas

Usage
-----
Reading data:
    >>> from junifer_eeg.reader import read_h5, DimensionType
    >>> reader = read_h5("output.h5")
    >>> data = reader.get("psd_alpha")
    >>> print(data.schema.describe())
    >>> # Aggregate across channels
    >>> agg = data.aggregate("mean", DimensionType.CHANNELS)

Building schemas in markers:
    >>> from junifer_eeg.reader import SchemaBuilder, schema_to_storage_kwargs
    >>> builder = SchemaBuilder("spectral", "psd_alpha")
    >>> schema = builder.add_epochs(100).add_channels(64, ch_names).build()
    >>> output["tensor_schema"] = schema.to_dict()
"""

from .aggregators import (
    AGGREGATION_FUNCTIONS,
    DataAggregator,
    aggregate,
    aggregate_channels,
    aggregate_connectivity,
    aggregate_epochs,
    aggregate_time,
    flatten_connectivity_matrix,
    generate_pair_names,
    unflatten_connectivity_matrix,
)
from .dimension_schema import (
    EXTENDED_BANDS,
    STANDARD_BANDS,
    AggregationInfo,
    DimensionInfo,
    DimensionType,
    TensorSchema,
    create_band_dim,
    create_channel_dim,
    create_channel_pair_dim,
    create_epoch_dim,
    create_time_dim,
)
from .marker_mixin import SchemaAwareMixin
from .marker_registry import (
    MarkerRegistry,
    MarkerType,
    MarkerTypeInfo,
    detect_marker_type,
    get_registry,
    infer_schema,
)
from .reader import JuniferH5Reader, MarkerData, read_h5
from .schema_builder import (
    SchemaBuilder,
    build_connectivity_schema,
    build_entropy_schema,
    build_erp_schema,
    build_spectral_schema,
    schema_to_storage_kwargs,
)

__all__ = [
    "AGGREGATION_FUNCTIONS",
    "EXTENDED_BANDS",
    "STANDARD_BANDS",
    "AggregationInfo",
    "DataAggregator",
    "DimensionInfo",
    "DimensionType",
    "JuniferH5Reader",
    "MarkerData",
    "MarkerRegistry",
    "MarkerType",
    "MarkerTypeInfo",
    "SchemaAwareMixin",
    "SchemaBuilder",
    "TensorSchema",
    "aggregate",
    "aggregate_channels",
    "aggregate_connectivity",
    "aggregate_epochs",
    "aggregate_time",
    "build_connectivity_schema",
    "build_entropy_schema",
    "build_erp_schema",
    "build_spectral_schema",
    "create_band_dim",
    "create_channel_dim",
    "create_channel_pair_dim",
    "create_epoch_dim",
    "create_time_dim",
    "detect_marker_type",
    "flatten_connectivity_matrix",
    "generate_pair_names",
    "get_registry",
    "infer_schema",
    "read_h5",
    "schema_to_storage_kwargs",
    "unflatten_connectivity_matrix",
]
