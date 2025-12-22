"""
Dimension Schema - Defines dimension types and their semantics for EEG marker tensors.

This module provides the foundational data structures for describing what each
dimension of a marker output tensor represents (epochs, channels, bands, etc.).
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class DimensionType(Enum):
    """Types of dimensions that can appear in marker output tensors."""

    # Core EEG dimensions
    EPOCHS = auto()  # Trial/epoch dimension
    CHANNELS = auto()  # EEG channel dimension
    TIMES = auto()  # Time points within epoch
    FREQUENCIES = auto()  # Frequency bins (raw PSD)

    # Derived/aggregated dimensions
    BANDS = auto()  # Frequency bands (delta, theta, alpha, etc.)
    CHANNEL_PAIRS = auto()  # Flattened connectivity pairs (ch1-ch2)
    CHANNELS_I = auto()  # First channel in connectivity matrix
    CHANNELS_J = auto()  # Second channel in connectivity matrix
    FEATURES = auto()  # Generic feature dimension (sleep features, etc.)

    # Scalar dimensions (after full aggregation)
    SCALAR = auto()  # Single value (fully aggregated)

    # Element dimension (across subjects/sessions)
    ELEMENTS = auto()  # Multiple elements (subjects/sessions)


@dataclass
class DimensionInfo:
    """Information about a single dimension of a tensor.

    Attributes
    ----------
    dim_type : DimensionType
        The semantic type of this dimension.
    size : int
        The size of this dimension.
    labels : list, optional
        Labels for each index (channel names, band names, etc.).
    metadata : dict, optional
        Additional metadata (e.g., frequency ranges for bands).
    """

    dim_type: DimensionType
    size: int
    labels: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate dimension info."""
        if self.labels is not None and len(self.labels) != self.size:
            raise ValueError(
                f"Labels length ({len(self.labels)}) must match size ({self.size})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HDF5 storage."""
        return {
            "dim_type": self.dim_type.name,
            "size": self.size,
            "labels": self.labels,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DimensionInfo":
        """Create from dictionary (loaded from HDF5)."""
        return cls(
            dim_type=DimensionType[d["dim_type"]],
            size=d["size"],
            labels=d.get("labels"),
            metadata=d.get("metadata"),
        )


@dataclass
class AggregationInfo:
    """Information about aggregation applied to the marker.

    Attributes
    ----------
    channel_method : str, optional
        Aggregation method applied across channels.
    trial_method : str, optional
        Aggregation method applied across trials/epochs.
    connectivity_method : str, optional
        Aggregation method applied to connectivity dimension.
    time_method : str, optional
        Aggregation method applied across time (e.g., 'mean' for ERP).
    band_method : str, optional
        Aggregation method applied across frequency bands.
    """

    channel_method: Optional[str] = None
    trial_method: Optional[str] = None
    connectivity_method: Optional[str] = None
    time_method: Optional[str] = None
    band_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HDF5 storage."""
        return {
            "channel_method": self.channel_method,
            "trial_method": self.trial_method,
            "connectivity_method": self.connectivity_method,
            "time_method": self.time_method,
            "band_method": self.band_method,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AggregationInfo":
        """Create from dictionary (loaded from HDF5)."""
        return cls(
            channel_method=d.get("channel_method"),
            trial_method=d.get("trial_method"),
            connectivity_method=d.get("connectivity_method"),
            time_method=d.get("time_method"),
            band_method=d.get("band_method"),
        )

    @property
    def is_fully_aggregated(self) -> bool:
        """Check if all possible dimensions are aggregated."""
        return (
            self.channel_method is not None and self.trial_method is not None
        )


@dataclass
class TensorSchema:
    """Complete schema describing a marker output tensor.

    Attributes
    ----------
    marker_type : str
        Type of marker (spectral, connectivity, entropy, erp, etc.).
    marker_name : str
        Specific marker name.
    dimensions : list of DimensionInfo
        Ordered list of dimension information (axis 0, 1, 2, ...).
    aggregation : AggregationInfo
        Information about aggregation applied.
    parameters : dict, optional
        Marker-specific parameters used in computation.
    """

    marker_type: str
    marker_name: str
    dimensions: List[DimensionInfo]
    aggregation: AggregationInfo = field(default_factory=AggregationInfo)
    parameters: Optional[Dict[str, Any]] = None

    @property
    def shape(self) -> tuple:
        """Get expected tensor shape."""
        return tuple(d.size for d in self.dimensions)

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return len(self.dimensions)

    @property
    def dim_types(self) -> List[DimensionType]:
        """Get list of dimension types."""
        return [d.dim_type for d in self.dimensions]

    def get_axis(self, dim_type: DimensionType) -> Optional[int]:
        """Get axis index for a dimension type, or None if not present."""
        for i, d in enumerate(self.dimensions):
            if d.dim_type == dim_type:
                return i
        return None

    def get_dim(self, dim_type: DimensionType) -> Optional[DimensionInfo]:
        """Get DimensionInfo for a dimension type, or None if not present."""
        for d in self.dimensions:
            if d.dim_type == dim_type:
                return d
        return None

    def has_dimension(self, dim_type: DimensionType) -> bool:
        """Check if tensor has a specific dimension type."""
        return any(d.dim_type == dim_type for d in self.dimensions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for HDF5 storage."""
        return {
            "marker_type": self.marker_type,
            "marker_name": self.marker_name,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "aggregation": self.aggregation.to_dict(),
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorSchema":
        """Create from dictionary (loaded from HDF5)."""
        return cls(
            marker_type=d["marker_type"],
            marker_name=d["marker_name"],
            dimensions=[
                DimensionInfo.from_dict(dim) for dim in d["dimensions"]
            ],
            aggregation=AggregationInfo.from_dict(d.get("aggregation", {})),
            parameters=d.get("parameters"),
        )

    def describe(self) -> str:
        """Get human-readable description of the tensor schema."""
        lines = [
            f"TensorSchema: {self.marker_name} ({self.marker_type})",
            f"  Shape: {self.shape}",
            "  Dimensions:",
        ]
        for i, dim in enumerate(self.dimensions):
            labels_str = f", labels={len(dim.labels)}" if dim.labels else ""
            lines.append(
                f"    [{i}] {dim.dim_type.name}: size={dim.size}{labels_str}"
            )

        agg_parts = []
        if self.aggregation.channel_method:
            agg_parts.append(f"channels={self.aggregation.channel_method}")
        if self.aggregation.trial_method:
            agg_parts.append(f"trials={self.aggregation.trial_method}")
        if self.aggregation.connectivity_method:
            agg_parts.append(
                f"connectivity={self.aggregation.connectivity_method}"
            )
        if self.aggregation.time_method:
            agg_parts.append(f"time={self.aggregation.time_method}")

        if agg_parts:
            lines.append(f"  Aggregation: {', '.join(agg_parts)}")
        else:
            lines.append("  Aggregation: None")

        return "\n".join(lines)


# Standard band definitions for reference
STANDARD_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "gamma": (30.0, 45.0),
}

EXTENDED_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 30.0),
    "jota": (30.0, 45.0),
    "gamma": (45.0, 80.0),
}


def create_channel_dim(
    n_channels: int, channel_names: Optional[List[str]] = None
) -> DimensionInfo:
    """Helper to create a channel dimension."""
    return DimensionInfo(
        dim_type=DimensionType.CHANNELS,
        size=n_channels,
        labels=channel_names,
    )


def create_epoch_dim(
    n_epochs: int, epoch_indices: Optional[List[int]] = None
) -> DimensionInfo:
    """Helper to create an epoch dimension."""
    labels = [str(i) for i in (epoch_indices or range(n_epochs))]
    return DimensionInfo(
        dim_type=DimensionType.EPOCHS,
        size=n_epochs,
        labels=labels,
    )


def create_band_dim(
    band_names: List[str], band_ranges: Optional[Dict[str, tuple]] = None
) -> DimensionInfo:
    """Helper to create a frequency band dimension."""
    return DimensionInfo(
        dim_type=DimensionType.BANDS,
        size=len(band_names),
        labels=band_names,
        metadata={"ranges": band_ranges} if band_ranges else None,
    )


def create_channel_pair_dim(
    n_pairs: int, pair_names: Optional[List[str]] = None
) -> DimensionInfo:
    """Helper to create a channel pair dimension for connectivity."""
    return DimensionInfo(
        dim_type=DimensionType.CHANNEL_PAIRS,
        size=n_pairs,
        labels=pair_names,
    )


def create_time_dim(
    n_times: int,
    times: Optional[List[float]] = None,
    sfreq: Optional[float] = None,
) -> DimensionInfo:
    """Helper to create a time dimension."""
    labels = [f"{t:.3f}" for t in times] if times else None
    metadata = {"sfreq": sfreq} if sfreq else None
    return DimensionInfo(
        dim_type=DimensionType.TIMES,
        size=n_times,
        labels=labels,
        metadata=metadata,
    )
