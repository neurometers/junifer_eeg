"""
Marker Registry - Defines per-marker-type dimension expectations.

This module provides a registry of known marker types and their expected
tensor structures, enabling dimension inference when explicit metadata
is not available in the HDF5 file.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

import numpy as np

from .dimension_schema import (
    AggregationInfo,
    DimensionInfo,
    DimensionType,
    TensorSchema,
)


class MarkerType(Enum):
    """Known marker types in junifer_eeg."""

    SPECTRAL = auto()  # SpectralPower, PowerSpectralDensity
    CONNECTIVITY = auto()  # SymbolicMutualInformation (WSMI/SMI)
    ENTROPY = auto()  # PermutationEntropy
    COMPLEXITY = auto()  # KolmogorovComplexity
    ERP = auto()  # TimeLockedTopography
    ERP_CONTRAST = auto()  # TimeLockedContrast
    DECODING = auto()  # WindowDecoding, TimeDecoding
    SLEEP = auto()  # SlowWavesDetection, SpindlesDetection
    CNV = auto()  # ContingentNegativeVariation
    UNKNOWN = auto()


# Keyword mappings for marker type detection
MARKER_TYPE_KEYWORDS = {
    MarkerType.SPECTRAL: [
        "psd",
        "spectral",
        "spectralpower",
        "powerspectral",
        "band_power",
    ],
    MarkerType.CONNECTIVITY: [
        "wsmi",
        "smi",
        "symbolicmutualinformation",
        "connectivity",
        "coherence",
    ],
    MarkerType.ENTROPY: [
        "pe_",
        "permutation",
        "permutationentropy",
        "entropy",
    ],
    MarkerType.COMPLEXITY: [
        "kolmogorov",
        "kolmogorovcomplexity",
        "complexity",
        "lzc",
    ],
    MarkerType.ERP: ["p1", "n1", "p2", "p3", "mmn", "timelockedtopo", "erp"],
    MarkerType.ERP_CONTRAST: [
        "timelockedcontrast",
        "contrast",
        "mmn_contrast",
    ],
    MarkerType.DECODING: [
        "decoding",
        "windowdecoding",
        "timedecoding",
        "generalization",
    ],
    MarkerType.SLEEP: ["slowwaves", "spindles", "slow_waves", "sleep"],
    MarkerType.CNV: ["cnv", "contingent", "contingentvariation"],
}


@dataclass
class MarkerTypeInfo:
    """Information about a marker type's expected structure."""

    marker_type: MarkerType
    base_dimensions: List[DimensionType]  # Dimensions before aggregation
    supports_bands: bool = False
    supports_connectivity_matrix: bool = False
    supports_time_series: bool = False

    def get_expected_dims(
        self,
        has_bands: bool = False,
        has_connectivity: bool = False,
        channel_aggregated: bool = False,
        trial_aggregated: bool = False,
        connectivity_aggregated: bool = False,
    ) -> List[DimensionType]:
        """Get expected dimensions based on configuration."""
        dims = []

        # Add band dimension if multi-band
        if has_bands and self.supports_bands:
            dims.append(DimensionType.BANDS)

        # Add trial dimension if not aggregated
        if not trial_aggregated:
            dims.append(DimensionType.EPOCHS)

        # Handle connectivity
        if self.supports_connectivity_matrix and has_connectivity:
            if not connectivity_aggregated:
                dims.append(DimensionType.CHANNELS_I)
                dims.append(DimensionType.CHANNELS_J)
            elif not channel_aggregated:
                dims.append(DimensionType.CHANNELS)
        elif not channel_aggregated:
            # Regular channel dimension
            dims.append(DimensionType.CHANNELS)

        return dims


# Registry of marker type configurations
MARKER_TYPE_REGISTRY: Dict[MarkerType, MarkerTypeInfo] = {
    MarkerType.SPECTRAL: MarkerTypeInfo(
        marker_type=MarkerType.SPECTRAL,
        base_dimensions=[DimensionType.EPOCHS, DimensionType.CHANNELS],
        supports_bands=True,
    ),
    MarkerType.CONNECTIVITY: MarkerTypeInfo(
        marker_type=MarkerType.CONNECTIVITY,
        base_dimensions=[DimensionType.EPOCHS, DimensionType.CHANNEL_PAIRS],
        supports_connectivity_matrix=True,
    ),
    MarkerType.ENTROPY: MarkerTypeInfo(
        marker_type=MarkerType.ENTROPY,
        base_dimensions=[DimensionType.EPOCHS, DimensionType.CHANNELS],
    ),
    MarkerType.COMPLEXITY: MarkerTypeInfo(
        marker_type=MarkerType.COMPLEXITY,
        base_dimensions=[DimensionType.EPOCHS, DimensionType.CHANNELS],
    ),
    MarkerType.ERP: MarkerTypeInfo(
        marker_type=MarkerType.ERP,
        base_dimensions=[DimensionType.EPOCHS, DimensionType.CHANNELS],
        supports_time_series=True,
    ),
    MarkerType.ERP_CONTRAST: MarkerTypeInfo(
        marker_type=MarkerType.ERP_CONTRAST,
        base_dimensions=[DimensionType.CHANNELS, DimensionType.TIMES],
        supports_time_series=True,
    ),
    MarkerType.DECODING: MarkerTypeInfo(
        marker_type=MarkerType.DECODING,
        base_dimensions=[DimensionType.TIMES],  # Decoding scores over time
    ),
    MarkerType.SLEEP: MarkerTypeInfo(
        marker_type=MarkerType.SLEEP,
        base_dimensions=[
            DimensionType.FEATURES,
            DimensionType.EPOCHS,
            DimensionType.CHANNELS,
        ],
    ),
    MarkerType.CNV: MarkerTypeInfo(
        marker_type=MarkerType.CNV,
        base_dimensions=[DimensionType.EPOCHS, DimensionType.CHANNELS],
    ),
}


class MarkerRegistry:
    """Registry for marker type detection and schema inference."""

    def __init__(self):
        """Initialize the registry."""
        self._custom_keywords: Dict[MarkerType, List[str]] = {}

    def detect_marker_type(self, name: str) -> MarkerType:
        """Detect marker type from name using keywords."""
        name_lower = name.lower()

        # Check custom keywords first
        for mtype, keywords in self._custom_keywords.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return mtype

        # Check built-in keywords
        for mtype, keywords in MARKER_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return mtype

        return MarkerType.UNKNOWN

    def register_keywords(self, marker_type: MarkerType, keywords: List[str]):
        """Register custom keywords for a marker type."""
        if marker_type not in self._custom_keywords:
            self._custom_keywords[marker_type] = []
        self._custom_keywords[marker_type].extend(keywords)

    def get_type_info(
        self, marker_type: MarkerType
    ) -> Optional[MarkerTypeInfo]:
        """Get type info for a marker type."""
        return MARKER_TYPE_REGISTRY.get(marker_type)

    def infer_schema(
        self,
        name: str,
        data: np.ndarray,
        col_names: Optional[List[str]] = None,
        row_names: Optional[List[str]] = None,
        kind: Optional[str] = None,
    ) -> TensorSchema:
        """Infer tensor schema from marker name and data shape.

        This is a fallback for when explicit schema is not stored.
        """
        marker_type = self.detect_marker_type(name)
        type_info = self.get_type_info(marker_type)

        # Build dimensions based on shape and marker type
        dimensions = self._infer_dimensions(
            marker_type, type_info, data, col_names, row_names
        )

        return TensorSchema(
            marker_type=marker_type.name.lower(),
            marker_name=name,
            dimensions=dimensions,
            aggregation=AggregationInfo(),
        )

    def _safe_labels(
        self, labels: Optional[List[str]], size: int
    ) -> Optional[List[str]]:
        """Return labels only if they match the expected size."""
        if labels is not None and len(labels) == size:
            return labels
        return None

    def _infer_dimensions(
        self,
        marker_type: MarkerType,
        type_info: Optional[MarkerTypeInfo],
        data: np.ndarray,
        col_names: Optional[List[str]],
        row_names: Optional[List[str]],
    ) -> List[DimensionInfo]:
        """Infer dimension information from data shape and type."""
        shape = data.shape

        # Use marker-type-specific inference (shape used below)
        if marker_type == MarkerType.SPECTRAL:
            return self._infer_spectral_dims(shape, col_names)
        elif marker_type == MarkerType.CONNECTIVITY:
            return self._infer_connectivity_dims(shape, col_names)
        elif marker_type in (
            MarkerType.ENTROPY,
            MarkerType.COMPLEXITY,
            MarkerType.CNV,
        ):
            return self._infer_channel_epoch_dims(shape, col_names)
        elif marker_type == MarkerType.ERP:
            return self._infer_erp_dims(shape, col_names)
        elif marker_type == MarkerType.SLEEP:
            return self._infer_sleep_dims(shape, col_names, row_names)
        else:
            return self._infer_generic_dims(shape, col_names)

    def _infer_spectral_dims(
        self, shape: tuple, col_names: Optional[List[str]]
    ) -> List[DimensionInfo]:
        """Infer dimensions for spectral markers."""
        dims = []

        if len(shape) == 3:
            # (bands, epochs, channels)
            n_bands = shape[0]
            band_names = self._guess_band_names(n_bands)
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.BANDS,
                    size=n_bands,
                    labels=band_names,
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.EPOCHS,
                    size=shape[1],
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.CHANNELS,
                    size=shape[2],
                    labels=self._safe_labels(col_names, shape[2]),
                )
            )
        elif len(shape) == 2:
            # (epochs, channels)
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.EPOCHS,
                    size=shape[0],
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.CHANNELS,
                    size=shape[1],
                    labels=self._safe_labels(col_names, shape[1]),
                )
            )
        elif len(shape) == 1:
            # Aggregated to 1D
            if col_names and len(col_names) == shape[0]:
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.CHANNELS,
                        size=shape[0],
                        labels=col_names,
                    )
                )
            else:
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.EPOCHS,
                        size=shape[0],
                    )
                )

        return dims

    def _infer_connectivity_dims(
        self, shape: tuple, col_names: Optional[List[str]]
    ) -> List[DimensionInfo]:
        """Infer dimensions for connectivity markers."""
        dims = []

        if len(shape) == 3:
            # Could be (epochs, ch_i, ch_j) or (epochs, channels, channels)
            if shape[1] == shape[2]:
                # Square matrix: (epochs, channels, channels)
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.EPOCHS,
                        size=shape[0],
                    )
                )
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.CHANNELS_I,
                        size=shape[1],
                        labels=self._safe_labels(col_names, shape[1]),
                    )
                )
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.CHANNELS_J,
                        size=shape[2],
                        labels=self._safe_labels(col_names, shape[2]),
                    )
                )
            else:
                # Non-square: treat as generic
                dims = self._infer_generic_dims(shape, col_names)
        elif len(shape) == 2:
            # (epochs, channel_pairs) - flattened connectivity
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.EPOCHS,
                    size=shape[0],
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.CHANNEL_PAIRS,
                    size=shape[1],
                    labels=self._safe_labels(col_names, shape[1]),
                )
            )

        return dims

    def _infer_channel_epoch_dims(
        self, shape: tuple, col_names: Optional[List[str]]
    ) -> List[DimensionInfo]:
        """Infer dimensions for standard (epochs, channels) markers."""
        dims = []

        if len(shape) == 2:
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.EPOCHS,
                    size=shape[0],
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.CHANNELS,
                    size=shape[1],
                    labels=self._safe_labels(col_names, shape[1]),
                )
            )
        elif len(shape) == 1:
            # Aggregated - check col_names to determine which dimension
            if col_names and len(col_names) == shape[0]:
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.CHANNELS,
                        size=shape[0],
                        labels=col_names,
                    )
                )
            else:
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.EPOCHS,
                        size=shape[0],
                    )
                )

        return dims

    def _infer_erp_dims(
        self, shape: tuple, col_names: Optional[List[str]]
    ) -> List[DimensionInfo]:
        """Infer dimensions for ERP markers."""
        dims = []

        if len(shape) == 3:
            # (epochs, channels, times)
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.EPOCHS,
                    size=shape[0],
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.CHANNELS,
                    size=shape[1],
                    labels=self._safe_labels(col_names, shape[1]),
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.TIMES,
                    size=shape[2],
                )
            )
        elif len(shape) == 2:
            # Could be (epochs, channels) after time averaging
            # or (channels, times) after epoch averaging
            # Heuristic: if last dim is large, likely times
            if shape[1] > 100:  # Likely time points
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.CHANNELS,
                        size=shape[0],
                        labels=self._safe_labels(col_names, shape[0]),
                    )
                )
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.TIMES,
                        size=shape[1],
                    )
                )
            else:
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.EPOCHS,
                        size=shape[0],
                    )
                )
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.CHANNELS,
                        size=shape[1],
                        labels=self._safe_labels(col_names, shape[1]),
                    )
                )
        elif len(shape) == 1:
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.CHANNELS,
                    size=shape[0],
                    labels=self._safe_labels(col_names, shape[0]),
                )
            )

        return dims

    def _infer_sleep_dims(
        self,
        shape: tuple,
        col_names: Optional[List[str]],
        row_names: Optional[List[str]],
    ) -> List[DimensionInfo]:
        """Infer dimensions for sleep markers."""
        dims = []

        if len(shape) == 3:
            # (features, epochs, channels)
            feature_names = self._guess_sleep_feature_names(shape[0])
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.FEATURES,
                    size=shape[0],
                    labels=feature_names,
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.EPOCHS,
                    size=shape[1],
                )
            )
            dims.append(
                DimensionInfo(
                    dim_type=DimensionType.CHANNELS,
                    size=shape[2],
                    labels=self._safe_labels(col_names, shape[2]),
                )
            )
        else:
            dims = self._infer_generic_dims(shape, col_names)

        return dims

    def _infer_generic_dims(
        self, shape: tuple, col_names: Optional[List[str]]
    ) -> List[DimensionInfo]:
        """Fallback dimension inference for unknown markers."""
        dims = []

        for i, size in enumerate(shape):
            # Last dimension with col_names is likely channels
            if i == len(shape) - 1 and col_names and len(col_names) == size:
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.CHANNELS,
                        size=size,
                        labels=col_names,
                    )
                )
            else:
                # Generic dimension
                dims.append(
                    DimensionInfo(
                        dim_type=DimensionType.EPOCHS
                        if i == 0
                        else DimensionType.CHANNELS,
                        size=size,
                    )
                )

        return dims

    def _guess_band_names(self, n_bands: int) -> List[str]:
        """Guess band names based on count."""
        if n_bands == 5:
            return ["delta", "theta", "alpha", "beta", "gamma"]
        elif n_bands == 6:
            return ["delta", "theta", "alpha", "beta", "jota", "gamma"]
        elif n_bands == 4:
            return ["delta", "theta", "alpha", "beta"]
        else:
            return [f"band_{i}" for i in range(n_bands)]

    def _guess_sleep_feature_names(self, n_features: int) -> List[str]:
        """Guess sleep feature names based on count."""
        if n_features == 5:
            return ["Duration", "PTP", "Frequency", "Slope", "Density"]
        elif n_features == 4:
            return ["Duration", "Amplitude", "Frequency", "Density"]
        else:
            return [f"feature_{i}" for i in range(n_features)]


# Global registry instance
_default_registry = MarkerRegistry()


def get_registry() -> MarkerRegistry:
    """Get the default marker registry."""
    return _default_registry


def detect_marker_type(name: str) -> MarkerType:
    """Detect marker type from name (convenience function)."""
    return _default_registry.detect_marker_type(name)


def infer_schema(
    name: str,
    data: np.ndarray,
    col_names: Optional[List[str]] = None,
    row_names: Optional[List[str]] = None,
    kind: Optional[str] = None,
) -> TensorSchema:
    """Infer tensor schema (convenience function)."""
    return _default_registry.infer_schema(
        name, data, col_names, row_names, kind
    )
