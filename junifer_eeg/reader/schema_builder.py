"""
Schema Builder - Utilities for markers to build and store tensor schemas.

This module provides helper functions that markers can use to build
TensorSchema objects and include them in their output for HDF5 storage.
"""

from typing import Any, Dict, List, Optional

from .dimension_schema import (
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


class SchemaBuilder:
    """Builder class for creating TensorSchema objects.

    This provides a fluent interface for markers to build schemas
    describing their output tensors.

    Examples
    --------
    >>> builder = SchemaBuilder("spectral", "psd_alpha")
    >>> schema = (builder
    ...     .add_epochs(n_epochs, epoch_indices)
    ...     .add_channels(n_channels, channel_names)
    ...     .set_channel_aggregation("mean")
    ...     .add_parameter("fmin", 8.0)
    ...     .build())
    """

    def __init__(self, marker_type: str, marker_name: str):
        """Initialize the builder.

        Parameters
        ----------
        marker_type : str
            Type of marker (e.g., 'spectral', 'connectivity').
        marker_name : str
            Specific marker name.
        """
        self.marker_type = marker_type
        self.marker_name = marker_name
        self.dimensions: List[DimensionInfo] = []
        self.aggregation = AggregationInfo()
        self.parameters: Dict[str, Any] = {}

    def add_dimension(self, dim_info: DimensionInfo) -> "SchemaBuilder":
        """Add a dimension to the schema.

        Parameters
        ----------
        dim_info : DimensionInfo
            Dimension information.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(dim_info)
        return self

    def add_epochs(
        self, n_epochs: int, epoch_indices: Optional[List[int]] = None
    ) -> "SchemaBuilder":
        """Add an epoch dimension.

        Parameters
        ----------
        n_epochs : int
            Number of epochs.
        epoch_indices : list of int, optional
            Original epoch indices.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(create_epoch_dim(n_epochs, epoch_indices))
        return self

    def add_channels(
        self, n_channels: int, channel_names: Optional[List[str]] = None
    ) -> "SchemaBuilder":
        """Add a channel dimension.

        Parameters
        ----------
        n_channels : int
            Number of channels.
        channel_names : list of str, optional
            Channel names.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(create_channel_dim(n_channels, channel_names))
        return self

    def add_bands(
        self,
        band_names: List[str],
        band_ranges: Optional[Dict[str, tuple]] = None,
    ) -> "SchemaBuilder":
        """Add a frequency band dimension.

        Parameters
        ----------
        band_names : list of str
            Band names.
        band_ranges : dict, optional
            Mapping of band names to (fmin, fmax) tuples.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(create_band_dim(band_names, band_ranges))
        return self

    def add_times(
        self,
        n_times: int,
        times: Optional[List[float]] = None,
        sfreq: Optional[float] = None,
    ) -> "SchemaBuilder":
        """Add a time dimension.

        Parameters
        ----------
        n_times : int
            Number of time points.
        times : list of float, optional
            Time values.
        sfreq : float, optional
            Sampling frequency.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(create_time_dim(n_times, times, sfreq))
        return self

    def add_channel_pairs(
        self, n_pairs: int, pair_names: Optional[List[str]] = None
    ) -> "SchemaBuilder":
        """Add a channel pair dimension for connectivity.

        Parameters
        ----------
        n_pairs : int
            Number of channel pairs.
        pair_names : list of str, optional
            Pair names in format "ch1-ch2".

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(create_channel_pair_dim(n_pairs, pair_names))
        return self

    def add_connectivity_matrix(
        self, n_channels: int, channel_names: Optional[List[str]] = None
    ) -> "SchemaBuilder":
        """Add two dimensions for connectivity matrix (channels_i x channels_j).

        Parameters
        ----------
        n_channels : int
            Number of channels.
        channel_names : list of str, optional
            Channel names.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(
            DimensionInfo(
                dim_type=DimensionType.CHANNELS_I,
                size=n_channels,
                labels=channel_names,
            )
        )
        self.dimensions.append(
            DimensionInfo(
                dim_type=DimensionType.CHANNELS_J,
                size=n_channels,
                labels=channel_names,
            )
        )
        return self

    def add_features(
        self, n_features: int, feature_names: Optional[List[str]] = None
    ) -> "SchemaBuilder":
        """Add a feature dimension.

        Parameters
        ----------
        n_features : int
            Number of features.
        feature_names : list of str, optional
            Feature names.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.dimensions.append(
            DimensionInfo(
                dim_type=DimensionType.FEATURES,
                size=n_features,
                labels=feature_names,
            )
        )
        return self

    def set_channel_aggregation(self, method: str) -> "SchemaBuilder":
        """Set channel aggregation method.

        Parameters
        ----------
        method : str
            Aggregation method name.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.aggregation.channel_method = method
        return self

    def set_trial_aggregation(self, method: str) -> "SchemaBuilder":
        """Set trial/epoch aggregation method.

        Parameters
        ----------
        method : str
            Aggregation method name.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.aggregation.trial_method = method
        return self

    def set_connectivity_aggregation(self, method: str) -> "SchemaBuilder":
        """Set connectivity aggregation method.

        Parameters
        ----------
        method : str
            Aggregation method name.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.aggregation.connectivity_method = method
        return self

    def set_time_aggregation(self, method: str) -> "SchemaBuilder":
        """Set time aggregation method.

        Parameters
        ----------
        method : str
            Aggregation method name.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.aggregation.time_method = method
        return self

    def add_parameter(self, key: str, value: Any) -> "SchemaBuilder":
        """Add a marker parameter.

        Parameters
        ----------
        key : str
            Parameter name.
        value : Any
            Parameter value.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.parameters[key] = value
        return self

    def add_parameters(self, params: Dict[str, Any]) -> "SchemaBuilder":
        """Add multiple marker parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameters.

        Returns
        -------
        SchemaBuilder
            Self for chaining.
        """
        self.parameters.update(params)
        return self

    def build(self) -> TensorSchema:
        """Build the TensorSchema.

        Returns
        -------
        TensorSchema
            The completed schema.
        """
        return TensorSchema(
            marker_type=self.marker_type,
            marker_name=self.marker_name,
            dimensions=self.dimensions,
            aggregation=self.aggregation,
            parameters=self.parameters if self.parameters else None,
        )


def build_spectral_schema(
    marker_name: str,
    n_epochs: int,
    n_channels: int,
    channel_names: Optional[List[str]] = None,
    band_names: Optional[List[str]] = None,
    band_ranges: Optional[Dict[str, tuple]] = None,
    channel_method: Optional[str] = None,
    trial_method: Optional[str] = None,
    **params,
) -> TensorSchema:
    """Build schema for spectral power markers.

    Parameters
    ----------
    marker_name : str
        Marker name.
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    channel_names : list of str, optional
        Channel names.
    band_names : list of str, optional
        Band names for multi-band markers.
    band_ranges : dict, optional
        Band frequency ranges.
    channel_method : str, optional
        Channel aggregation method.
    trial_method : str, optional
        Trial aggregation method.
    **params
        Additional parameters (fmin, fmax, etc.).

    Returns
    -------
    TensorSchema
        Built schema.
    """
    builder = SchemaBuilder("spectral", marker_name)

    # Add bands if multi-band
    if band_names and len(band_names) > 1:
        builder.add_bands(band_names, band_ranges)

    # Add epochs if not aggregated
    if trial_method is None:
        builder.add_epochs(n_epochs)

    # Add channels if not aggregated
    if channel_method is None:
        builder.add_channels(n_channels, channel_names)

    # Set aggregation info
    if channel_method:
        builder.set_channel_aggregation(channel_method)
    if trial_method:
        builder.set_trial_aggregation(trial_method)

    # Add parameters
    if params:
        builder.add_parameters(params)

    return builder.build()


def build_connectivity_schema(
    marker_name: str,
    n_epochs: int,
    n_channels: int,
    channel_names: Optional[List[str]] = None,
    flattened: bool = True,
    pair_names: Optional[List[str]] = None,
    n_channel_pairs: Optional[int] = None,
    tau_names: Optional[List[str]] = None,
    connectivity_method: Optional[str] = None,
    channel_method: Optional[str] = None,
    trial_method: Optional[str] = None,
    **params,
) -> TensorSchema:
    """Build schema for connectivity markers (WSMI, SMI, coherence, etc.).

    Parameters
    ----------
    marker_name : str
        Marker name.
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    channel_names : list of str, optional
        Channel names.
    flattened : bool
        Whether connectivity is flattened to pairs.
    pair_names : list of str, optional
        Pair names for flattened connectivity.
    n_channel_pairs : int, optional
        Number of channel pairs (if different from n_channels*(n_channels-1)//2).
    tau_names : list of str, optional
        Names for tau/timescale dimension (for multi-tau WSMI).
        If provided, adds a BANDS dimension for the taus.
    connectivity_method : str, optional
        Connectivity aggregation method.
    channel_method : str, optional
        Channel aggregation method.
    trial_method : str, optional
        Trial aggregation method.
    **params
        Additional parameters.

    Returns
    -------
    TensorSchema
        Built schema.

    Examples
    --------
    >>> # Single-tau WSMI: shape (n_epochs, n_channel_pairs)
    >>> schema = build_connectivity_schema(
    ...     marker_name='wsmi_theta',
    ...     n_epochs=100,
    ...     n_channels=256,
    ... )

    >>> # Multi-tau WSMI: shape (n_taus, n_epochs, n_channel_pairs)
    >>> schema = build_connectivity_schema(
    ...     marker_name='wsmi_multiscale',
    ...     n_epochs=100,
    ...     n_channels=256,
    ...     tau_names=['tau_8_theta', 'tau_4_alpha', 'tau_2_beta', 'tau_1_gamma'],
    ... )
    """
    builder = SchemaBuilder("connectivity", marker_name)

    # Add tau/timescale dimension if multi-tau (WSMI with multiple taus)
    # This comes FIRST because shape is (n_taus, n_epochs, n_channel_pairs)
    if tau_names is not None and len(tau_names) > 0:
        builder.add_bands(tau_names)

    # Add epochs if not aggregated
    if trial_method is None:
        builder.add_epochs(n_epochs)

    # Add connectivity dimensions
    if flattened:
        if n_channel_pairs is None:
            n_channel_pairs = n_channels * (n_channels - 1) // 2
        builder.add_channel_pairs(n_channel_pairs, pair_names)
    else:
        if connectivity_method is None:
            builder.add_connectivity_matrix(n_channels, channel_names)
        elif channel_method is None:
            builder.add_channels(n_channels, channel_names)

    # Set aggregation info
    if connectivity_method:
        builder.set_connectivity_aggregation(connectivity_method)
    if channel_method:
        builder.set_channel_aggregation(channel_method)
    if trial_method:
        builder.set_trial_aggregation(trial_method)

    if params:
        builder.add_parameters(params)

    return builder.build()


def build_entropy_schema(
    marker_name: str,
    n_epochs: int,
    n_channels: int,
    channel_names: Optional[List[str]] = None,
    channel_method: Optional[str] = None,
    trial_method: Optional[str] = None,
    **params,
) -> TensorSchema:
    """Build schema for entropy/complexity markers.

    Parameters
    ----------
    marker_name : str
        Marker name.
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    channel_names : list of str, optional
        Channel names.
    channel_method : str, optional
        Channel aggregation method.
    trial_method : str, optional
        Trial aggregation method.
    **params
        Additional parameters.

    Returns
    -------
    TensorSchema
        Built schema.
    """
    builder = SchemaBuilder("entropy", marker_name)

    if trial_method is None:
        builder.add_epochs(n_epochs)

    if channel_method is None:
        builder.add_channels(n_channels, channel_names)

    if channel_method:
        builder.set_channel_aggregation(channel_method)
    if trial_method:
        builder.set_trial_aggregation(trial_method)

    if params:
        builder.add_parameters(params)

    return builder.build()


def build_erp_schema(
    marker_name: str,
    n_epochs: int,
    n_channels: int,
    n_times: Optional[int] = None,
    channel_names: Optional[List[str]] = None,
    times: Optional[List[float]] = None,
    sfreq: Optional[float] = None,
    time_aggregated: bool = True,
    channel_method: Optional[str] = None,
    trial_method: Optional[str] = None,
    **params,
) -> TensorSchema:
    """Build schema for ERP/time-locked markers.

    Parameters
    ----------
    marker_name : str
        Marker name.
    n_epochs : int
        Number of epochs.
    n_channels : int
        Number of channels.
    n_times : int, optional
        Number of time points (if not time-aggregated).
    channel_names : list of str, optional
        Channel names.
    times : list of float, optional
        Time values.
    sfreq : float, optional
        Sampling frequency.
    time_aggregated : bool
        Whether time dimension is averaged.
    channel_method : str, optional
        Channel aggregation method.
    trial_method : str, optional
        Trial aggregation method.
    **params
        Additional parameters.

    Returns
    -------
    TensorSchema
        Built schema.
    """
    builder = SchemaBuilder("erp", marker_name)

    if trial_method is None:
        builder.add_epochs(n_epochs)

    if channel_method is None:
        builder.add_channels(n_channels, channel_names)

    if not time_aggregated and n_times is not None:
        builder.add_times(n_times, times, sfreq)

    if time_aggregated:
        builder.set_time_aggregation("mean")
    if channel_method:
        builder.set_channel_aggregation(channel_method)
    if trial_method:
        builder.set_trial_aggregation(trial_method)

    if params:
        builder.add_parameters(params)

    return builder.build()


def schema_to_storage_kwargs(schema: TensorSchema) -> Dict[str, Any]:
    """Convert TensorSchema to kwargs for HDF5 storage.

    This is used by markers to include schema in their output
    so it gets stored in the HDF5 file.

    Parameters
    ----------
    schema : TensorSchema
        The tensor schema.

    Returns
    -------
    dict
        Kwargs to pass to storage method.
    """
    return {"tensor_schema": schema.to_dict()}
