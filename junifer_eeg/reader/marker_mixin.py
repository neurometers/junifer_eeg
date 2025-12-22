"""
Marker Mixin - Base mixin class for markers to support schema storage.

This module provides a mixin class that markers can inherit from to
automatically include tensor schema metadata in their output.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .dimension_schema import TensorSchema
from .schema_builder import SchemaBuilder


class SchemaAwareMixin:
    """Mixin class for markers to support tensor schema storage.

    Markers can inherit from this mixin to gain helper methods for
    building and including tensor schemas in their output.

    The schema is stored as a dictionary under the 'tensor_schema' key
    in the marker output, which gets persisted to HDF5.

    Example
    -------
    class MyMarker(BaseMarker, SchemaAwareMixin):
        def compute(self, input, extra_input=None):
            # ... compute data ...

            # Build schema
            schema = self.build_schema(
                marker_type="spectral",
                n_epochs=n_epochs,
                n_channels=n_channels,
                channel_names=ch_names,
            )

            # Return with schema
            return self.wrap_output(
                key="myfeature",
                data=result_data,
                schema=schema,
                col_names=ch_names,
            )
    """

    def build_schema(
        self,
        marker_type: str,
        n_epochs: Optional[int] = None,
        n_channels: Optional[int] = None,
        channel_names: Optional[List[str]] = None,
        band_names: Optional[List[str]] = None,
        band_ranges: Optional[Dict[str, tuple]] = None,
        n_times: Optional[int] = None,
        times: Optional[List[float]] = None,
        sfreq: Optional[float] = None,
        n_pairs: Optional[int] = None,
        pair_names: Optional[List[str]] = None,
        n_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        connectivity_method: Optional[str] = None,
        time_method: Optional[str] = None,
        **params,
    ) -> TensorSchema:
        """Build a TensorSchema for the marker output.

        This method creates a schema that describes the dimensions
        of the output tensor. Call this in compute() before returning.

        Parameters
        ----------
        marker_type : str
            Type of marker ('spectral', 'connectivity', 'entropy', 'erp', etc.).
        n_epochs : int, optional
            Number of epochs (if not trial-aggregated).
        n_channels : int, optional
            Number of channels (if not channel-aggregated).
        channel_names : list of str, optional
            Channel names.
        band_names : list of str, optional
            Frequency band names for multi-band markers.
        band_ranges : dict, optional
            Band frequency ranges {band_name: (fmin, fmax)}.
        n_times : int, optional
            Number of time points (for time-series output).
        times : list of float, optional
            Time values.
        sfreq : float, optional
            Sampling frequency.
        n_pairs : int, optional
            Number of channel pairs (for flattened connectivity).
        pair_names : list of str, optional
            Channel pair names.
        n_features : int, optional
            Number of features (for feature-based markers).
        feature_names : list of str, optional
            Feature names.
        channel_method : str, optional
            Channel aggregation method applied.
        trial_method : str, optional
            Trial aggregation method applied.
        connectivity_method : str, optional
            Connectivity aggregation method applied.
        time_method : str, optional
            Time aggregation method applied.
        **params
            Additional marker parameters to store.

        Returns
        -------
        TensorSchema
            The built schema.
        """
        # Get marker name from self if available
        marker_name = getattr(self, "name", None) or self.__class__.__name__

        builder = SchemaBuilder(marker_type, marker_name)

        # Add dimensions in order (bands first if present)
        if band_names and len(band_names) > 1:
            builder.add_bands(band_names, band_ranges)

        if n_features is not None and feature_names:
            builder.add_features(n_features, feature_names)

        if n_epochs is not None and trial_method is None:
            builder.add_epochs(n_epochs)

        if n_pairs is not None and connectivity_method is None:
            builder.add_channel_pairs(n_pairs, pair_names)
        elif n_channels is not None and channel_method is None:
            builder.add_channels(n_channels, channel_names)

        if n_times is not None and time_method is None:
            builder.add_times(n_times, times, sfreq)

        # Set aggregation info
        if channel_method:
            builder.set_channel_aggregation(channel_method)
        if trial_method:
            builder.set_trial_aggregation(trial_method)
        if connectivity_method:
            builder.set_connectivity_aggregation(connectivity_method)
        if time_method:
            builder.set_time_aggregation(time_method)

        # Add parameters
        if params:
            builder.add_parameters(params)

        return builder.build()

    def wrap_output(
        self,
        key: str,
        data: np.ndarray,
        schema: Optional[TensorSchema] = None,
        col_names: Optional[List[str]] = None,
        row_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Wrap marker output with schema metadata.

        Parameters
        ----------
        key : str
            Output feature key (e.g., 'spectralpower', 'permutationentropy').
        data : np.ndarray
            The computed data array.
        schema : TensorSchema, optional
            The tensor schema (if None, reader will infer).
        col_names : list of str, optional
            Column names (typically channel names).
        row_names : list of str, optional
            Row names (for matrix outputs).

        Returns
        -------
        dict
            Output dictionary ready for storage.
        """
        output = {
            key: {
                "data": data,
            }
        }

        if col_names is not None:
            output[key]["col_names"] = col_names

        if row_names is not None:
            output[key]["row_names"] = row_names

        if schema is not None:
            output[key]["tensor_schema"] = schema.to_dict()

        return output

    def build_spectral_schema(
        self,
        n_epochs: int,
        n_channels: int,
        channel_names: Optional[List[str]] = None,
        band_names: Optional[List[str]] = None,
        band_ranges: Optional[Dict[str, tuple]] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        **params,
    ) -> TensorSchema:
        """Convenience method to build spectral marker schema.

        Parameters
        ----------
        n_epochs : int
            Number of epochs.
        n_channels : int
            Number of channels.
        channel_names : list of str, optional
            Channel names.
        band_names : list of str, optional
            Band names for multi-band output.
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
        return self.build_schema(
            marker_type="spectral",
            n_epochs=n_epochs if trial_method is None else None,
            n_channels=n_channels if channel_method is None else None,
            channel_names=channel_names,
            band_names=band_names,
            band_ranges=band_ranges,
            channel_method=channel_method,
            trial_method=trial_method,
            **params,
        )

    def build_connectivity_schema(
        self,
        n_epochs: int,
        n_channels: int,
        channel_names: Optional[List[str]] = None,
        flattened: bool = True,
        pair_names: Optional[List[str]] = None,
        connectivity_method: Optional[str] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        **params,
    ) -> TensorSchema:
        """Convenience method to build connectivity marker schema.

        Parameters
        ----------
        n_epochs : int
            Number of epochs.
        n_channels : int
            Number of channels.
        channel_names : list of str, optional
            Channel names.
        flattened : bool
            Whether connectivity is flattened to pairs.
        pair_names : list of str, optional
            Pair names for flattened output.
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
        """
        n_pairs = n_channels * (n_channels - 1) // 2 if flattened else None

        return self.build_schema(
            marker_type="connectivity",
            n_epochs=n_epochs if trial_method is None else None,
            n_channels=n_channels
            if not flattened and channel_method is None
            else None,
            channel_names=channel_names,
            n_pairs=n_pairs if connectivity_method is None else None,
            pair_names=pair_names,
            connectivity_method=connectivity_method,
            channel_method=channel_method,
            trial_method=trial_method,
            **params,
        )

    def build_entropy_schema(
        self,
        n_epochs: int,
        n_channels: int,
        channel_names: Optional[List[str]] = None,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        **params,
    ) -> TensorSchema:
        """Convenience method to build entropy/complexity marker schema.

        Parameters
        ----------
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
        return self.build_schema(
            marker_type="entropy",
            n_epochs=n_epochs if trial_method is None else None,
            n_channels=n_channels if channel_method is None else None,
            channel_names=channel_names,
            channel_method=channel_method,
            trial_method=trial_method,
            **params,
        )

    def build_erp_schema(
        self,
        n_epochs: int,
        n_channels: int,
        channel_names: Optional[List[str]] = None,
        n_times: Optional[int] = None,
        times: Optional[List[float]] = None,
        sfreq: Optional[float] = None,
        time_aggregated: bool = True,
        channel_method: Optional[str] = None,
        trial_method: Optional[str] = None,
        **params,
    ) -> TensorSchema:
        """Convenience method to build ERP marker schema.

        Parameters
        ----------
        n_epochs : int
            Number of epochs.
        n_channels : int
            Number of channels.
        channel_names : list of str, optional
            Channel names.
        n_times : int, optional
            Number of time points.
        times : list of float, optional
            Time values.
        sfreq : float, optional
            Sampling frequency.
        time_aggregated : bool
            Whether time is averaged.
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
        return self.build_schema(
            marker_type="erp",
            n_epochs=n_epochs if trial_method is None else None,
            n_channels=n_channels if channel_method is None else None,
            channel_names=channel_names,
            n_times=n_times if not time_aggregated else None,
            times=times,
            sfreq=sfreq,
            time_method="mean" if time_aggregated else None,
            channel_method=channel_method,
            trial_method=trial_method,
            **params,
        )
