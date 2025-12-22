# Junifer HDF5 Reader - Dimension-Aware Reader

A modular, dimension-aware reader for junifer HDF5 output files with full support for all marker types and aggregation configurations.

## Features

- **Dimension-aware**: Understands tensor dimensions (epochs, channels, bands, etc.)
- **Schema support**: Markers can store explicit dimension metadata
- **Schema inference**: Automatic inference when metadata is not available
- **Easy aggregation**: Aggregate data along named dimensions
- **Modular design**: Separate modules for different concerns

## Quick Start

### Reading Data

```python
from junifer_eeg.reader import read_h5, DimensionType

# Open HDF5 file
reader = read_h5("output.h5")

# List available markers
print(reader.list_markers())

# Get marker data with schema
data = reader.get("psd_alpha")

# View schema description
print(data.schema.describe())
# Output:
# TensorSchema: psd_alpha (spectral)
#   Shape: (100, 64)
#   Dimensions:
#     [0] EPOCHS: size=100
#     [1] CHANNELS: size=64, labels=64
#   Aggregation: None

# Access raw numpy array
arr = data.data

# Check dimension types
if data.has_dimension(DimensionType.CHANNELS):
    print(f"Has {data.shape[data.get_axis(DimensionType.CHANNELS)]} channels")
```

### Aggregating Data

```python
from junifer_eeg.reader import read_h5, DimensionType

reader = read_h5("output.h5")
data = reader.get("psd_alpha")

# Aggregate across channels
channel_avg = data.aggregate("mean", DimensionType.CHANNELS)
print(channel_avg.shape)  # Now (100,) - epochs only

# Aggregate across epochs  
epoch_avg = data.aggregate("trim_mean80", DimensionType.EPOCHS)
print(epoch_avg.shape)  # Now (64,) - channels only

# Chain aggregations
fully_agg = (data
    .aggregate("mean", DimensionType.CHANNELS)
    .aggregate("mean", DimensionType.EPOCHS))
print(fully_agg.data)  # Single scalar value
```

### Selecting Data

```python
# Select specific band (for multi-band markers)
alpha_data = reader.get("psd_bands").select(DimensionType.BANDS, "alpha")

# Select specific channels
frontal = data.select(DimensionType.CHANNELS, ["E1", "E2", "E3"])

# Select epoch range
first_50 = data.select(DimensionType.EPOCHS, slice(0, 50))
```

### Converting to DataFrame

```python
# Convert to pandas DataFrame
df = data.to_dataframe()

# Rows = epochs, Columns = channels
print(df.head())
```

## Module Overview

### `dimension_schema.py`
Defines dimension types and tensor schemas:
- `DimensionType`: Enum of dimension types (EPOCHS, CHANNELS, BANDS, etc.)
- `DimensionInfo`: Information about a single dimension
- `TensorSchema`: Complete schema for a marker output tensor
- `AggregationInfo`: Information about applied aggregations

### `marker_registry.py`
Marker type detection and schema inference:
- `MarkerType`: Enum of known marker types
- `MarkerRegistry`: Registry for type detection and inference
- `detect_marker_type()`: Detect marker type from name
- `infer_schema()`: Infer schema from data and name

### `aggregators.py`
Data aggregation utilities:
- `aggregate()`: Main aggregation function
- `AGGREGATION_FUNCTIONS`: Dict of available methods
- Helper functions for specific dimensions
- Connectivity matrix utilities

### `reader.py`
Main reader classes:
- `JuniferH5Reader`: Main reader class
- `MarkerData`: Container for loaded data with schema
- `read_h5()`: Convenience function

### `schema_builder.py`
Utilities for markers to build schemas:
- `SchemaBuilder`: Fluent builder class
- Helper functions for common marker types
- `schema_to_storage_kwargs()`: Convert schema for HDF5

### `marker_mixin.py`
Mixin class for markers:
- `SchemaAwareMixin`: Add to markers for schema support

## For Marker Developers

### Adding Schema Support to a Marker

```python
from junifer.markers import BaseMarker
from junifer_eeg.reader import SchemaAwareMixin

class MyMarker(BaseMarker, SchemaAwareMixin):
    def compute(self, input, extra_input=None):
        # ... compute data ...
        n_epochs = data.shape[0]
        n_channels = data.shape[1]
        
        # Build schema
        schema = self.build_spectral_schema(
            n_epochs=n_epochs,
            n_channels=n_channels,
            channel_names=ch_names,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        
        # Return with schema
        return self.wrap_output(
            key="myfeature",
            data=result_data,
            schema=schema,
            col_names=ch_names,
        )
```

### Using SchemaBuilder Directly

```python
from junifer_eeg.reader import SchemaBuilder

# Build schema with fluent interface
schema = (SchemaBuilder("spectral", "psd_alpha")
    .add_bands(["delta", "theta", "alpha", "beta", "gamma"])
    .add_epochs(100)
    .add_channels(64, channel_names)
    .set_channel_aggregation("mean")
    .add_parameter("fmin", 1.0)
    .add_parameter("fmax", 45.0)
    .build())

# Include in marker output
output = {
    "spectralpower": {
        "data": result_array,
        "col_names": channel_names,
        "tensor_schema": schema.to_dict(),  # This gets stored in HDF5
    }
}
```

## Supported Aggregation Methods

- `mean`, `median`, `std`, `var`
- `min`, `max`, `sum`, `range`
- `nanmean`, `nanmedian`, `nanstd` (NaN-safe)
- `trim_mean80`, `trim_mean90` (trimmed means)
- `percentile_25`, `percentile_75`, `iqr`
- `count`

## Dimension Types

| Type | Description | Example Use |
|------|-------------|-------------|
| `EPOCHS` | Trial/epoch dimension | Spectral, entropy markers |
| `CHANNELS` | EEG channels | All markers |
| `TIMES` | Time points | ERP markers |
| `BANDS` | Frequency bands | Multi-band spectral |
| `CHANNEL_PAIRS` | Flattened connectivity | WSMI |
| `CHANNELS_I`, `CHANNELS_J` | Matrix connectivity | Full connectivity matrix |
| `FREQUENCIES` | Raw frequency bins | Raw PSD |
| `FEATURES` | Generic features | Sleep markers |

