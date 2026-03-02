# EEG Reader

`junifer_eeg.reader`

The reader module provides tools for loading and inspecting the **HDF5 output files** produced by a junifer-eeg pipeline. It is not involved in reading raw EEG files during data grabbing — that is handled by the DataGrabber.

---

## `read_h5`

```python
junifer_eeg.reader.read_h5(filepath)
```

Open an HDF5 output file for reading.

### Parameters

| Parameter  | Type            | Description |
|------------|-----------------|-------------|
| `filepath` | `str` or `Path` | Path to the HDF5 file to open. |

### Returns

`SimplifiedH5Reader` — reader object with methods to list and retrieve markers.

### Raises

`FileNotFoundError` — if the file does not exist.

### Example

```python
from junifer_eeg.reader import read_h5

reader = read_h5("output/sub-01_ses-01_task-rest_markers.h5")
print(reader.list_markers())
# ['psd_alpha', 'psd_beta', 'pe_theta']

data = reader.get("psd_alpha")
print(data)
# MarkerData('psd_alpha': bands(5) x epochs(100) x channels(64))
```

---

## `SimplifiedH5Reader`

`junifer_eeg.reader.SimplifiedH5Reader`

Thin wrapper around junifer's `HDF5FeatureStorage` that adds dimension-name mapping based on the stored marker class.

### Constructor

```python
SimplifiedH5Reader(filepath)
```

| Parameter  | Type            | Description |
|------------|-----------------|-------------|
| `filepath` | `str` or `Path` | Path to the HDF5 file. |

### Methods

#### `list_markers()`

Returns the names of all markers stored in the file.

```python
reader.list_markers()
# ['cnv', 'psd_alpha', 'pe_theta', ...]
```

**Returns:** `list of str`

---

#### `get(marker_name)`

Load a single marker by name.

| Parameter     | Type  | Description |
|---------------|-------|-------------|
| `marker_name` | `str` | Name of the marker as stored (e.g., `"psd_alpha"`). |

**Returns:** `MarkerData`

```python
data = reader.get("psd_alpha")
print(data.shape)      # (5, 100, 64)
print(data.dim_names)  # ['bands', 'epochs', 'channels']
print(data.col_names)  # channel names or band names
```

---

#### `dump_all_to_pkl(filepath)`

Save all markers from the HDF5 file into a single pickle file.

| Parameter  | Type            | Description |
|------------|-----------------|-------------|
| `filepath` | `str` or `Path` | Destination path for the pickle file. |

The pickle file contains a dictionary where:
- Keys are marker names (strings)
- Values are dictionaries with:

| Key          | Type           | Description |
|--------------|----------------|-------------|
| `"name"`     | `str`          | Marker name |
| `"data"`     | `np.ndarray`   | Data array  |
| `"dim_names"`| `list of str`  | Dimension labels |
| `"col_names"`| `list of str`  | Column/channel labels |
| `"metadata"` | `dict`         | Marker class, file path, storage kind |
| `"shape"`    | `tuple`        | Array shape |

```python
reader.dump_all_to_pkl("all_markers.pkl")

import pickle
with open("all_markers.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())                # marker names
print(data["psd_alpha"]["shape"]) # (5, 100, 64)
```

---

## `MarkerData`

`junifer_eeg.reader.MarkerData`

Container returned by `SimplifiedH5Reader.get()`.

### Attributes

| Attribute    | Type           | Description |
|--------------|----------------|-------------|
| `name`       | `str`          | Marker name |
| `data`       | `np.ndarray`   | Data array |
| `dim_names`  | `list of str`  | Names for each dimension (e.g., `["bands", "epochs", "channels"]`) |
| `col_names`  | `list of str`  | Column labels (e.g., channel names or frequency band names) |
| `metadata`   | `dict`         | `marker_class`, `filepath`, `kind` |
| `shape`      | `tuple`        | Shortcut for `data.shape` |
| `ndim`       | `int`          | Shortcut for `data.ndim` |

### `dump_to_pkl(filepath)`

Save this single marker to a pickle file.

| Parameter  | Type            | Description |
|------------|-----------------|-------------|
| `filepath` | `str` or `Path` | Destination path. |

---

## Marker Dimension Reference

The reader maps each marker class to its expected dimension order:

| Marker class                    | Dimensions                                |
|---------------------------------|-------------------------------------------|
| `SpectralPowerBands`            | `bands × epochs × channels`               |
| `PermutationEntropy`            | `taus × epochs × channels`                |
| `KolmogorovComplexity`          | `epochs × channels`                       |
| `TimeLockedTopography`          | `epochs × channels`                       |
| `TimeLockedContrast`            | `epochs × channels`                       |
| `ContingentNegativeVariation`   | `epochs × channels`                       |
| `PowerSpectralDensitySummary`   | `epochs × channels`                       |
| `SlowWavesDetection`            | `epochs × channels`                       |
| `SpindlesDetection`             | `epochs × channels`                       |
| `SymbolicMutualInformation`     | `taus × epochs × channels_i × channels_j` |
| `SymbolicMutualInformationROIs` | `taus × epochs × channels_i × channels_j` |
| `PowerSpectralDensityEstimator` | `channels × frequencies`                  |
| `WindowDecoding`                | scalar                                    |
| `TimeDecoding`                  | `times`                                   |

---

## See Also

- [BIDSFeatureStorage](../storage/bids_storage.md) — how results are written as HDF5 files.
