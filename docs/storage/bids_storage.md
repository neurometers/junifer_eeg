# BIDSFeatureStorage

`junifer_eeg.storage.BIDSFeatureStorage`

A BIDS-compliant feature storage that wraps junifer's [`HDF5FeatureStorage`](https://juaml.github.io/junifer/main/api/storage.html#junifer.storage.HDF5FeatureStorage). Instead of writing all results to a single flat file, it automatically builds a proper BIDS directory tree and generates filenames from the element's BIDS entities (`sub`, `ses`, `task`, `acq`, `run`).

---

## Output Structure

For each element processed, `BIDSFeatureStorage` creates a file at:

```
<base_dir>/sub-<subject>/[ses-<session>/]eeg/
    sub-<subject>[_ses-<session>][_task-<task>][_acq-<acq>][_run-<run>]_desc-<desc>_<suffix>.<ext>
```

**Example** — element `{"subject": "01", "session": "01", "task": "lg", "acq": "01"}` with `uri="output/wsmi_features.h5"`:

```
output/
└── sub-01/
    └── ses-01/
        └── eeg/
            ├── sub-01_ses-01_task-lg_acq-01_desc-wsmi_features_markers.h5
            └── sub-01_ses-01_task-lg_acq-01_desc-wsmi_features_markers.pkl  # if output_pickle=True
```

---

## Parameters

| Parameter                | Type                  | Default       | Description |
|--------------------------|-----------------------|---------------|-------------|
| `uri`                    | `str` or `Path`       | **required**  | Base path for output. If a filename with extension is provided (e.g., `output/wsmi_features.h5`), the stem (`wsmi_features`) becomes the `desc-` field and the parent (`output/`) becomes the output directory. If a directory is given (no extension), `desc-markers` is used. |
| `suffix`                 | `str`                 | `"markers"`   | BIDS suffix appended before the extension (e.g., `"markers"`, `"features"`, `"connectivity"`). |
| `single_output`          | `bool`                | `False`       | If `True`, all elements are written to the same file. If `False` (recommended for BIDS), one file per element with BIDS naming. |
| `output_pickle`          | `bool`                | `False`       | If `True`, automatically converts each HDF5 file to a pickle (`.pkl`) file with the same BIDS filename. |
| `delete_h5_after_pickle` | `bool`                | `False`       | If `True` and `output_pickle=True`, deletes the original `.h5` file after pickle conversion. |
| `overwrite`              | `bool` or `"update"`  | `"update"`    | `True` overwrites existing files. `"update"` appends/updates existing entries. |
| `compression`            | `int` (0–9)           | `7`           | gzip compression level for HDF5. `0` = no compression, `9` = maximum. |
| `force_float32`          | `bool`                | `True`        | Cast `float64` arrays to `float32` before writing to reduce file size. |
| `chunk_size`             | `int`                 | `100`         | Number of element files processed per batch during `collect()`. |

---

## `desc-` Field Derivation

The `desc-<label>` BIDS entity is taken from the **stem** of the `uri` filename:

| `uri` value                          | `desc-` label       |
|--------------------------------------|---------------------|
| `output/wsmi_features.h5`            | `wsmi_features`     |
| `derivatives/junifer-eeg/psd.h5`     | `psd`               |
| `output/` *(directory, no extension)*| `markers`           |

---

## BIDS Entity Extraction

BIDS entities are extracted from the element dictionary in the following priority order:

1. **Direct keys** in the element dict — `subject`, `session`, `task`, `acq`/`acquisition`, `run`.
2. **File path parsing** — if an entity is missing from the dict but a file path in the element contains BIDS-style tokens (e.g., `_ses-01_`, `_task-rest_`), those are parsed from the filename.

BIDS prefix stripping is applied automatically: `"sub-01"` and `"01"` are both normalised to `"01"` in the filename label.

---

## Usage Examples

### YAML configuration — basic

```yaml
storage:
  kind: BIDSFeatureStorage
  uri: output/derivatives/junifer-eeg/wsmi_features.h5
  suffix: markers
  single_output: false
```

### YAML configuration — with pickle output

```yaml
storage:
  kind: BIDSFeatureStorage
  uri: output/derivatives/junifer-eeg/wsmi_features.h5
  suffix: markers
  single_output: false
  output_pickle: true
  delete_h5_after_pickle: false
```

### Python

```python
from junifer_eeg.storage import BIDSFeatureStorage

storage = BIDSFeatureStorage(
    uri="output/derivatives/junifer-eeg/wsmi_features.h5",
    suffix="markers",
    single_output=False,
    output_pickle=True,
)
```

---

## Utility Functions

### `element_to_bids_path`

```python
junifer_eeg.storage.element_to_bids_path(
    element, base_dir, base_name, suffix="markers", extension=".h5"
)
```

Converts an element dict to a full BIDS-compliant `Path` (directory + filename). Used internally by `BIDSFeatureStorage`.

| Parameter   | Type            | Default      | Description |
|-------------|-----------------|--------------|-------------|
| `element`   | `dict`          | **required** | Element dictionary with BIDS entities. |
| `base_dir`  | `Path`          | **required** | Root output directory. |
| `base_name` | `str`           | **required** | Used as the `desc-<label>` field. |
| `suffix`    | `str`           | `"markers"`  | BIDS suffix (before the extension). |
| `extension` | `str`           | `".h5"`      | File extension including the dot. |

**Returns:** `pathlib.Path`

**Example:**

```python
from pathlib import Path
from junifer_eeg.storage.bids_storage import element_to_bids_path

element = {"subject": "sub-01", "session": "01", "task": "lg", "acq": "01"}
path = element_to_bids_path(element, Path("output"), "wsmi", "markers", ".h5")
# output/sub-01/ses-01/eeg/sub-01_ses-01_task-lg_acq-01_desc-wsmi_markers.h5
```

---

### `convert_h5_to_bids_pkl`

```python
junifer_eeg.storage.convert_h5_to_bids_pkl(
    h5_path, output_dir=None, suffix="markers", delete_h5=False
)
```

Standalone utility to convert an existing HDF5 output file to a pickle file. Useful for post-hoc conversion without re-running the pipeline.

| Parameter    | Type                   | Default       | Description |
|--------------|------------------------|---------------|-------------|
| `h5_path`    | `str` or `Path`        | **required**  | Path to the source HDF5 file. |
| `output_dir` | `str`, `Path`, or `None` | `None`      | Output directory. If `None`, uses the same directory as `h5_path`. |
| `suffix`     | `str`                  | `"markers"`   | BIDS suffix (informational only in this function). |
| `delete_h5`  | `bool`                 | `False`       | Delete the HDF5 file after successful pickle conversion. |

**Returns:** `pathlib.Path` — path to the created pickle file.

**Raises:** `FileNotFoundError` — if `h5_path` does not exist.

**Example:**

```python
from junifer_eeg.storage.bids_storage import convert_h5_to_bids_pkl

pkl_path = convert_h5_to_bids_pkl(
    "output/sub-01_ses-01_task-lg_markers.h5",
    delete_h5=False,
)
# output/sub-01_ses-01_task-lg_markers.pkl
```

---

## Inheritance

```
BaseFeatureStorage
└── HDF5FeatureStorage
    └── BIDSFeatureStorage
```

`BIDSFeatureStorage` inherits all storage methods from `HDF5FeatureStorage`:

| Method                | Description |
|-----------------------|-------------|
| `store_timeseries()`  | Store 2D timeseries data. |
| `store_timeseries_2d()` | Store 3D timeseries data. |
| `store_vector()`      | Store 1D vector data. |
| `store_matrix()`      | Store matrix data (full, upper/lower triangular). |
| `store_scalar_table()`| Store table of scalar values. |
| `store_metadata()`    | Store feature metadata. |
| `list_features()`     | List all stored features with metadata. |
| `read()`              | Read a stored feature by name or MD5. |
| `read_df()`           | Read a stored feature as a `pandas.DataFrame`. |
| `collect()`           | Merge per-element files into a single output (when `single_output=False`). |

---

## Pickle File Format

Pickle files created by `output_pickle=True` (or `convert_h5_to_bids_pkl`) have the following structure (written by `SimplifiedH5Reader.dump_all_to_pkl`):

```python
{
    "marker_name_1": {
        "name": "marker_name_1",
        "data": np.ndarray,        # shape depends on marker type
        "dim_names": ["dim0", ...], # e.g., ["bands", "epochs", "channels"]
        "col_names": ["ch1", ...],  # column labels
        "metadata": {
            "marker_class": "SpectralPowerBands",
            "filepath": "/path/to/file.h5",
            "kind": "timeseries",
        },
        "shape": (5, 100, 64),
    },
    "marker_name_2": { ... },
}
```

---

## Notes

- For BIDS compliance, always use `single_output: false`.
- Place outputs inside a `derivatives/` subdirectory of your BIDS dataset (e.g., `derivatives/junifer-eeg/`).
- The `delete_h5_after_pickle` flag is permanent — deleted HDF5 files cannot be recovered. Use with caution.
- Pickle conversion happens automatically at object destruction time (`__del__`). If the pipeline crashes mid-run, conversion may be incomplete. Use `convert_h5_to_bids_pkl` manually in that case.

---

## See Also

- [EEGDataGrabber](../datagrabbers/eeg_datagrabber.md) — how data is located and grabbed.
- [EEG Reader](../reader/reader.md) — how to read the stored HDF5/pickle outputs.
- [`HDF5FeatureStorage`](https://juaml.github.io/junifer/main/api/storage.html#junifer.storage.HDF5FeatureStorage) — underlying storage implementation.
