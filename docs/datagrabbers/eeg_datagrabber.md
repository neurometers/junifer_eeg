# EEGDataGrabber

`junifer_eeg.datagrabber.EEGDataGrabber`

A concrete DataGrabber for EEG datasets stored in any directory structure. It extends junifer's [`PatternDataGrabber`](https://juaml.github.io/junifer/main/api/datagrabbers.html#junifer.datagrabber.PatternDataGrabber) and registers the `EEG` data type, with automatic equipment/montage detection on load.

---

## Supported File Formats

The grabber (via the reader registry) handles the following EEG file extensions automatically:

| Extension     | Format              | Notes                              |
|---------------|---------------------|------------------------------------|
| `.edf`        | EDF                 | European Data Format               |
| `.bdf`        | EDF (BioSemi)       | BioSemi variant of EDF             |
| `.gdf`        | GDF                 | General Data Format (EDF successor)|
| `.fif`        | FIF                 | MNE native (raw or epochs)         |
| `.raw`        | RAW FIF             | Standalone raw FIF files           |
| `.vhdr`       | BrainVision         | Header file (`.eeg`/`.vmrk` also needed) |
| `.eeg`        | Nihon Kohden        | Clinical EEG format                |
| `.mff`        | EGI MFF             | Directory-based format             |
| `.set`        | EEGLAB              | With optional `.fdt` data file     |
| `.cnt`        | Neuroscan           | Continuous EEG                     |
| `.sqd` / `.con` | KIT/Yokogawa      | MEG/EEG hybrid system              |
| `.cdt` / `.dap` | Curry             | Curry 7/8 data files               |
| `.data`       | Nicolet             | Clinical EEG                       |
| `.mat`        | FieldTrip           | MATLAB format (epoched data)       |

---

## Parameters

| Parameter      | Type                    | Default            | Description |
|----------------|-------------------------|--------------------|-------------|
| `datadir`      | `str` or `Path`         | **required**       | Path to the root directory containing EEG data. |
| `types`        | `list of str` or `None` | `["EEG"]`          | Data types to grab. Currently only `"EEG"` is supported by this grabber. |
| `patterns`     | `dict` or `None`        | `{"EEG": {"pattern": "{subject}/*.fif", "space": "native"}}` | Patterns for each data type. Each entry must include `"pattern"` and `"space"`. If `None`, uses the default shown. |
| `replacements` | `list of str` or `None` | `["subject"]`      | Variables used as placeholders in the pattern strings. These become the element keys (e.g., `subject`, `session`, `task`). |
| `**kwargs`     | any                     | —                  | Additional keyword arguments forwarded to `PatternDataGrabber`. |

---

## Pattern Schema for `EEG`

When providing a custom `patterns` dictionary, each entry for `"EEG"` must follow this schema:

```yaml
EEG:
  pattern: "{subject}/{session}/eeg/{subject}_{session}_task-{task}_eeg.fif"
  space: "native"
```

**Mandatory keys:**
- `pattern` — glob-style path pattern with `{replacement}` placeholders.
- `space` — coordinate space of the data (typically `"native"` for EEG).

**Optional keys:**
- `mask` — dictionary with its own `pattern` and `space`.

---

## Usage Examples

### Minimal (default patterns)

```yaml
datagrabber:
  kind: EEGDataGrabber
  datadir: /data/eeg_study
```

This will look for files matching `{subject}/*.fif` under `/data/eeg_study`, where `subject` is inferred from the directory names.

---

### Custom single-level pattern

```yaml
datagrabber:
  kind: EEGDataGrabber
  datadir: /data/eeg_study
  patterns:
    EEG:
      pattern: "{subject}/eeg/{subject}_task-rest_eeg.fif"
      space: native
  replacements:
    - subject
```

---

### BIDS-like multi-level pattern (subject + session + task)

```yaml
datagrabber:
  kind: EEGDataGrabber
  datadir: /data/bids_dataset
  patterns:
    EEG:
      pattern: "sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-{task}_eeg.vhdr"
      space: native
  replacements:
    - subject
    - session
    - task
```

Elements returned by `get_elements()` will be tuples of `(subject, session, task)`.

---

### EDF format

```yaml
datagrabber:
  kind: EEGDataGrabber
  datadir: /data/sleep_study
  patterns:
    EEG:
      pattern: "{subject}/{subject}_night{session}.edf"
      space: native
  replacements:
    - subject
    - session
```

---

## Inheritance

```
BaseDataGrabber
└── PatternDataGrabber
    └── EEGDataGrabber
```

`EEGDataGrabber` inherits all methods from `PatternDataGrabber`:

| Method               | Description |
|----------------------|-------------|
| `get_elements()`     | Returns a list of all elements (subjects / tuples) found by matching patterns on disk. |
| `get_element_keys()` | Returns the replacement variable names (e.g., `["subject"]`). |
| `get_item(**element)`| Returns a dict mapping data type → file path for the specified element. |
| `get_types()`        | Returns the list of data types being grabbed. |

---

## Notes

- Equipment detection and montage assignment happen automatically when the file is loaded by the reader (not at grab time).
- For BrainVision format, point `pattern` to the `.vhdr` file; the `.eeg`/`.vmrk` sidecar files are expected to be in the same directory.
- For EGI MFF format, the pattern should match the `.mff` **directory** (not a file inside it).
- `space` is always set to `"native"` for EEG data since no standardised EEG coordinate space exists equivalent to MNI.

---

## See Also

- [`PatternDataGrabber`](https://juaml.github.io/junifer/main/api/datagrabbers.html#junifer.datagrabber.PatternDataGrabber) — parent class documentation.
- [EEG Reader](../reader/reader.md) — how to read HDF5 output files produced by the pipeline.
- [BIDS Feature Storage](../storage/bids_storage.md) — how to store extracted features.
