# EEGEpoching

`junifer_eeg.preprocessors.EEGEpoching`

Creates epochs from continuous EEG data. Supports two modes: event-based epoching (using triggers or annotations) and fixed-length epoching.

---

## Parameters

| Parameter          | Type                                        | Default      | Description |
|--------------------|---------------------------------------------|--------------|-------------|
| `tmin`             | `float`                                     | `-0.2`       | Start time of each epoch relative to the event (seconds). Only used in event-based mode. Ignored when `duration` is set. |
| `tmax`             | `float`                                     | `1.0`        | End time of each epoch relative to the event (seconds). Only used in event-based mode. Ignored when `duration` is set. |
| `duration`         | `float` or `None`                           | `None`       | If set, creates fixed-length epochs of this duration (seconds) using `mne.make_fixed_length_epochs`. When set, both `tmin` and `tmax` are ignored. |
| `overlap`          | `float`                                     | `0.0`        | Overlap between consecutive fixed-length epochs (seconds). Only used when `duration` is set. |
| `baseline`         | `tuple (float or None, float)` or `None`    | `(None, 0)`  | Baseline correction interval. `(None, 0)` corrects using the interval from the start of the epoch to `t=0`. |
| `event_id`         | `dict` or `None`                            | `None`       | Mapping of event names to trigger codes, e.g. `{"target": 1, "standard": 2}`. If `None`, all events found in the data are used. |
| `exclude_channels` | `list of str` or `None`                     | `None`       | Additional channel names to exclude from the resulting epochs, beyond non-EEG channels which are always excluded. |
| `dump_path`        | `str` or `None`                             | `None`       | If provided, saves the epoch data to this path. |
| `on`               | `list of str` or `None`                     | `None`       | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.io.BaseRaw` (continuous EEG only)
- **Output:** `mne.BaseEpochs`

---

## Event Detection (event-based mode)

When `duration` is `None`, events are detected as follows:

1. First attempts `mne.find_events` on stim channels.
2. If that fails (no stim channels or no events found), falls back to `mne.events_from_annotations`.
3. If `event_id` is provided, only the matching event types are included. If none match, an error is raised listing the available event names.
4. If `event_id` is `None`, all found events are used.

---

## Channel Selection

After epoching, only EEG channels are kept. Any channels listed in `exclude_channels` are additionally dropped.

---

## Metadata

After processing, `meta["n_epochs_original"]` is set to the number of epochs created (before any subsequent artifact rejection steps).

---

## YAML Examples

### Fixed-length epochs

```yaml
preprocess:
  - kind: EEGEpoching
    duration: 2.0
    overlap: 0.5
    baseline: [null, 0]
```

### Event-based, select specific events

```yaml
preprocess:
  - kind: EEGEpoching
    tmin: -0.2
    tmax: 0.8
    baseline: [null, 0]
    event_id:
      target: 1
      standard: 2
```

### Event-based, all events

```yaml
preprocess:
  - kind: EEGEpoching
    tmin: -0.5
    tmax: 1.5
    baseline: [null, 0]
```

---

## See Also

- [EEGFilter](eeg_filter.md) — applied before epoching on continuous data.
- [EEGReference](eeg_reference.md) — applied before epoching on continuous data.
- [BadChannelsThreshold](bad_channels_threshold.md) — applied after epoching.
- [BadEpochsThreshold](bad_epochs_threshold.md) — applied after epoching.
