# EEGInterpolation

`junifer_eeg.preprocessors.EEGInterpolation`

Interpolates channels previously marked as bad in `inst.info["bads"]`.

---

## Parameters

| Parameter    | Type                          | Default             | Description |
|--------------|-------------------------------|---------------------|-------------|
| `method`     | `dict` or `None`              | `{"eeg": "spline"}` | Interpolation method per channel type. Passed directly to MNE's `interpolate_bads`. |
| `reset_bads` | `bool`                        | `True`              | If `True`, clears `info["bads"]` after interpolation. |
| `origin`     | `str` or array-like (3,)      | `"auto"`            | Origin of the sphere used for spherical spline interpolation. Passed to MNE's `interpolate_bads`. |
| `dump_path`  | `str` or `None`               | `None`              | If provided, saves the result to this path. |
| `on`         | `list of str` or `None`       | `None`              | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.io.BaseRaw` or `mne.BaseEpochs`
- **Output:** same type as input, on a copy with bad channels interpolated

---

## Behaviour

Before interpolating, any EEG channel with an **invalid electrode position** (NaN, Inf, or all-zero coordinates) is automatically **dropped** from the data. This prevents MNE's interpolation matrix from failing on unlocalized channels such as `"Vertex Reference"`. A warning is logged listing any dropped channels.

If `info["bads"]` is empty, no interpolation is performed and the data is returned unchanged.

---

## YAML Example

```yaml
preprocess:
  - kind: EEGInterpolation
    method:
      eeg: spline
    reset_bads: true
    origin: auto
```

---

## See Also

- [BadChannelsThreshold](bad_channels_threshold.md) — marks bad channels by amplitude threshold.
- [BadChannelsVariance](bad_channels_variance.md) — marks bad channels by variance z-scoring.
- [BadChannelsHighFrequency](bad_channels_high_frequency.md) — marks bad channels by high-frequency variance z-scoring.
