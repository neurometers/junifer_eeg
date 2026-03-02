# EEGReference

`junifer_eeg.preprocessors.EEGReference`

Applies an EEG reference to continuous raw data or epochs using MNE's `set_eeg_reference`.

---

## Parameters

| Parameter      | Type                              | Default      | Description |
|----------------|-----------------------------------|--------------|-------------|
| `ref_channels` | `str`, `list of str`, or `None`   | `"average"`  | Reference to apply. `"average"` → average reference. `"REST"` → Reference Electrode Standardisation Technique. `list of str` → use specific channel names as reference. `None` → no referencing is applied (pass-through). |
| `projection`   | `bool`                            | `True`       | If `True`, the reference is added as a projection rather than applied directly to the data. Typically used together with `apply_proj=True`. |
| `apply_proj`   | `bool`                            | `True`       | If `True`, applies all projections (including the reference projection) immediately after setting the reference. Only has effect when `ref_channels` is not `None`. |
| `on`           | `list of str` or `None`           | `None`       | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.io.BaseRaw` or `mne.BaseEpochs`
- **Output:** same type as input, re-referenced on a copy

---

## Notes

- All operations are performed on a **copy** of the data; the original is not modified.
- If `ref_channels=None`, the data is returned unchanged (useful as a conditional no-op).
- `projection` and `apply_proj` together control whether the reference is stored as a projection first and then applied, or applied immediately. Setting `projection=True` with `apply_proj=True` is the standard workflow for average reference in MNE.

---

## YAML Example

```yaml
preprocess:
  - kind: EEGReference
    ref_channels: average
    projection: true
    apply_proj: true
```

---

## See Also

- [EEGFilter](eeg_filter.md) — typically applied before referencing.
- [EEGEpoching](eeg_epoching.md) — applied after referencing on continuous data.
