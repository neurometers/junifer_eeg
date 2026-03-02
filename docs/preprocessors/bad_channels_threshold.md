# BadChannelsThreshold

`junifer_eeg.preprocessors.BadChannelsThreshold`

Detects bad channels by checking how many epochs exceed a peak-to-peak amplitude threshold, then optionally interpolates them.

---

## Parameters

| Parameter          | Type               | Default            | Description |
|--------------------|--------------------|--------------------|-------------|
| `reject`           | `dict` or `None`   | `{"eeg": 100e-6}`  | Rejection threshold per channel type (in Volts). A channel is flagged if its peak-to-peak amplitude (`max - min`) exceeds this value in more than `n_epochs_bad_ch` fraction of epochs. |
| `n_epochs_bad_ch`  | `float`            | `0.5`              | Fraction of epochs (0–1) in which a channel must exceed the threshold to be considered bad. |
| `min_channels`     | `float`            | `0.7`              | Minimum fraction of good channels required after rejection. An error is raised if too many channels are flagged. |
| `interpolate`      | `bool`             | `True`             | If `True`, automatically interpolates all detected bad channels. |
| `on`               | `list of str` or `None` | `None`        | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.BaseEpochs`
- **Output:** `mne.BaseEpochs` with bad channels marked in `info["bads"]` and, if `interpolate=True`, interpolated (bads cleared)

---

## Notes

- Already-marked bad channels in `info["bads"]` are excluded from the threshold check.
- New bad channels are appended to `info["bads"]` before interpolation.
- If `interpolate=True`, MNE's `interpolate_bads(reset_bads=True)` is called, which clears `info["bads"]` afterwards.

---

## YAML Example

```yaml
preprocess:
  - kind: BadChannelsThreshold
    reject:
      eeg: 100e-6
    n_epochs_bad_ch: 0.5
    min_channels: 0.7
    interpolate: true
```

---

## See Also

- [BadChannelsVariance](bad_channels_variance.md) — detects bad channels by variance z-scoring.
- [BadChannelsHighFrequency](bad_channels_high_frequency.md) — detects bad channels by high-frequency variance.
- [EEGInterpolation](eeg_interpolation.md) — standalone interpolation step.
- [BadEpochsThreshold](bad_epochs_threshold.md) — detects and drops bad epochs.
