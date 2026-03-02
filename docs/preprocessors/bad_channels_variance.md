# BadChannelsVariance

`junifer_eeg.preprocessors.BadChannelsVariance`

Detects bad channels using iterative z-score outlier detection over channel variance, then optionally interpolates them.

---

## Parameters

| Parameter       | Type                    | Default  | Description |
|-----------------|-------------------------|----------|-------------|
| `zscore_thresh` | `float`                 | `4`      | Z-score threshold above which a channel's variance is considered an outlier. |
| `max_iter`      | `int`                   | `2`      | Maximum number of z-scoring iterations. Each iteration removes the most extreme outlier and re-computes z-scores. |
| `min_channels`  | `float`                 | `0.7`    | Minimum fraction of good channels required. An error is raised if too many channels are flagged. |
| `interpolate`   | `bool`                  | `True`   | If `True`, automatically interpolates all detected bad channels. |
| `on`            | `list of str` or `None` | `None`   | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.BaseEpochs`
- **Output:** `mne.BaseEpochs` with bad channels marked in `info["bads"]` and, if `interpolate=True`, interpolated (bads cleared)

---

## Notes

- Already-marked bad channels in `info["bads"]` are excluded from the variance computation.
- New bad channels are appended to `info["bads"]` before interpolation.
- If `interpolate=True`, MNE's `interpolate_bads(reset_bads=True)` is called, which clears `info["bads"]` afterwards.

---

## YAML Example

```yaml
preprocess:
  - kind: BadChannelsVariance
    zscore_thresh: 4
    max_iter: 2
    min_channels: 0.7
    interpolate: true
```

---

## See Also

- [BadChannelsThreshold](bad_channels_threshold.md) — detects bad channels by amplitude threshold.
- [BadChannelsHighFrequency](bad_channels_high_frequency.md) — detects bad channels by high-frequency variance.
- [EEGInterpolation](eeg_interpolation.md) — standalone interpolation step.
- [BadEpochsThreshold](bad_epochs_threshold.md) — detects and drops bad epochs.
