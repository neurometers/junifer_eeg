# BadChannelsHighFrequency

`junifer_eeg.preprocessors.BadChannelsHighFrequency`

Detects bad channels using iterative z-score outlier detection over each channel's **high-frequency variance** (a proxy for muscle artifacts or poor electrode contact), then optionally interpolates them.

---

## Parameters

| Parameter       | Type                    | Default  | Description |
|-----------------|-------------------------|----------|-------------|
| `zscore_thresh` | `float`                 | `4`      | Z-score threshold above which a channel's high-frequency variance is considered an outlier. |
| `max_iter`      | `int`                   | `2`      | Maximum number of z-scoring iterations. |
| `min_channels`  | `float`                 | `0.7`    | Minimum fraction of good channels required. An error is raised if too many channels are flagged. |
| `interpolate`   | `bool`                  | `True`   | If `True`, automatically interpolates all detected bad channels. |
| `dump_path`     | `str` or `None`         | `None`   | If provided, saves the preprocessed data to this path after processing. Supports the `{element}` placeholder. |
| `on`            | `list of str` or `None` | `None`   | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.BaseEpochs`
- **Output:** `mne.BaseEpochs` with bad channels marked in `info["bads"]` and, if `interpolate=True`, interpolated (bads cleared)

---

## Notes

- Already-marked bad channels in `info["bads"]` are excluded from the high-frequency variance computation.
- New bad channels are appended to `info["bads"]` before interpolation.
- If `interpolate=True`, MNE's `interpolate_bads(reset_bads=True)` is called, clearing `info["bads"]` afterwards.
- The list of bad channels detected **before** interpolation is preserved in the output metadata.

---

## Metadata Stored

After processing, the following is written to `output["meta"]["preprocessing_info"]` and also to `epochs.info["temp"]["preprocessing_info"]`:

| Key | Description |
|-----|-------------|
| `bad_channels_detected` | List of detected bad channel names (before interpolation) |
| `n_epochs_before_rejection` | Epoch count at the start of this step |
| `n_epochs_after_rejection` | Epoch count at the end of this step |
| `n_channels_interpolated` | Number of channels that were interpolated |

---

## YAML Example

```yaml
preprocess:
  - kind: BadChannelsHighFrequency
    zscore_thresh: 4
    max_iter: 2
    min_channels: 0.7
    interpolate: true
```

---

## See Also

- [BadChannelsThreshold](bad_channels_threshold.md) — detects bad channels by amplitude threshold.
- [BadChannelsVariance](bad_channels_variance.md) — detects bad channels by variance z-scoring.
- [EEGInterpolation](eeg_interpolation.md) — standalone interpolation step.
- [BadEpochsThreshold](bad_epochs_threshold.md) — detects and drops bad epochs.
