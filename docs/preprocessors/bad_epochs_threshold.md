# BadEpochsThreshold

`junifer_eeg.preprocessors.BadEpochsThreshold`

Detects bad epochs by checking how many channels exceed a peak-to-peak amplitude threshold, then optionally drops them.

---

## Parameters

| Parameter              | Type               | Default            | Description |
|------------------------|--------------------|--------------------|-------------|
| `reject`               | `dict` or `None`   | `{"eeg": 100e-6}`  | Rejection threshold per channel type (in Volts). An epoch is flagged if more than `n_channels_bad_epoch` fraction of channels exceed this peak-to-peak amplitude. |
| `n_channels_bad_epoch` | `float`            | `0.1`              | Fraction of channels (0–1) that must exceed the threshold for an epoch to be considered bad. |
| `min_events`           | `float`            | `0.3`              | Minimum fraction of epochs that must survive rejection. An error is raised if too many epochs are dropped. |
| `drop_bad_epochs`      | `bool`             | `True`             | If `True`, drops the detected bad epochs from the data. |
| `on`                   | `list of str` or `None` | `None`        | Data types to apply this step to. Defaults to all valid inputs (`["EEG"]`). |

---

## Input / Output

- **Input:** `mne.BaseEpochs`
- **Output:** `mne.BaseEpochs` with bad epochs removed if `drop_bad_epochs=True`

---

## Notes

- Already-marked bad channels in `info["bads"]` are excluded from the channel count when assessing each epoch.
- The indices of detected bad epochs are stored in the output metadata even if `drop_bad_epochs=False`.

---

## Metadata Stored

After processing, the following is written to `epochs.info["temp"]["preprocessing_info"]`:

| Key | Description |
|-----|-------------|
| `bad_epochs_detected` | List of indices of the detected bad epochs |
| `n_epochs_before_rejection` | Epoch count before dropping |
| `n_epochs_after_rejection` | Epoch count after dropping |

---

## YAML Example

```yaml
preprocess:
  - kind: BadEpochsThreshold
    reject:
      eeg: 100e-6
    n_channels_bad_epoch: 0.1
    min_events: 0.3
    drop_bad_epochs: true
```

---

## See Also

- [BadChannelsThreshold](bad_channels_threshold.md) — detects bad channels by amplitude threshold.
- [BadChannelsVariance](bad_channels_variance.md) — detects bad channels by variance z-scoring.
- [BadChannelsHighFrequency](bad_channels_high_frequency.md) — detects bad channels by high-frequency variance.
- [EEGEpoching](eeg_epoching.md) — creates epochs that this step operates on.
