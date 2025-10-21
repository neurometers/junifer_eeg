import math

import mne
import numpy as np
from mne.utils import logger
from scipy.signal import butter, filtfilt

# Equipment-specific filter parameters (from nice_ext/equipments/filters.py)
EQUIPMENT_FILTER_PARAMS = {
    "egi": {
        "lpass": 45.0,
        "hpass": 0.5,
        "notches": [50, 100],
        "resample_freq": 250,
        "filter_method": "iir",
        "hp_order": 6,
        "lp_order": 8,
        "l_trans_bandwidth": 0.1,
    },
    "brainvision": {
        "lpass": 45.0,
        "hpass": 0.5,
        "notches": [50, 100],  # 200Hz added if sfreq > 400
        "filter_method": "iir",
        "hp_order": 4,
        "lp_order": 8,
        "l_trans_bandwidth": 0.1,
    },
    "biosemi": {
        "lpass": 45.0,
        "hpass": 0.5,
        "notches": [50, 100, 200, 400],
        "filter_method": "iir",
        "hp_order": 4,
        "lp_order": 8,
        "l_trans_bandwidth": 0.1,
    },
    "eximia": {
        "lpass": 45.0,
        "hpass": 0.5,
        "notches": [50, 100, 200, 400],
        "filter_method": "iir",
        "hp_order": 4,
        "lp_order": 8,
        "l_trans_bandwidth": 0.1,
    },
    "ant": {
        "lpass": 45.0,
        "hpass": 0.5,
        "notches": [50, 100],
        "filter_method": "iir",
        "hp_order": 4,
        "lp_order": 8,
        "l_trans_bandwidth": 0.1,
    },
}


def _check_min_channels(epochs, bad_channels, min_channels):
    if isinstance(min_channels, float):
        logger.info(
            "Using relative min_channels: {} * {} = {} "
            "channels remaining to reject preprocess".format(
                min_channels,
                epochs.info["nchan"],
                epochs.info["nchan"] * min_channels,
            )
        )
        min_channels = int(epochs.info["nchan"] * min_channels)

    chans_remaining = epochs.info["nchan"] - len(bad_channels)
    if chans_remaining < min_channels:
        msg = (
            "Can not clean data. Only {} out of {} channels remaining.".format(
                chans_remaining, epochs.info["nchan"]
            )
        )
        logger.error(msg)
        raise Exception(msg)


def _check_min_events(epochs, min_events):
    n_orig_epochs = len([x for x in epochs.drop_log if "IGNORED" not in x])
    if isinstance(min_events, float):
        logger.info(
            f"Using relative min_events: {min_events} * {n_orig_epochs} = {int(n_orig_epochs * min_events)} "
            "epochs remaining to reject preprocess"
        )
        min_events = int(n_orig_epochs * min_events)

    epochs_remaining = len(epochs)
    if epochs_remaining < min_events:
        msg = (
            f"Can not clean data. Only {epochs_remaining} out of {n_orig_epochs} epochs "
            "remaining."
        )
        logger.error(msg)
        raise Exception(msg)


def find_bads_channels_variance(inst, picks, zscore_thresh=4, max_iter=2):
    """Find bad channels based on Z-scoring outliers of the channel variances.

    First, the channel variances are calculated using numpy.var along the sample axis.
    Then, the channel variances are passed to Z-scoring outlier detection
    to find bad channels based on Z-scoring over the calculated variance.
    This procedure compares the absolute z-score of the variances against the threshold.

    Parameters
    ----------
    inst : instance of mne.Epochs
        The data.
    picks : list of int
        The indices of the channels to be used.
    zscore_thresh : int
        The threshold for the z-score outliers. Default is 4 std.
    max_iter : int
        The maximum number of iterations of the iterative z-scoring. Default is 2.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.

    """

    logger.info("Looking for bad channels with variance")
    if isinstance(inst, mne.BaseEpochs):  # Use BaseEpochs instead of Epochs
        data = inst.get_data()
    else:
        data = inst._data[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.NaN)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_var = np.ma.hstack(masked_data).var(axis=-1)

    # Use MNE's iterated z-scoring for outlier detection (original NICE approach)
    from mne.preprocessing.bads import _find_outliers

    bad_ch_var = _find_outliers(
        ch_var, threshold=zscore_thresh, max_iter=max_iter
    )
    logger.info(f"Reject by variance: bad_channels: {bad_ch_var}")
    bad_chs = list({inst.ch_names[i] for i in bad_ch_var})
    return bad_chs


def find_bads_channels_high_frequency(
    inst, picks, zscore_thresh=4, max_iter=2
):
    """Find bad channels based on Z-scoring outliers of the channel high frequencies standard deviation.

    First, the channel high frequencies standard deviations are calculated.
    Then, the channel high frequencies standard deviations are passed to
    Z-scoring outlier detection to find bad channels based on Z-scoring
    over the calculated standard deviation of the channels high frequencies.
    This procedure compares the absolute z-score of the standard deviations against the threshold.

    Parameters
    ----------
    inst : instance of mne.Epochs
        The data.
    picks : list of int
        The indices of the channels to be used.
    zscore_thresh : int
        The threshold for the z-score outliers. Default is 4 std.
    max_iter : int
        The maximum number of iterations of the iterative z-scoring. Default is 2.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.
    """
    logger.info("Looking for bad channels with high frequency variance")
    if isinstance(inst, mne.BaseEpochs):  # Use BaseEpochs instead of Epochs
        data = inst.get_data()
    else:
        data = inst._data[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.NaN)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    filter_freq = 25
    b, a = butter(4, 2.0 * filter_freq / inst.info["sfreq"], "highpass")
    filt_data = filtfilt(b, a, np.ma.hstack(masked_data))
    filt_masked_data = np.ma.masked_array(filt_data, fill_value=np.NaN)
    if len(exclude) > 0:
        filt_masked_data[exclude, :] = np.ma.masked
    # Use MNE's iterated z-scoring for outlier detection (original NICE approach)
    from mne.preprocessing.bads import _find_outliers

    hf_std = filt_masked_data.std(axis=-1)
    bad_ch_hf = _find_outliers(
        hf_std, threshold=zscore_thresh, max_iter=max_iter
    )
    logger.info(f"Reject by high frequency std: bad_channels: {bad_ch_hf}")
    bad_chs = list({inst.ch_names[i] for i in bad_ch_hf})
    return bad_chs


def find_bads_epochs_threshold(
    epochs, picks, reject, n_channels_bad_epoch=0.1
):
    """Find bad epochs based on the number of channels where the values range
    is over the reject threshold.

    The range of the values is defined as the difference between the maximum value
    and the minimum value of a single channel in a single epoch. This defines the
    channel range per epoch. After the range is calculated for each epoch, the
    number of channels where the range is over the reject threshold is
    calculated for each epoch. If the number of channels is over the
    n_channels_bad_epoch for an epoch, then that epoch is returned as bad.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    picks : list of int
        The indices of the channels to be used.
    reject : dict
        The rejection threshold for each channel type (EEG, MEG, etc).
    n_channels_bad_epoch : int or float
        The number of channels that have to be over the reject threshold for an
        epoch to be considered bad. Default is 0.1.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.

    """

    n_channels = len(picks)
    bad_ep_idx = np.ndarray((0,), dtype=np.int32)
    if isinstance(n_channels_bad_epoch, float):
        n_channels_bad_epoch = math.floor(n_channels_bad_epoch * n_channels)

    data = epochs.get_data()
    masked_data = np.ma.masked_array(data, fill_value=np.NaN)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_types_inds = mne.channel_indices_by_type(epochs.info)
    n_epochs = masked_data.shape[0]
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        count_bad_chans = np.zeros((n_epochs), dtype=np.int32)
        for i_ep, epoch in enumerate(masked_data[:, idx]):
            deltas = epoch.max(axis=1) - epoch.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_chans[i_ep] = idx_deltas.shape[0]
        reject_bad_epochs = np.where(count_bad_chans > n_channels_bad_epoch)[0]
        logger.info(
            f"Reject by threshold {reject_thresh} on {key.upper()} : bad_epochs: {reject_bad_epochs}"
        )
    bad_epochs = np.unique(np.concatenate((bad_ep_idx, reject_bad_epochs)))

    return bad_epochs


def find_bads_channels_threshold(epochs, picks, reject, n_epochs_bad_ch=0.5):
    """Find bad channels based on the number of epochs where the values range
    is over the reject threshold.

    The range of the values is defined as the difference between the maximum value
    and the minimum value of a single channel in a single epoch. This defines the
    channel range per epoch. After the range is calculated for each epoch, the
    number of epochs where the channel range is over the reject threshold is
    calculated. If the number of epochs is over the n_epochs_bad_ch, then
    the channel is returned as bad.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    picks : list of int
        The indices of the channels to be used.
    reject : dict
        The rejection threshold for each channel type (EEG, MEG, etc).
    n_epochs_bad_ch : int or float
        The minimum number of epochs over the threshold to consider a channel to
        be bad. If float, it is the fraction of epochs. Default is 0.5.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.

    """

    n_channels = len(picks)
    data = epochs.get_data()
    n_epochs = data.shape[0]

    if isinstance(n_epochs_bad_ch, float):
        n_epochs_bad_ch = math.floor(n_epochs_bad_ch * n_epochs)

    ch_types_inds = mne.channel_indices_by_type(epochs.info)
    data = np.transpose(data, (1, 0, 2))
    bad_ch_idx = np.ndarray((0,), dtype=np.int32)
    for key, reject_thresh in reject.items():
        idx = np.array([x for x in ch_types_inds[key] if x in picks])
        count_bad_epochs = np.zeros((n_channels), dtype=np.int32)
        for i_ch, channel in enumerate(data[idx]):
            deltas = channel.max(axis=1) - channel.min(axis=1)
            idx_deltas = np.where(np.greater(deltas, reject_thresh))[0]
            count_bad_epochs[i_ch] = idx_deltas.shape[0]
        reject_bad_channels = np.where(count_bad_epochs > n_epochs_bad_ch)[0]
        logger.info(
            f"Reject by threshold {reject_thresh} on {key.upper()} {len(reject_bad_channels)} : bad_channels: {reject_bad_channels}"
        )
        bad_ch_idx = np.concatenate((bad_ch_idx, reject_bad_channels))

    bad_chs = list({epochs.ch_names[i] for i in bad_ch_idx})
    return bad_chs


def _adaptive_egi(
    epochs,
    reject,
    n_epochs_bad_ch=0.5,
    n_channels_bad_epoch=0.1,
    zscore_thresh=4,
    max_iter=4,
    summary=None,
):
    """Find bad channels and bad epochs based on 4 adaptative steps.

    The steps are:
    1. Find bad channels based on the number of epochs where the channel values range (max - min) is over the reject threshold (see: find_bads_channels_threshold).
    2. Find bad channels based on iterative z-score outlier detection over the channel variance (see: find_bads_channels_variance).
    3. Find bad epochs based on the number of channels where the values range is over the reject threshold (see: find_bads_epochs_threshold).
    4. Find bad channels based on iterative z-score outlier detection over the channel high frequency variance (see: find_bads_channels_high_freq).

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    reject : dict
        The rejection threshold for each channel type (EEG, MEG, etc).
    n_epochs_bad_ch : int or float
        The number of epochs that have to be over the reject threshold for a channel to be considered bad. Default is 0.5.
    n_channels_bad_epoch : int or float
        The number of channels that have to be over the reject threshold for an epoch to be considered bad. Default is 0.1.
    zscore_thresh : int or float
        The z-score threshold for the z-scoring outlier detection. Default is 4.
    max_iter : int
        The maximum number of iterations for the z-scoring outlier detection. Default is 4.
    summary : dict
        The summary dictionary.

    Returns
    -------
    bad_channels : list of str
        The names of the bad channels.
    bad_epochs : list of int
        The indices of the bad epochs.

    """

    if isinstance(reject, float):
        reject = {"eeg": reject}

    bad_channels = set()
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, exclude="bads")

    # 1. Adaptive - threshold (Channels)
    method_params = {"reject": reject, "n_epochs_bad_ch": n_epochs_bad_ch}
    bad_chs = find_bads_channels_threshold(epochs, picks, **method_params)
    bad_channels.update(bad_chs)

    if summary is not None:
        summary["steps"].append(
            {
                "step": "adaptive/threshold",
                "params": method_params,
                "bad_chs": bad_chs,
            }
        )

    if len(bad_channels) == len(epochs.info["ch_names"]):
        logger.info("All channels are bad, skipping adaptive")
        return bad_channels, []

    # 2. Adaptive - variance (Channels)
    picks = mne.pick_channels(
        epochs.info["ch_names"], include=[], exclude=list(bad_channels)
    )
    method_params = {"zscore_thresh": zscore_thresh, "max_iter": max_iter}
    bad_chs = find_bads_channels_variance(epochs, picks, **method_params)
    bad_channels.update(bad_chs)

    if summary is not None:
        summary["steps"].append(
            {
                "step": "adaptive/variance",
                "params": method_params,
                "bad_chs": bad_chs,
            }
        )

    # 3. Adaptive - Threshold (Epochs)
    picks = mne.pick_channels(
        epochs.info["ch_names"], include=[], exclude=list(bad_channels)
    )
    method_params = {
        "reject": reject,
        "n_channels_bad_epoch": n_channels_bad_epoch,
    }
    bad_epochs = find_bads_epochs_threshold(epochs, picks, **method_params)
    epochs.drop(bad_epochs, reason="artifacted")

    if summary is not None:
        summary["steps"].append(
            {
                "step": "adaptive/threshold",
                "params": method_params,
                "bad_epochs": bad_epochs,
            }
        )

    if len(epochs) == 0:
        logger.info("All epochs are bad, skipping adaptive")
        return bad_channels, bad_epochs

    logger.info(f"found bad epochs: {len(bad_epochs)} {bad_epochs!s}")

    # 4. Adaptive - High frequency (Channels)
    picks = mne.pick_channels(
        epochs.info["ch_names"], include=[], exclude=list(bad_channels)
    )
    method_params = {"zscore_thresh": zscore_thresh, "max_iter": max_iter}
    bad_chs = find_bads_channels_high_frequency(epochs, picks, **method_params)
    bad_channels.update(bad_chs)

    if summary is not None:
        summary["steps"].append(
            {
                "step": "adaptive/highfreq",
                "params": method_params,
                "bad_chs": bad_chs,
            }
        )

    if summary is not None:
        method_params = {
            "zscore_thresh": zscore_thresh,
            "max_iter": max_iter,
            "reject": reject,
            "n_epochs_bad_ch": n_epochs_bad_ch,
            "n_channels_bad_epoch": n_channels_bad_epoch,
        }
        summary["steps"].append(
            {
                "step": "adative (total)",
                "params": method_params,
                "bad_chs": bad_channels,
                "bad_epochs": bad_epochs,
            }
        )

    bad_channels = sorted(bad_channels)
    if summary is not None:
        summary["bad_channels"] = bad_channels
        summary["bad_epochs"] = bad_epochs
    return bad_channels, bad_epochs


# ICM LG Event ID mapping (from next_icm/lg/constants.py)
ICM_LG_EVENT_ID = {
    "HSTD": 10,
    "HDVT": 20,
    "LSGS": 30,
    "LSGD": 40,
    "LDGS": 60,
    "LDGD": 50,
}
