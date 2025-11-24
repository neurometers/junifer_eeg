"""Utilities for artifact rejection preprocessing."""

import math

import mne
import numpy as np
from mne.utils import logger
from scipy.signal import butter, filtfilt


def find_bads_channels_variance(inst, picks, zscore_thresh=4, max_iter=2):
    """Find bad channels based on Z-scoring outliers of the channel variances.

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
    if isinstance(inst, mne.BaseEpochs):
        data = inst.get_data()
    else:
        data = inst._data[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    ch_var = np.ma.hstack(masked_data).var(axis=-1)

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
    """Find bad channels based on Z-scoring outliers of high frequency variance.

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
    if isinstance(inst, mne.BaseEpochs):
        data = inst.get_data()
    else:
        data = inst._data[None, :]
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
    exclude = np.array([x for x in range(data.shape[1]) if x not in picks])
    if len(exclude) > 0:
        masked_data[:, exclude, :] = np.ma.masked
    filter_freq = 25
    b, a = butter(4, 2.0 * filter_freq / inst.info["sfreq"], "highpass")
    filt_data = filtfilt(b, a, np.ma.hstack(masked_data))
    filt_masked_data = np.ma.masked_array(filt_data, fill_value=np.nan)
    if len(exclude) > 0:
        filt_masked_data[exclude, :] = np.ma.masked

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
    """Find bad epochs based on threshold rejection.

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
    bad_epochs : list of int
        The indices of the bad epochs.
    """
    n_channels = len(picks)
    bad_ep_idx = np.ndarray((0,), dtype=np.int32)
    if isinstance(n_channels_bad_epoch, float):
        n_channels_bad_epoch = math.floor(n_channels_bad_epoch * n_channels)

    data = epochs.get_data()
    masked_data = np.ma.masked_array(data, fill_value=np.nan)
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
    """Find bad channels based on threshold rejection.

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


def check_min_channels(epochs, bad_channels, min_channels):
    """Check if minimum number of channels is maintained.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    bad_channels : list of str
        List of bad channel names.
    min_channels : int or float
        Minimum number or fraction of channels required.

    Raises
    ------
    Exception
        If too many channels are marked as bad.
    """
    if isinstance(min_channels, float):
        logger.info(
            f"Using relative min_channels: {min_channels} * {epochs.info['nchan']} = {epochs.info['nchan'] * min_channels} "
            "channels remaining to reject preprocess"
        )
        min_channels = int(epochs.info["nchan"] * min_channels)

    chans_remaining = epochs.info["nchan"] - len(bad_channels)
    if chans_remaining < min_channels:
        msg = f"Can not clean data. Only {chans_remaining} out of {epochs.info['nchan']} channels remaining."
        logger.error(msg)
        raise Exception(msg)


def check_min_events(epochs, min_events):
    """Check if minimum number of events is maintained.

    Parameters
    ----------
    epochs : instance of mne.Epochs
        The epochs object.
    min_events : int or float
        Minimum number or fraction of events required.

    Raises
    ------
    Exception
        If too many epochs are dropped.
    """
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
