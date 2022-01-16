import numpy as np
import json
import matplotlib.pyplot as mpl
import warnings


def butter_filter(data, *, fs, low_pass=None, high_pass=None, order=5):
    """
    Applies Butterworth bandpass double filter (to minimize shift).

    Parameters
    ---------
    data : vector
        signal to be filtered
    fs : real
        sampling frequency (Hz)
    low_pass : real
        frequency below passing data (Hz)
    high_pass : real
        frequency above passing data (Hz)
    order : int
        filter order (will be rounded up to even integer)

    Returns
    -------
    filtered signal
    """

    from scipy.signal import (sosfiltfilt, butter)

    if low_pass is None and high_pass is None:
        # TODO: Maybe we could delete this warning message
        warnings.warn("No low_pass or high_pass filtered supplied,"
                      " so no filtering was performed")
        return data

    fs = float(fs)
    nyq = 0.5 * fs

    if high_pass is not None and low_pass is not None:
        lowcut = float(high_pass)/nyq
        highcut = float(low_pass)/nyq
        Wn = [lowcut, highcut]
        btype = 'band'
    elif high_pass is not None:
        lowcut = float(high_pass)/nyq
        Wn = lowcut
        btype = 'highpass'
    else:
        highcut = float(low_pass)/nyq
        Wn = highcut
        btype = 'lowpass'

    sos = butter(N=np.ceil(order/2),
                 Wn=Wn,
                 analog=False,
                 btype=btype,
                 output='sos')

    return sosfiltfilt(sos, data)


# TODO: Ask Andrew about the use of this function
def filter_signals(meta, sig_file, show_signals=False):
    """
    Applies filters to data and write to new npy file

    Parameters
    ---------
    meta : str, pathlike
        name of json file containing meta data
    sig_file : str, pathlike
        name of npy file containing signal array
    show_signals: bool, optional
        flag to plot signals, default False
    """

    # load signal, columns: time, channels * nch, pulse, resp
    signal = np.load(sig_file)
    nch = signal.shape[1]-3

    # extract filtering information from JSON file
    meta = json.load(open(meta))
    sf = meta['frequency_info']['sampling_rate']
    card_range = meta['frequency_info']['cardiac_range']
    resp_range = meta['frequency_info']['respiratory_range']

    # filter
    filtered_signal = np.zeros_like(signal)
    filtered_signal[:, 0] = signal[:, 0]
    filtered_signal[:, -2] = butter_filter(signal[:, -2],
                                           high_pass=card_range[0],
                                           low_pass=card_range[1],
                                           fs=sf)
    filtered_signal[:, -1] = butter_filter(signal[:, -1],
                                           high_pass=resp_range[0],
                                           low_pass=resp_range[1],
                                           fs=sf)
    # note: factor of 5 is empirical
    for i in range(nch):
        filtered_signal[:, i] = butter_filter(signal[:, i],
                                              fs=sf,
                                              low_pass=card_range[1])
    # save filtered signal to filtered_signal.npy
    np.save('filtered_signal', filtered_signal)

    # plot signals if desired
    if show_signals:
        mpl.plot(signal[:, -2]-signal[:, -2].mean(), 'r')
        mpl.plot(filtered_signal[:, -2], 'b')
        mpl.show()
        mpl.plot(signal[:, -1]-signal[:, -1].mean(), 'r')
        mpl.plot(filtered_signal[:, -1], 'b')
        mpl.show()
        mpl.plot(signal[:, 1]-signal[:, 1].mean(), 'r')
        mpl.plot(filtered_signal[:, 1], 'b')
        mpl.show()

###############################################################################

# meta = 'meta.txt'
# sig_file = 'signal.npy'
# filter_signals(meta, sig_file, show_signals=True)


def _transform_filter(data,
                      *,
                      sampling_rate,
                      transform=None,
                      high_pass=None,
                      low_pass=None):

    # Guarantee original data is not overwritten
    data = data.copy()

    # TODO: Should we add an option with no transformation at all?
    if transform == "zscore":
        # zscore data
        data = zscore(data)
    elif transform == "abs":
        # Absolute value transformation on the data and zero mean the series
        data = abs(data)
        data = data - np.mean(data)
    else:
        # Only demean
        data = data - np.mean(data)

    data = butter_filter(data,
                         high_pass=high_pass,
                         low_pass=low_pass,
                         fs=sampling_rate)
    return data


def zscore(x, axis=1, nan_omit=True):
    """Standardize data."""
    if nan_omit:
        mean = np.nanmean
        std = np.nanstd
    else:
        mean = np.mean
        std = np.std

    zscores = (x - mean(x))/std(x)
    return zscores
