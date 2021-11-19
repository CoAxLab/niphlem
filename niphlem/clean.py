import numpy as np
import json
import matplotlib.pyplot as mpl



def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies Butterworth bandpass double filter (to minimize shift).

    Parameters
    ---------
    data : vector
        signal to be filtered
    lowcut : real
        lowpass cutoff frequency (Hz)
    highcut : real
        highpass cutoff frequency (Hz)
    fs : real
        sampling frequency (Hz)
    order : int
        filter order (will be rounded up to even integer)

    Returns
    -------
    bandpass filtered signal
    """

    # fs, lowcut and highcut are in frequency units (Hz)
    from scipy.signal import (sosfiltfilt, butter)

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(N=np.ceil(order/2),
                 Wn=[low, high],
                 analog=False,
                 btype='band',
                 output='sos')

    return sosfiltfilt(sos, data)



def gaussian_lowpass_filter(data, fs, cut):
    """
    Applies Gaussian lowpass filter.

    Parameters
    ---------
    data : array
        signal to be filtered
    fs : real
        sampling frequency (Hz)
    cut : real
        cutoff frequency (Hz)

    Returns
    -------
    lowpass filtered signal
    """

    from scipy.ndimage import gaussian_filter1d

    sigma = fs/(2*np.pi*cut)
    signal = gaussian_filter1d(data, sigma)

    return signal



def _transform_filter(data,
                      ground_ch=0,
                      transform="demean",
                      filtering="none",
                      average_signal=False,
                      high_pass=0,
                      low_pass=0,
                      sampling_rate=0,
                      save_name=''):
    """
    Ground, transform, and filter signal as specified

    Parameters
    ---------
    data : array
        data to ground/transform/filter
    ground_ch : int
        ground channel index, one-based index (i.e. not pythonic)
    transform : str
        transform option <none, demean, abs, zscore>
    filtering : str
        filtering option <none, butter, guassian>
    average_signal : bool
        flag to average signal across channels
    high_pass : real
        high pass frequency for filtering (Hz)
    low_pass : real
        low pass frequency for filtering (Hz)
    sampling_rate : real
        signal sampling frequency (Hz)
    save_name : str
        filename to save cleaned signal to, empty does not save

    Returns
    -------
    data : array
        grounded/transformed/filtered signal
    """

    # Value checking
    if filtering == "butter" or filtering == "gaussian":
        if low_pass <= 0:
            raise Exception("Low pass frequency must be provided to filter")
        if sampling_rate <= 0:
            raise Exception("Sampling rate must be provided to filter")
    if filtering == "butter" and high_pass <= 0:
            raise Exception("High pass frequency must be providedd to filter")
    # TODO: ensure data is 2D array?

    # Guarantee original data is not overwritten
    data = data.copy()
    nch = data.shape[1]

    # convert to python indexing
    ground_ch = ground_ch - 1

    for ich in range(nch):

        if ground_ch >= 0:
            # ground channels
            if nch != ground_ch:
                data[:, ich] = data[:, ich] - data[:, ground_ch]

        if transform == "zscore":
            # zscore data
            data[:, ich] = zscore(data[:, ich])
        elif transform == "abs":
            # Absolute value transformation on the data and zero mean the series
            data[:, ich] = abs(data[:, ich])
            data[:, ich] = data[:, ich] - np.mean(data[:, ich])
        elif transform == "demean":
            # Only demean
            data[:, ich] = data[:, ich] - np.mean(data[:, ich])

        if filtering == "butter":
            # TODO: Add more flexible butter filter, not only bandpass?
            # high_pass == frequency above to clean, then here is the lower range,
            # low_pass == frequency below to clean, then here is the higher range.
            data[:, ich] = butter_bandpass_filter(data[:, ich],
                                                  lowcut=high_pass,
                                                  highcut=low_pass,
                                                  fs=sampling_rate)
        elif filtering == "gaussian":
            data[:, ich] = gaussian_lowpass_filter(data[:, ich],
                                                   fs=sampling_rate,
                                                   cut=low_pass)
    if nch > 1 and average_signal:
        # average signals across channels
        data = np.average(data, axis=1)

    if save_name != '':
        np.savetxt(save_name, data, delimiter=',')

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

