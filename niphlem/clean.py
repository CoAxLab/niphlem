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
    filtered_signal[:, -2] = butter_bandpass_filter(signal[:, -2],
                                                    card_range[0],
                                                    card_range[1],
                                                    sf)
    filtered_signal[:, -1] = butter_bandpass_filter(signal[:, -1],
                                                    resp_range[0],
                                                    resp_range[1],
                                                    sf)
    # note: factor of 5 is empirical
    for i in range(nch):
        filtered_signal[:, i] = gaussian_lowpass_filter(signal[:, i],
                                                        sf,
                                                        card_range[1])
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
                      transform,
                      filtering,
                      high_pass,
                      low_pass,
                      sampling_rate):

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

    if filtering == "butter":
        # TODO: Add more flexible butter filter, not only bandpass?
        # high_pass == frequency above to clean, then here is the lower range,
        # low_pass == frequency below to clean, then here is the higher range.
        data = butter_bandpass_filter(data,
                                      lowcut=high_pass,
                                      highcut=low_pass,
                                      fs=sampling_rate)
    elif filtering == "gaussian":
        data = gaussian_lowpass_filter(data,
                                       fs=sampling_rate,
                                       cut=low_pass)
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
  