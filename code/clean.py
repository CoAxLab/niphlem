import numpy as np

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # fs, lowcut and highcut are in frequency units (Hz)
    from scipy.signal import (sosfiltfilt, butter)

    low = lowcut / fs
    high = highcut / fs
    sos = butter(np.ceil(order/2), [low, high], analog=False, btype='band', output='sos')

    y = sosfiltfilt(sos, data)
    return y

def gaussian_lowpass_filter(data, fs, fwhm):
    # sampling rate is frequency (in Hz)
    # fwhm is time (in seconds)

    from scipy.ndimage import gaussian_filter1d

    sigma = fs*fwhm #
    return gaussian_filter1d(data, sigma)

def filter_signals(json, signal)
