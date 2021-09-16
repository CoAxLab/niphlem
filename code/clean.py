import numpy as np
import json
import matplotlib.pyplot as mpl

##############################################################################
# applies Butterworth bandpass double filter (to minimize shift)             #
# in:  data - signal to be filtered                                          #
#      lowcut, highcut - cutoff frequencies (Hz)                             #
#      fs - sampling frequency (Hz)                                          #
#      order - filter order (will be rounded up to even integer)             #
# out: bandpass filtered signal                                              #
##############################################################################


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # fs, lowcut and highcut are in frequency units (Hz)
    from scipy.signal import (sosfiltfilt, butter)

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(np.ceil(order/2),
                 [low, high],
                 analog=False,
                 btype='band',
                 output='sos')

    return sosfiltfilt(sos, data)

##############################################################################
# applies low pass filter to signal                                          #
# in:  data - signal to be filtered                                          #
#      fs - sampling frequency (Hz)                                          #
#      cut - cutoff frequency (Hz)                                           #
# out: low pass filtered signal                                              #
##############################################################################


def gaussian_lowpass_filter(data, fs, cut):

    from scipy.ndimage import gaussian_filter1d

    sigma = fs/(2*np.pi*cut)
    signal = gaussian_filter1d(data, sigma)
    return signal-np.mean(signal)  # TODO: Ask Andrew why to demean data here

################################################################################
# main routine to read in signals and apply appropriate filter                 #
# in:  meta - json file containing frequencies for filtering                   #
#      sigFile - npy file containing processed signals                         #
#      showSignals - flag to plot the filtered signals (default False)         #
################################################################################

def filterSignals(meta, sigFile, showSignals=False):

  # load signal, columns: time, channels * nch, pulse, resp
  signal = np.load(sigFile)
  nch = signal.shape[1]-3

  # extract filtering information from JSON file
  meta = json.load(open(meta))
  sf = meta['samplingRate'][0]['freq']
  cardRange = meta['frequencyRanges'][0]['Cardiac']
  respRange = meta['frequencyRanges'][0]['Respiratory']

  # filter
  fSignal = np.zeros_like(signal)
  fSignal[:, 0] = signal[:, 0]
  fSignal[:, -2] = butter_bandpass_filter(signal[:, -2], cardRange[0], cardRange[1], sf)
  fSignal[:, -1] = butter_bandpass_filter(signal[:, -1], respRange[0], respRange[1], sf)
  # note: factor of 5 is empirical
  for i in range(1, nch+1):
    fSignal[:, i] = gaussian_lowpass_filter(signal[:, i], sf, cardRange[1])
  # save filtered signal to fSignal.npy
  np.save('fSignal', fSignal)

  # plot signals if desired
  if showSignals:
    mpl.plot(signal[:, -2]-signal[:, -2].mean(), 'r')
    mpl.plot(fSignal[:, -2], 'b')
    mpl.show()
    mpl.plot(signal[:, -1]-signal[:, -1].mean(), 'r')
    mpl.plot(fSignal[:, -1], 'b')
    mpl.show()
    mpl.plot(signal[:, 1]-signal[:, 1].mean(), 'r')
    mpl.plot(fSignal[:, 1], 'b')
    mpl.show()

###############################################################################

#meta = 'meta.txt'
#sigFile = 'signal.npy'
#filterSignals(meta, sigFile, showSignals=True)


def clean_data(data,
               transform,
               filtering,
               high_pass,
               low_pass,
               sampling_rate):

    # Guarantee original data is not overwritten
    data = data.copy()

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
