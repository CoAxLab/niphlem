import numpy as np
import json
import matplotlib.pyplot as mpl

###############################################################################
# applies Butterworth bandpass double filter (to minimize shift)              #
# in:  data - signal to be filtered                                           #
#      lowcut, highcut - cutoff frequencies (Hz)                              #
#      fs - sampling frequency (Hz)                                           #
#      order - filter order (will be rounded up to even integer)              #
# out: bandpass filtered sign                                                 #
###############################################################################


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
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

###############################################################################
# applies low pass filter to signal                                           #
# in:  data - signal to be filtered                                           #
#      fs - sampling frequency (Hz)                                           #
#      cut - cutoff frequency (Hz)                                            #
# out: low pass filtered signal                                               #
###############################################################################


def gaussian_lowpass_filter(data, fs, cut):

    from scipy.ndimage import gaussian_filter1d

    sigma = fs/(2*np.pi*cut)
    signal = gaussian_filter1d(data, sigma)
    return signal-np.mean(signal)

###############################################################################
# main routine to read in signals and apply appropriate filter                #
# in:  meta - json file containing frequencies for filtering                  #
#      sigFile - npy file containing processed signals                        #
#      showSignals - flag to plot the filtered signals (default False)        #
###############################################################################


def filter_signals(meta, sig_file, show_signals=False):

    # load signal, columns: time, channels * nch, pulse, resp
    signal = np.load(sig_file)
    nch = signal.shape[1]-3

    # extract filtering information from JSON file
    meta = json.load(open(meta))
    sf = meta['samplingRate'][0]['freq']
    card_range = meta['frequencyRanges'][0]['Cardiac']
    resp_range = meta['frequencyRanges'][0]['Respiratory']

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
    for i in range(1, nch+1):
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

#meta = 'meta.txt'
#sig_file = 'signal.npy'
#filter_signals(meta, sig_file, show_signals=True)
