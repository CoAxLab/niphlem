import numpy as np
import json
import matplotlib.pyplot as mpl

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

################################################################################
# main routine to read in signals and apply appropriate filter                 #
# in:  meta - json file containing frequencies for filtering                   #
#      sigFile - npy file containing processed signals                         #
################################################################################

#def filter_signals(meta, sigFile):

sigFile = 'signal.npy'
meta = 'meta.txt'

signal = np.load(sigFile)
meta = json.load(open(meta))
cardRange = meta['frequencyRanges'][0]['Cardiac']
respRange = meta['frequencyRanges'][0]['Respiratory']

# TODO: verify the 0.001 is right and signal is being filtered properly!
PULS = butter_bandpass_filter(signal[:, -2], cardRange[0], cardRange[1], 0.001)

mpl.plot(signal[:,-2], 'r')
mpl.plot(PULS, 'b')
mpl.show()
