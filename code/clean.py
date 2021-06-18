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
nch = signal.shape[1]-3

meta = json.load(open(meta))
cardRange = meta['frequencyRanges'][0]['Cardiac']
respRange = meta['frequencyRanges'][0]['Respiratory']

sf = 1000
# TODO: verify 1000 Hz is right
#       extract this from the data
#       make sure the filter is working as expected
#       try to rectify the shift
PULS = butter_bandpass_filter(signal[:, -2], 0.5*cardRange[0], 2*cardRange[1], sf)
RESP = butter_bandpass_filter(signal[:, -1], 0.5*respRange[0], 2*respRange[1], sf)
ECG = np.zeros((len(signal), nch))
for i in range(nch):
  ECG[:, i] = signal[:, i]-gaussian_lowpass_filter(signal[:, i], sf, 0.5/cardRange[1])

mpl.plot(signal[:,-2]-signal[:,-2].mean(), 'r')
mpl.plot(PULS, 'b')
mpl.show()

mpl.plot(signal[:,-1]-signal[:,-1].mean(), 'r')
mpl.plot(PULS, 'b')
mpl.show()

mpl.plot(signal[:,1]-signal[:,1].mean(), 'r')
mpl.plot(ECG[:,1], 'b')
mpl.show()

