import numpy as np
from OUTLIERS import smirnov_grubbs as grubbs

def peakdet(v, delta=0.5, x=None):
    """
    Translation to python of the function  peakdet.m in
    Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.
    """
    maxtab, mintab = [], []
    if x is None:
        x = np.arange(len(v))

    if len(v) != len(x):
        raise ValueError('Input vectors v and x must have same length')

    if hasattr(delta, "__iter__") and (not isinstance(delta, str)):
        raise ValueError('Input argument th (threshold) must be a scalar')

    if delta <= 0:
        raise ValueError('Input argument th (threshold) must be positive')

    N = len(v)
    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan
    lookformax = True

    for i in range(N):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append([mxpos, mx])
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append([mnpos, mn])
                mx = this
                mxpos = x[i]
                lookformax = True

    maxtab = np.array(maxtab)
    mintab = np.array(mintab)

    return (maxtab, mintab)


def compute_max_events(signal, peak_rise=0.75, delta=200):

    # Compute peaks from a first pass
    maxtab, mintab = peakdet(signal, delta=1e-14)

    # set the threshold based on the 20th highest
    sorted_peaks = np.sort(maxtab[:, 1])[::-1]
    if sorted_peaks.size > 20:
        peak_resp = sorted_peaks[20]
    else:
        peak_resp = sorted_peaks[0]

    # Second pass, more robustly filtered, to get the actual peaks:
    maxtab, mintab = peakdet(signal, delta=peak_rise*peak_resp)

    pks_time = maxtab[:, 0]
    dpks_time = np.diff(pks_time)
    # find separated peaks by delta
    kppks_time = np.where(dpks_time > delta)[0] + 1
    new_pks = np.insert(pks_time[kppks_time], 0, values=pks_time[0])
    peaks = new_pks.astype(int)

    return peaks



def correct_anomalies(peaks, alpha=0.05, save_name=''):
    """
    Outlier detection (Grubb's test) and removal

    Parameters
    ----------
    peak_diffs : array
        vector of peak-to-peak distances
    alpha : real
        significance level for Grubb's test
    save_name : str
        filename to save peaks as to, empty does not save

    Results
    -------
    peaks : array
        vector of peak locations
    """

    peak_diffs = abs(np.diff(peaks))
    max_indices = grubbs.max_test_indices(peak_diffs, alpha=0.05)
    # insert new peak halfway between too long RR interval
    too_slow = np.array(max_indices)
    new_peaks = np.zeros_like(too_slow, dtype=float)

    for index,i in enumerate(too_slow):
        new_diff = (peaks[i+1] - peaks[i])/2 #new diff_peak
        new_peak = peaks[i] + new_diff #new peak to insert into corrected peaks array
        new_peaks[index] = new_peak

    corrected_peaks = np.insert(peaks, (too_slow+1).reshape(-1), new_peaks.reshape(-1))
    corrected_peak_diffs = abs(np.diff(corrected_peaks))
    mean_RR = np.mean(corrected_peak_diffs)

    min_indices = grubbs.min_test_indices(corrected_peak_diffs, alpha=0.05)

    # deleting peak such that resultant RR interval is furthest from mean RR (i.e. gives longer RR interval)
    too_fast = np.array(min_indices)
    peaks_to_replace = np.zeros_like(too_fast) #index of peaks to delete (and then reinsert)
    new_peaks2 = np.zeros_like(too_fast, dtype=float)

    for index,i in enumerate(too_fast):

        #print(index, i)
        if i == (corrected_peak_diffs.size - 1): #if last RR interval (edge case)
            peaks_to_replace[index] = i #replace first peak
            new_diff = (corrected_peaks[i+1] - corrected_peaks[i-1])/2 #new diff_peak
            new_peaks2[index] = corrected_peaks[i-1] + new_diff
        else:
            new_diff1 = corrected_peaks[i+1] - corrected_peaks[i-1] #replace first peak
            new_diff2 = corrected_peaks[i+2] - corrected_peaks[i] #replace second peak

            if (new_diff1 - mean_RR) > (new_diff2 - mean_RR):
                peaks_to_replace[index] = i #replacing first peak results in new RR interval furthest from mean RR interval
                new_diff = (corrected_peaks[i+1] - corrected_peaks[i-1])/2 #new diff_peak
                new_peaks2[index] = corrected_peaks[i-1] + new_diff
            else:
                peaks_to_replace[index] = i+1 #replacing second peak results in new RR interval furthest from mean RR interval
                new_diff = (corrected_peaks[i+2] - corrected_peaks[i])/2 #new diff_peak
                new_peaks2[index] = corrected_peaks[i] + new_diff

    corrected_peaks2 = corrected_peaks.copy()
    np.put(corrected_peaks2, peaks_to_replace.astype(int), new_peaks2)
    corrected_peak_diffs2 = abs(np.diff(corrected_peaks2))

    # save peaks
    if save_name != '':
        np.savetxt(save_name, corrected_peak_diffs2, delimiter=',')

    return corrected_peak_diffs2, max_indices, min_indices
