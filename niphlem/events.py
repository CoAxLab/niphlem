import numpy as np


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


def compute_max_events(signal, peak_rise, delta):

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

    return new_pks.astype(int)
