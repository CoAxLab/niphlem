import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.interpolate import interp1d
from events import compute_max_events


class BasePhysio(BaseEstimator, TransformerMixin):

    def __init__(self,
                 physio_rate,
                 filtering="butter",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):

        # common arguments for all classess
        self.filtering = filtering
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.physio_rate = physio_rate
        self.columns = columns
        self.n_jobs = n_jobs

    def compute_regressors(self,
                           signal,
                           time_physio,
                           time_scan):

        # TODO: Add checks for input arguments and data

        # Decide how to handle data and loop through
        if self.columns:
            if self.columns == "mean":
                signal = signal.dot(self.columns)
            else:
                signal = signal.dot(self.columns)

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        # TODO: Add previous transformations: Filtering, what else?

        parallel = Parallel(n_jobs=self.n_jobs)
        func = self._process_regressors
        regressors = parallel(delayed(func)(s, time_physio, time_scan)
                              for s in signal.T)
        regressors = np.column_stack(regressors)
        return regressors

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        # bare method to be overwitten by derived classes
        raise NotImplementedError


class RetroicorPhysio(BasePhysio):

    def __init__(self,
                 physio_rate,
                 delta,
                 peak_rise=0.5,
                 order=1,
                 filtering="butter",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):

        self.order = order
        self.delta = delta
        self.peak_rise = peak_rise

        super().__init__(filtering=filtering,
                         high_pass=high_pass,
                         low_pass=low_pass,
                         physio_rate=physio_rate,
                         columns=columns,
                         n_jobs=n_jobs)

    def _process_regressors(self, signal, time_physio, time_scan):

        # TODO: Add checks specific to this task, like the Fourier order
        # expansion to be # greater > 0

        # Compute peaks in signal
        peaks = compute_max_events(signal, self.peak_rise, self.delta)

        # Compute phases according to peaks
        phases = compute_phases(time_physio, peaks)

        # Expand phases according to Fourier expansion order
        phases_fourier = [(m*phases).reshape(-1, 1)
                          for m in range(1, self.order+1)]
        phases_fourier = np.column_stack(phases_fourier)

        # Downsample phases
        phases_scan = np.zeros((len(time_scan), phases_fourier.shape[1]))
        for ii, phases in enumerate(phases_fourier.T):
            interp = interp1d(time_physio,
                              phases,
                              kind='linear',  # TODO: Add this as parameter?
                              fill_value='extrapolate')
            phases_scan[:, ii] = interp(time_scan)

        # Apply sin and cos functions
        sin_phases = np.sin(phases_scan)
        cos_phases = np.cos(phases_scan)

        # This is just to be ordered according to the fourier expansion
        regressors = [np.column_stack((a, b))
                      for a, b in zip(sin_phases.T, cos_phases.T)]
        return np.column_stack(regressors)


class RVPhysio(BasePhysio):

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        """
        Compute rate variations for respiration signal computed by taking
        the standard deviation of the raw respiratory waveform over the 3 TR
        time interval defined by the (k − 1)th, kth, and (k + 1)th TRs.
        Thus, RV(k) is essentially a sliding-window measure related
        to the inspired volume over time. (Chang et al 2009)
        """

        # TODO: Add checks specific to this task

        N = len(time_scan)
        rv_values = np.zeros(N)

        for ii in range(N):
            # TODO: See border effects and maybe a way to optimize this
            if ii == 0:
                t_i = 0
                t_f = time_scan[ii+1]
            elif ii == N-1:
                t_i = time_scan[ii-1]
                t_f = time_scan[ii]
            else:
                t_i = time_scan[ii-1]
                t_f = time_scan[ii+1]

            # Take points of recording between these times
            mask_times = (time_physio >= t_i) & (time_physio <= t_f)
            rv_values[ii] = np.std(signal[mask_times])

        # return zscores of these values
        rv_values = zscore(rv_values)

        def RRF(t):
            """
            Respiration Response Function (eq. 3 Birn 2008)
            """
            rrf = 0.6*(t**2.1)*np.exp(-t/1.6)
            rrf -= 0.0023*(t**3.54)*np.exp(-t/4.25)
            return rrf

        t_rrf = np.arange(0, 28, self.scan_rate)
        rrf = np.apply_along_axis(RRF, axis=0, arr=t_rrf)
        # Convolve rv values with response function
        regressors = np.convolve(rv_values, rrf)
        regressors = regressors[:N]  # Trim excess
        return regressors


class HVPhysio(BasePhysio):

    def __init__(self,
                 physio_rate,
                 delta,
                 peak_rise=0.5,
                 filtering="butter",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):

        self.delta = delta
        self.peak_rise = peak_rise

        super().__init__(filtering=filtering,
                         high_pass=high_pass,
                         low_pass=low_pass,
                         physio_rate=physio_rate,
                         columns=columns,
                         n_jobs=n_jobs)

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        # Compute peaks in signal
        peaks = compute_max_events(signal, self.peak_rise, self.delta)
        # TODO: Add checks specific to this task
        peaks = peaks.astype(int)

        # Compute times of maximum event peaks
        time_peaks = time_physio[peaks]

        N = len(time_scan)

        hv_values = np.zeros(N)
        for ii in range(N):
            # TODO: See border effects and maybe a way to optimize this
            if ii == 0:
                t_i = 0
                t_f = time_scan[ii+1]
            elif ii == N-1:
                t_i = time_scan[ii-1]
                t_f = time_scan[ii]
            else:
                t_i = time_scan[ii-1]
                t_f = time_scan[ii+1]

            mask_times = np.where((time_peaks >= t_i) & (time_peaks <= t_f))[0]
            hv_values[ii] = 60./np.mean(np.diff(time_peaks[mask_times]))

        # return zscores of these values
        hv_values = zscore(hv_values)

        def CRF(t):
            """

            Cardiac Response Function (eq. 5 Chang 2009)

            """
            crf = 0.6*(t**2.7)*np.exp(-t/1.6)
            crf -= 16./(np.sqrt(2*np.pi*9))*np.exp(-(0.5/9)*(t-12.)**2)
            return crf

        t_crf = np.arange(0, 28, self.scan_rate)
        crf = np.apply_along_axis(CRF, axis=0, arr=t_crf)
        # Convolve rv values with response function
        regressors = np.convolve(hv_values, crf)
        regressors = regressors[:N]  # Trim excess
        return regressors


def compute_phases(time, max_peaks):
    """
    This function compute the phase between successive peaks
    events as provided by compute_max_events function.
    The values between peaks are mapped to being in the range [0, 2 pi]
    (eq. 2 Glover 2000)
    """

    n_maxs = len(max_peaks)
    phases = np.zeros_like(time, dtype=np.float)
    N = len(time)

    for ii in range(n_maxs):
        # Look at the tails
        if ii == n_maxs-1:
            i_o, i_f = int(max_peaks[ii]), int(max_peaks[0])
            t_o, t_f = time[i_o], time[::-1][0] + time[i_f]

            phases[i_o:] = 2*np.pi*(time[i_o:N] - t_o)/(t_f-t_o)
            phases[:i_f] = 2*np.pi*(time[::-1][0] + time[:i_f] - t_o)/(t_f-t_o)
        else:
            i_o, i_f = int(max_peaks[ii]), int(max_peaks[ii+1])
            t_o, t_f = time[i_o], time[i_f]
            phases[i_o:i_f] = 2*np.pi*(time[i_o:i_f] - t_o)/(t_f-t_o)

    return phases


def zscore(x, axis=1, nan_omit=True):

    if nan_omit:
        mean = np.nanmean
        std = np.nanstd
    else:
        mean = np.mean
        std = np.std

    zscores = (x - mean(x))/std(x)
    return zscores
