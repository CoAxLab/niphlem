"""Module for regressors computation."""

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array, column_or_1d)

from events import compute_max_events
from clean import _transform_filter, zscore


class BasePhysio(BaseEstimator):
    """
     Bare class.

     Bare class from which more specific, e.g. Retroicor and
     Variational, classes will inherit.

    Parameters
    ----------
    physio_rate : float
        Sampling rate for the recording in Hz.
        This is needed for filtering to define the nyquist frequency.
    scan_rate : float
        Sampling rate for the scanner (the usual T_R) in Hz.
    tranform : {"mean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "mean".
    filtering : {"butter", "gaussian", None}, optional
        Filtering operation to perform. The default is None.
    high_pass : float, optional
        High-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    columns : List of n_channels elements, "mean" or None, optional
        It describe how to hande input signal channels. If a list,
        it will take the dot product. If "mean", the average across
        the channels. If None, it will consider each channel separately.
        The default is None.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 physio_rate,
                 scan_rate,
                 transform="mean",
                 filtering=None,
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):
        # common arguments for all classess
        self.physio_rate = physio_rate
        self.scan_rate = scan_rate
        self.transform = transform
        self.filtering = filtering
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.columns = columns
        self.n_jobs = n_jobs

    def compute_regressors(self,
                           signal,
                           time_physio,
                           time_scan):
        """
        Compute regressors.

         It basically takes care of all the preprocessing before
         concentrating on the specific type of physiological regressors

        Parameters
        ----------
        signal : array -like of shape (n_physio_samples, n_channels)
            Signal, where each column corresponds to a recording.
        time_physio : array -like of shape (n_physio_samples, )
            Time ticks (in secs) at the physio recording resolution.
        time_scan :  array -like of shape (n_scan_samples, )
            Time ticks (in secs) at the scanner resolution.

        Returns
        -------
        array-like with the physiological regressors

        """
        signal = check_array(signal)
        time_physio = column_or_1d(time_physio)
        time_scan = column_or_1d(time_scan)

        if self.transform not in ["mean", "zscore", "abs"]:
            raise ValueError(f"{self.transform} transform option passed, "
                             "but only 'mean' (default), 'zscore' or 'abs' "
                             "are allowed")
        if self.filtering not in [None, "butter", "gaussian"]:
            raise ValueError(f"{self.transform} filtering option passed, "
                             "but only None (default), 'butter' or 'gaussian' "
                             "are allowed")
        if self.filtering == "butter":
            if (self.high_pass is None) or (self.low_pass is None):
                raise ValueError("Butterworth bandapss selected, but "
                                 "either high_pass or low_pass is missing")
        elif self.filtering == "gaussian":
            if self.low_pass is None:
                raise ValueError("gaussian lowpass selected, but "
                                 "low_pass argument is missing")

        # Decide how to handle data and loop through
        if self.columns:
            if self.columns == "mean":
                signal = signal.dot(self.columns)
            else:
                signal = signal.dot(self.columns)

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

        parallel = Parallel(n_jobs=self.n_jobs)

        signal_clean = parallel(
            delayed(_transform_filter)(data=s,
                                       transform=self.transform,
                                       filtering=self.filtering,
                                       high_pass=self.high_pass,
                                       low_pass=self.low_pass,
                                       sampling_rate=self.physio_rate)
            for s in signal.T)

        func = self._process_regressors
        regressors = parallel(delayed(func)(s,
                                            time_physio,
                                            time_scan)
                              for s in signal_clean.T)
        regressors = np.column_stack(regressors)
        return regressors

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        # bare method to be overwitten by derived classes
        raise NotImplementedError


class RetroicorPhysio(BasePhysio):
    """
     Physiological regressors using Retroicor.

    Parameters
    ----------
    physio_rate : float
        Sampling rate for the recording in Hz.
        This is needed for filtering to define the nyquist frequency.
    scan_rate : float
        Sampling rate for the scanner (the usual T_R) in Hz.
    delta: float
        minimum separation (in physio recording units) between
        events in signal to be considered peaks
    peak_rise: float
        relative height with respect to the 20th tallest events in signal
        to consider events as peak.
    order: int or array-like (# TODO) of shape (n_orders,)
        Fourier expansion for phases. If int, the fourier expansion is
        performed to that order, starting from 1. If an array is provided,
        each element will multiply the phases.
    tranform : {"mean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "mean".
    filtering : {"butter", "gaussian", None}, optional
        Filtering operation to perform. The default is None.
    high_pass : float, optional
        High-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    columns : List of n_channels elements, "mean" or None, optional
        It describe how to hande input signal channels. If a list,
        it will take the dot product. If "mean", the average across
        the channels. If None, it will consider each channel separately.
        The default is None.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 physio_rate,
                 scan_rate,
                 delta,
                 peak_rise=0.5,
                 order=1,
                 transform="mean",
                 filtering="butter",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):

        self.order = order
        self.delta = delta
        self.peak_rise = peak_rise

        super().__init__(physio_rate=physio_rate,
                         scan_rate=scan_rate,
                         transform=transform,
                         filtering=filtering,
                         high_pass=high_pass,
                         low_pass=low_pass,
                         columns=columns,
                         n_jobs=n_jobs)

    def _process_regressors(self, signal, time_physio, time_scan):
        """
        Generate regressors as phases using RETROICOR method.

        First, peaks in the signal are identified. Then, phases
        are sampled such a complete cycle [0, 2pi] happens between peaks.
        Then, phases are expanded up to or for a given fourier order. Then,
        phases are subsampled to the scanner time. Finally, sine and cosine
        are generated for each fourier mode.

        Parameters
        ----------
        signal : array -like of shape (n_physio_samples, n_channels)
            Signal, where each column corresponds to a recording.
        time_physio : array -like of shape (n_physio_samples, )
            Time ticks (in secs) at the physio recording resolution.
        time_scan :  array -like of shape (n_scan_samples, )
            Time ticks (in secs) at the scanner resolution.

        Returns
        -------
        array-like with the physiological regressors

        """
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
    """
     Physiological regressors for variations in breathing rate/volume.

    Parameters
    ----------
    physio_rate : float
        Sampling rate for the recording in Hz.
        This is needed for filtering to define the nyquist frequency.
    scan_rate : float
        Sampling rate for the scanner (the usual T_R) in Hz.
    tranform : {"mean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "mean".
    filtering : {"butter", "gaussian", None}, optional
        Filtering operation to perform. The default is None.
    high_pass : float, optional
        High-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    columns : List of n_channels elements, "mean" or None, optional
        It describe how to hande input signal channels. If a list,
        it will take the dot product. If "mean", the average across
        the channels. If None, it will consider each channel separately.
        The default is None.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        """
        Generate regressors as variations in breathing rate/volume.

        Compute rate variations for respiration signal computed by taking
        the standard deviation of the raw respiratory waveform over the 3 TR
        time interval defined by the (k − 1)th, kth, and (k + 1)th TRs.
        The resulting vector is then convolved with the respiratory response
        function (Birn et al 2008).

        Parameters
        ----------
        signal : array -like of shape (n_physio_samples, n_channels)
            Signal, where each column corresponds to a recording.
        time_physio : array -like of shape (n_physio_samples, )
            Time ticks (in secs) at the physio recording resolution.
        time_scan :  array -like of shape (n_scan_samples, )
            Time ticks (in secs) at the scanner resolution.

        Returns
        -------
        array-like with the physiological regressors

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
            """Respiration Response Function (eq. 3 Birn 2008)."""
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
    """
     Physiological regressors for variations in heart rate.

    Parameters
    ----------
    physio_rate : float
        Sampling rate for the recording in Hz.
        This is needed for filtering to define the nyquist frequency.
    scan_rate : float
        Sampling rate for the scanner (the usual T_R) in Hz.
    delta: float
        minimum separation (in physio recording units) between
        events in signal to be considered peaks
    peak_rise: float
        relative height with respect to the 20th tallest events in signal
        to consider events as peak.
    tranform : {"mean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "mean".
    filtering : {"butter", "gaussian", None}, optional
        Filtering operation to perform. The default is None.
    high_pass : float, optional
        High-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). Only if filtering option
        is not None. The default is None.
    columns : List of n_channels elements, "mean" or None, optional
        It describe how to hande input signal channels. If a list,
        it will take the dot product. If "mean", the average across
        the channels. If None, it will consider each channel separately.
        The default is None.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 physio_rate,
                 scan_rate,
                 delta,
                 peak_rise=0.5,
                 transform="mean",
                 filtering="butter",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):

        self.delta = delta
        self.peak_rise = peak_rise

        super().__init__(physio_rate=physio_rate,
                         scan_rate=scan_rate,
                         transform=transform,
                         filtering=filtering,
                         high_pass=high_pass,
                         low_pass=low_pass,
                         columns=columns,
                         n_jobs=n_jobs)

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        """
        Generate regressors as variations in hear rate.

        Compute rate variations in heart rate by taking the average
        time differences between peaks over the 3 TR
        time interval defined by the (k − 1)th, kth, and (k + 1)th TRs.
        The resulting vector is then convolved with the cardiac response
        function (Chang et al 2009).

        Parameters
        ----------
        signal : array -like of shape (n_physio_samples, n_channels)
            Signal, where each column corresponds to a recording.
        time_physio : array -like of shape (n_physio_samples, )
            Time ticks (in secs) at the physio recording resolution.
        time_scan :  array -like of shape (n_scan_samples, )
            Time ticks (in secs) at the scanner resolution.

        Returns
        -------
        array-like with the physiological regressors

        """
        from clean import zscore
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
            """Cardiac Response Function (eq. 5 Chang 2009)."""
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
    Compute phases for RETROICOR.

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
