"""Module for regressors computation."""

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d

from sklearn.base import BaseEstimator
from sklearn.utils.validation import (check_array, column_or_1d)

from .events import compute_max_events, correct_anomalies
from .clean import _transform_filter, zscore


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
    t_r : float
        Repetition time for the scanner (the usual T_R) in secs.
    transform : {"demean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "demean".
    high_pass : float, optional
        High-pass filtering frequency (in Hz). The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). The default is None.
    columns : List or array of n_channels elements, "mean" or None, optional
        It describes how to handle input signal channels. If a list, it will
        weight each channel and take the dot product. If "mean",
        the average across the channels. If None, it will consider each
        channel separately. The default is None.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 *,
                 physio_rate,
                 t_r,
                 transform="demean",
                 filtering=None,
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):
        # common arguments for all classess
        self.physio_rate = physio_rate
        self.t_r = t_r
        self.transform = transform
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.columns = columns
        self.n_jobs = n_jobs

    def compute_regressors(self,
                           signal,
                           time_scan,
                           time_physio=None):
        """
        Compute regressors.

         It basically takes care of all the preprocessing before
         concentrating on the specific type of physiological regressors

        Parameters
        ----------
        signal : array -like of shape (n_physio_samples, n_channels)
            Signal, where each column corresponds to a recording.
        time_scan :  array -like of shape (n_scan_samples, )
            Time ticks (in secs) at the scanner resolution.
        time_physio : array -like of shape (n_physio_samples, )
            Time ticks (in secs) at the physio recording resolution.
            The default is None. In this default case, the time of the
            physiological recording is computed multiplying
            the number of points with the sampling period.

        Returns
        -------
        array-like with the physiological regressors

        """

        # Validate input parameters and arguments
        signal, time_scan, time_physio = self._validate_inputs(signal,
                                                               time_scan,
                                                               time_physio)

        parallel = Parallel(n_jobs=self.n_jobs)

        signal_prep = parallel(
            delayed(_transform_filter)(data=s,
                                       transform=self.transform,
                                       high_pass=self.high_pass,
                                       low_pass=self.low_pass,
                                       sampling_rate=self.physio_rate)
            for s in signal.T)
        signal_prep = np.column_stack(signal_prep)

        func = self._process_regressors
        regressors = parallel(delayed(func)(s,
                                            time_physio,
                                            time_scan)
                              for s in signal_prep.T)
        regressors = np.column_stack(regressors)
        return regressors

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        # bare method to be overwitten by derived classes
        raise NotImplementedError

    def _validate_inputs(self,
                         signal,
                         time_scan,
                         time_physio):

        signal = np.asarray(signal)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        signal = check_array(signal)

        time_scan = column_or_1d(time_scan)

        if time_physio is not None:
            time_physio = column_or_1d(time_physio)
            if time_physio.shape[0] != signal.shape[0]:
                raise ValueError(f"Signal has {signal.shape[0]} time points, "
                                 " whilst physiological times has"
                                 f" {time_physio.shape[0]}."
                                 )
        else:
            time_physio = np.arange(signal.shape[0])*1/self.physio_rate

        if self.transform not in ["demean", "zscore", "abs"]:
            raise ValueError(f"'{self.transform}' transform option passed, "
                             "but only 'mean' (default), 'zscore' or 'abs' "
                             "are allowed."
                             )
        if self.high_pass is not None:
            try:
                float(self.high_pass)
            except ValueError:
                raise ValueError(f" '{self.high_pass}' was provided "
                                 "as highpass frequency, but it should "
                                 "be a number")
        if self.low_pass is not None:
            try:
                float(self.low_pass)
            except ValueError:
                raise ValueError(f" '{self.low_pass}' was provided "
                                 "as lowpass frequency, but it should "
                                 "be a number")
        if self.high_pass is not None and self.low_pass is not None:
            if float(self.high_pass) > float(self.low_pass):
                raise ValueError("high pass frequency should be lower "
                                 "than the low pass frequency for a "
                                 "bandpass filtering"
                                 )
        # Decide how to handle data and loop through
        if self.columns is not None:
            if self.columns == "mean":
                signal = np.mean(signal, axis=1)
            else:
                columns = np.asarray(self.columns, dtype=float)
                if len(columns) != signal.shape[1]:
                    raise ValueError(f"supplied columns has {len(columns)},"
                                     f" but signal has {signal.shape[1]}"
                                     " channels."
                                     )
                signal = signal.dot(self.columns)
            # reshape again to 2D
            signal = signal.reshape(-1, 1)

        return signal, time_scan, time_physio

    def generate_report():
        # TODO
        raise NotImplementedError


class RetroicorPhysio(BasePhysio):
    """
     Physiological regressors using Retroicor.

    Parameters
    ----------
    physio_rate : float
        Sampling rate for the recording in Hz.
        This is needed for filtering to define the nyquist frequency.
    t_r : float
        Repetition time for the scanner (the usual T_R) in secs.
    delta: float
        minimum separation (in physio recording units) between
        events in signal to be considered peaks
    peak_rise: float
        relative height with respect to the 20th tallest events in signal
        to consider events as peak.
    order: int or array-like of shape (n_orders,)
        Fourier expansion for phases. If int, the fourier expansion is
        performed to that order, starting from 1. If an array is provided,
        each element will multiply the phases.
    transform : {"demean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "demean".
    high_pass : float, optional
        High-pass filtering frequency (in Hz). The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). The default is None.
    columns : List or array of n_channels elements, "mean" or None, optional
        It describes how to handle input signal channels. If a list, it will
        weight each channel and take the dot product. If "mean",
        the average across the channels. If None, it will consider each
        channel separately. The default is None.
    peak_correct : bool, optional
	Whether to apply an automatic Grubbs' test for peak outlier
        correction. The default is True.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 *,
                 physio_rate,
                 t_r,
                 delta,
                 peak_rise=0.5,
                 order=1,
                 transform="demean",
                 filtering=None,
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 peak_correct=True,
                 n_jobs=1):

        self.order = order
        self.delta = delta
        self.peak_rise = peak_rise
        self.peak_correct = peak_correct

        super().__init__(physio_rate=physio_rate,
                         t_r=t_r,
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

        # Compute peaks in signal
        peaks = compute_max_events(signal, self.peak_rise, self.delta)

        # Correct peaks
        if self.peak_correct:
            peaks, _, _ = correct_anomalies(peaks)

        # Compute phases according to peaks (changed to an interpolation)
        phases_in_peaks = 2*np.pi*np.arange(len(peaks))
        phases = interp1d(x=time_physio[peaks],
                          y=phases_in_peaks,
                          kind="linear",
                          fill_value="extrapolate")(time_physio)

        # Expand phases according to Fourier expansion order
        phases_fourier = [(m*phases).reshape(-1, 1)
                          for m in range(1, int(self.order)+1)]
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

    def _validate_inputs(self,
                         signal,
                         time_scan,
                         time_physio):

        signal, time_scan, time_physio = super()._validate_inputs(signal,
                                                                  time_scan,
                                                                  time_physio)

        # cast to integer
        order = int(self.order)

        if order < 1:
            raise ValueError("Fourier expansion should be "
                             " a positive integer")

        return signal, time_scan, time_physio


class RVPhysio(BasePhysio):
    """
     Physiological regressors for variations in breathing rate/volume.

    Parameters
    ----------
    physio_rate : float
        Sampling rate for the recording in Hz.
        This is needed for filtering to define the nyquist frequency.
    t_r : float
        Repetition time for the scanner (the usual T_R) in secs.
    time_window : float
        Time window (in secs) around the T_R from which computing variations
        (standard deviation of signal). The default is 6 secs.
    transform : {"demean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "demean".
    high_pass : float, optional
        High-pass filtering frequency (in Hz). The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). The default is None.
    columns : List or array of n_channels elements, "mean" or None, optional
        It describes how to handle input signal channels. If a list, it will
        weight each channel and take the dot product. If "mean",
        the average across the channels. If None, it will consider each
        channel separately. The default is None.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 *,
                 physio_rate,
                 t_r,
                 time_window=6.0,
                 transform="demean",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 n_jobs=1):

        self.time_window = time_window

        super().__init__(physio_rate=physio_rate,
                         t_r=t_r,
                         transform=transform,
                         high_pass=high_pass,
                         low_pass=low_pass,
                         columns=columns,
                         n_jobs=n_jobs)

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

        N = len(time_scan)
        rv_values = np.zeros(N)

        for ii in range(N):

            # TODO: See border effects and maybe a way to optimize this
            t_i = time_scan[ii] - self.time_window/2.
            t_f = time_scan[ii] + self.time_window/2.

            if t_i < time_physio[0]:
                t_i = time_physio[0]
            if t_f > time_physio[-1]:
                t_f = time_physio[-1]

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

        t_rrf = np.arange(0, 28, self.t_r)
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
    t_r : float
        Repetition time for the scanner (the usual T_R) in secs.
    delta: float
        minimum separation (in physio recording units) between
        events in signal to be considered peaks
    peak_rise: float
        relative height with respect to the 20th tallest events in signal
        to consider events as peak.
    time_window : float
        Time window (in secs) around the T_R from which computing variations
        (time differences between signal events ). The default is 6 secs.
    transform : {"demean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "demean".
    high_pass : float, optional
        High-pass filtering frequency (in Hz). The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz).  The default is None.
    columns : List or array of n_channels elements, "mean" or None, optional
        It describes how to handle input signal channels. If a list, it will
        weight each channel and take the dot product. If "mean",
        the average across the channels. If None, it will consider each
        channel separately. The default is None.
    peak_correct : bool, optional
	Whether to apply an automatic Grubbs' test for peak outlier
        correction. The default is True.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 *,
                 physio_rate,
                 t_r,
                 delta,
                 peak_rise=0.5,
                 time_window=6.0,
                 transform="demean",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 peak_correct=True,
                 n_jobs=1):

        self.delta = delta
        self.peak_rise = peak_rise
        self.time_window = time_window
        self.peak_correct = peak_correct

        super().__init__(physio_rate=physio_rate,
                         t_r=t_r,
                         transform=transform,
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

        # TODO: Add checks specific to this task
        # Compute peaks in signal
        peaks = compute_max_events(signal, self.peak_rise, self.delta)

        # Correct peaks
        if self.peak_correct:
            peaks, _, _ = correct_anomalies(peaks)

        peaks = peaks.astype(int)

        # Compute times of maximum event peaks
        time_peaks = time_physio[peaks]

        N = len(time_scan)

        hv_values = np.zeros(N)
        for ii in range(N):
            # TODO: See border effects and maybe a way to optimize this
            t_i = time_scan[ii] - self.time_window/2
            t_f = time_scan[ii] + self.time_window/2

            if t_i < time_physio[0]:
                t_i = time_physio[0]
            if t_f > time_physio[-1]:
                t_f = time_physio[-1]

            mask_times = (time_peaks >= t_i) & (time_peaks <= t_f)
            if sum(mask_times) < 2:
                hv_values[ii] = 0
            else:
                hv_values[ii] = 60./np.mean(np.diff(time_peaks[mask_times]))

        # return zscores of these values
        hv_values = zscore(hv_values)

        def CRF(t):
            """Cardiac Response Function (eq. 5 Chang 2009)."""
            crf = 0.6*(t**2.7)*np.exp(-t/1.6)
            crf -= 16./(np.sqrt(2*np.pi*9))*np.exp(-(0.5/9)*(t-12.)**2)
            return crf

        t_crf = np.arange(0, 28, self.t_r)
        crf = np.apply_along_axis(CRF, axis=0, arr=t_crf)
        # Convolve rv values with response function
        regressors = np.convolve(hv_values, crf)
        regressors = regressors[:N]  # Trim excess
        return regressors


class DownsamplePhysio(BasePhysio):
    """
     Physiological regressors by downsampling

     As in Verstynen 2011, raw physiological data is downsample
     to the scanner resolution.

    Parameters
    ----------
    physio_rate : float
        Sampling rate for the recording in Hz.
        This is needed for filtering to define the nyquist frequency.
    t_r : float
        Repetition time for the scanner (the usual T_R) in secs.
    transform : {"demean", "zscore", "abs"}, optional
        Transform data before filtering. The default is "demean".
    high_pass : float, optional
        High-pass filtering frequency (in Hz). The default is None.
    low_pass : float, optional
        Low-pass filtering frequency (in Hz). The default is None.
    columns : List or array of n_channels elements, "mean" or None, optional
        It describes how to handle input signal channels. If a list, it will
        weight each channel and take the dot product. If "mean",
        the average across the channels. If None, it will consider each
        channel separately. The default is None.
    kind : str or int, optional
        This is just the kind of interpolation to use. The allowed values are
        those in interp1d function of scipy. Just copying its documentation,
        the string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’,
        ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
        or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’. which refer to
        a spline interpolation of zeroth, first, second or third order;
        ‘previous’ and ‘next’ simply return the previous or
        next value of the point; ‘nearest-up’ and ‘nearest’ differ
        when interpolating half-integers (e.g. 0.5, 1.5) in that
        ‘nearest-up’ rounds up and ‘nearest’ rounds down.
        Our default value is 'cubic'.
    n_jobs : int, optional
        Number of jobs to consider in parallel. The default is 1.
    """

    def __init__(self,
                 *,
                 physio_rate,
                 t_r,
                 transform="demean",
                 high_pass=None,
                 low_pass=None,
                 columns=None,
                 kind="cubic",
                 n_jobs=1):
        # common arguments for all classess
        self.physio_rate = physio_rate
        self.t_r = t_r
        self.transform = transform
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.columns = columns
        self.kind = kind
        self.n_jobs = n_jobs

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        """
        Generate regressors by downsampling.

        Just downsample the data to the scanner resolution by using the
        provided scanner times.

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

        downsampled_signal = interp1d(time_physio,
                                      signal,
                                      kind=self.kind,
                                      fill_value='extrapolate')(time_scan)
        return downsampled_signal
