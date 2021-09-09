#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:27:12 2021

@author: javi
"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin


class BasePhysio(BaseEstimator, TransformerMixin):

    def __init__(self,
                 sampling_rate,
                 filtering="butter",
                 high_pass=None,
                 low_pass=None,
                 electrodes=None,
                 n_jobs=1):

        # common arguments for all classess
        self.filtering = filtering
        self.high_pass = high_pass
        self.low_pass = low_pass
        self.sampling_rate = sampling_rate
        self.electrodes = electrodes
        self.n_jobs = n_jobs

    def compute_regressors(self,
                           signal,
                           time_physio,
                           time_scan):

        # TODO: Add checks for input arguments and data

        # Decide how to handle data and loop through
        if self.electrodes:
            if self.electrodes == "mean":
                signal = signal.dot(self.electrodes)
            else:
                signal = signal.dot(self.electrodes)

        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)

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
                 order,
                 sampling_rate,
                 delta,
                 peak_rise=0.05,
                 filtering="butter",
                 high_pass=None,
                 low_pass=None,
                 electrodes=None,
                 n_jobs=1):

        self.order = order
        self.delta = delta
        self.peak_rise = peak_rise

        super().__init__(filtering=self.filtering,
                         high_pass=self.high_pass,
                         low_pass=self.low_pass,
                         sampling_rate=self.sampling_rate,
                         electrodes=self.electrodes,
                         n_jobs=self.n_jobs)

    def _process_regressors(self, signal, time_physio, time_scan):

        # Compute peaks in signal
        peaks = compute_max_events(signal, peak_rise, delta)
        # Phases
        phases = compute_phaes(time, peaks)
        # Expand phases according to Fourier expansion order
        phases_expanded = expand_phases(phases, self.order)
        # Downsample phases
        phases_scan = downsample_phases(phases_expanded,
                                        time_physio,
                                        time_scan)
        #
        regressors = sin_cos_phases(phases_scan)
        return regressors


class RVPhysio(BasePhysio):

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
       
    N = len(scan_time)

    rv_values = np.zeros(N)

    for ii in range(N):
        
        # The 2 to account for this and the previous TR window
        t_i = scan_time[ii] - 2.0*scan_time[0]
        t_f = scan_time[ii] + scan_time[0]
    
        # Take points of recording between these times
        mask_times = (physio_time >= t_i) & (physio_time <= t_f)
        rv_values[ii] = np.std(signal[mask_times])
    
    #return zscores of these values
    zscores  = (rv_values - np.nanmean(rv_values))/np.nanstd(rv_values)
    return zscores


class HVPhysio(BasePhysio):

    pass


def compute_rv_values(signal, physio_time, scan_time):
    """
    
    Compute rate variations for respiration signal computed by taking 
    the standard deviation of the raw respiratory waveform over the 3 TR time 
    interval defined by the (k − 1)th, kth, and (k + 1)th TRs. 
    Thus, RV(k) is essentially a sliding-window measure related 
    to the inspired volume over time. (Chang et al 2009)
    
    Parameters
    ----------
    signal : str
        The signal values
    ticks : int
        the times of each TR

    Returns
    -------
    list
    
        standarised rv_values
    
    """
    

def compute_hv_values(signal, physio_time, scan_time, max_pks):
    """
    
    Compute rate variations for cardiac signal as the average of 
    the time differences between pairs of adjacent ECG triggers contained in 
    the 3 TR window defined by the (k − 1)th, kth, and (k + 1)th TRs, 
    and dividing the result into 60 to convert it to units of beats-per-minute. 
    (Chang 2009)
       
    Parameters
    ----------
    signal : str
        The signal values
    ticks : int
        the times of each TR

    Returns
    -------
    list
    
        standarised hv_values
    
    """
    
    maxpos = max_pks.astype(int)
    
    # Compute times of maximum event peaks
    maxpos_time = physio_time[maxpos]
    
    N = len(scan_time)
    #First TR?
    hv_values = np.zeros(N)
        
    for ii in range(N):
        
        # The 2 to account for this and the previous TR window
        t_i = scan_time[ii] - 2.0*scan_time[0]
        t_f = scan_time[ii] + scan_time[0]
        
        #Take previous peaks?
        peaks_tr = np.where((maxpos_time >= t_i) & (maxpos_time <= t_f))[0]
        
        hv_values[ii] = 60./np.mean(np.diff(maxpos_time[peaks_tr]))
        
        
   #return zscores of these values
    zscores  = (hv_values - np.nanmean(hv_values))/np.nanstd(hv_values)
    return zscores


def compute_phases(time, max_peaks):
    """
    This function compute the phase between successive peaks
    events as provided by compute_max_events function.
    The values between peaks are mapped to being in the range [0, 2 pi]
    (eq. 2 Glover 2000)
    """

    n_maxs = max_peaks.size
    phases = np.zeros_like(time, dtype=np.float)
    N = len(time)

    for ii in range(n_maxs):

        #Look at the tails
        if ii == n_maxs-1:
            i_o, i_f =  int(max_peaks[ii]), int(max_peaks[0])
            t_o, t_f = time[i_o], time[::-1][0] + time[i_f]

            phases[i_o:] = 2*np.pi*(time[i_o:N] - t_o)/(t_f-t_o)
            phases[:i_f] = 2*np.pi*(time[::-1][0] + time[:i_f] - t_o)/(t_f-t_o)
        else:
            i_o, i_f =  int(max_peaks[ii]), int(max_peaks[ii+1])
            t_o, t_f = time[i_o], time[i_f]
            phases[i_o:i_f] = 2*np.pi*(time[i_o:i_f] - t_o)/(t_f-t_o)

    return phases

def expand_phases(phases, M):
    phases_mat =  [(m+1)*phases for m in range(M)]

    if len(phases_mat) == 1:
        phases_mat = np.array(phases_mat).reshape(-1, 1)
    else:
        phases_mat =  np.column_stack(phases_mat)
    return phases_mat

def downsample_phases(phases_expanded, time, new_time):

    from scipy.interpolate import interp1d

    #TODO: This using np.apply_along_axis
    downsampled_phases = []

    for jj in range(phases_expanded.shape[1]):
        original_phases = phases_expanded[:,jj]

        interp = interp1d(time,
                          original_phases,
                          kind='linear',
                          fill_value = 'extrapolate')


        #new_phases = resample(original_phases, time, new_time)
        new_phases = interp(new_time)
        downsampled_phases.append(new_phases)

    downsampled_phases = np.column_stack(downsampled_phases)
    return downsampled_phases


def sin_cos_phases(phases_mat):
    regressors_mat =  []

    #TODO: This using np.apply_along_axis
    for jj in range(phases_mat.shape[1]):
        sin_signal, cos_signal, = np.sin(phases_mat[:,jj]), np.cos(phases_mat[:,jj])
        regressors_mat.append(np.column_stack((sin_signal, cos_signal)))

    regressors_mat =  np.column_stack(regressors_mat)
    return regressors_mat


def zscore(x, axis=1, nan_omit=True):
    if nan_omit:
        mean = np.nanmean
        std = np.nanstd
    else:
        mean = np.mean
        std = np.std

    zscores = (x - mean(hv_values))/std(x)
    return zscores
