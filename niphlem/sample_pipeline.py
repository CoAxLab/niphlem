"""
Sample pipeline using repository data
"""

import os
import numpy as np
import input_data as ni
import clean as nc
import events as ne
#import niphlem.input_data as ni
#import niphlem.clean as nc
#import niphlem.events as  ne

# required user defined parameters
path = os.path.dirname(os.path.realpath(__file__)) + '/tests/datasets/sample1/'
info_file = 'Physio_sample1_Info.log'
ecg_file = 'Physio_sample1_ECG.log'
sampling_frequency = 400
signal_file = 'sample_signal.csv'
peak_file = 'sample_peaks.csv'

# optional user defined parameters
meta_filename = 'sample_meta.json'
cardiac_range = [0.6, 5.0]
peak_rise = 0.75
delta = 200
alpha = 0.05

# basic data processing from input file (empty slots for PULS and RESP data)
_, _, ECG = ni.proc_input(path=path,
                          info_file=info_file,
                          ecg_file=ecg_file,
                          sampling_frequency=sampling_frequency,
                          meta_filename=meta_filename)
ECG_filt = nc._transform_filter(data=ECG,
                          ground_ch=1,
                          transform='demean',
                          filtering='butter',
                          average_signal=True,
                          high_pass=cardiac_range[0],
                          low_pass=cardiac_range[1],
                          sampling_rate=sampling_frequency,
                          save_name=signal_file)
peaks = ne.compute_max_events(signal=ECG_filt,
                          peak_rise=peak_rise,
                          delta=delta)
peaks_diffs, max_idx, min_idx = ne.correct_anomalies(peaks=peaks,
                          alpha=alpha,
                          save_name=peak_file)
