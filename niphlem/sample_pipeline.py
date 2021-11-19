"""
Sample pipeline using repository data
"""

import os
import input_data as ni
import clean as nc
#import niphlem.input_data as ni
#import niphlem.clean as nc
#import niphlem.events

path = os.path.dirname(os.path.realpath(__file__)) + '/tests/datasets/sample1/'
info_file = 'Physio_sample1_Info.log'
puls_file = 'Physio_sample1_PULS.log'
resp_file = 'Physio_sample1_RESP.log'
ecg_file = 'Physio_sample1_ECG.log'
sampling_frequency = 400
meta_filename = 'sample_meta.json'
cardiac_range=[0.6, 5.0]
PULS, RESP, ECG = ni.proc_input(path=path,
                                info_file=info_file,
                                puls_file=puls_file,
                                resp_file=resp_file,
                                ecg_file=ecg_file,
                                sampling_frequency=sampling_frequency,
                                meta_filename=meta_filename)
ECG = nc._transform_filter(data=ECG,
                           ground_ch=1,
                           transform='demean',
                           filtering='butter',
                           high_pass=cardiac_range[0],
                           low_pass=cardiac_range[1],
                           sampling_rate=sampling_frequency)
