import numpy as np
import os
import json
from pathlib import Path

import niphlem.input_data as unit


def test_get_lines():
    """
    Test that get_lines function returns correct first, middle, and last lines
    of a sample file
    """
    fn = Path(__file__).parent.as_posix() + '/datasets/sample1/Physio_sample1_Info.log'
    lines = unit.get_lines(fn)
    # Test first, random middle, and last line
    assert lines[0] == 'UUID        = 6ec4c7ab-b798-4eec-989c-9458617d425c'
    assert lines[10000] == '   312      19         21985776         21985799     0'
    assert lines[23627] == 'LastTime    = 22113530'

def test_load_cmrr_info():
    """
    Test that load_cmrr_info returns correct meta info and and first, middle,
    and last time ticks of a sample file
    """
    fn = Path(__file__).parent.as_posix() + '/datasets/sample1/Physio_sample1_Info.log'
    traces, meta_info = unit.load_cmrr_info(fn)
    # test meta_info
    assert meta_info['uuid'] == '6ec4c7ab-b798-4eec-989c-9458617d425c'
    assert meta_info['scan_date'] == '20190813_151159'
    assert meta_info['log_version'] == 'EJA_1'
    assert meta_info['n_vols'] == 738
    assert meta_info['n_slices'] == 32
    assert meta_info['n_echoes'] == 1
    assert meta_info['init_physio'] == 21889410
    assert meta_info['end_physio'] == 22113530
    assert meta_info['init_scan'] == 21892140
    assert meta_info['end_scan'] == 22113518
    assert meta_info['repetition_time'] == 300.0
    # test traces
    assert np.shape(traces) == (2, 738, 32, 1)
    assert traces[0, 0, 0, 0] == 21892140
    assert traces[1, 300, 10, 0] == 21982381
    assert traces[0, 737, 31, 0] == 22113422

def test_load_cmrr_data():
    """
    Test that load_cmrr_data returns correct meta info and and first, middle,
    and last time ticks of a sample file
    """

    # acquire meta_info
    fn = Path(__file__).parent.as_posix() + '/datasets/sample1/Physio_sample1_Info.log'
    traces, meta_info = unit.load_cmrr_info(fn)

    fn = Path(__file__).parent.as_posix() + '/datasets/sample1/Physio_sample1_ECG.log'
    # test dictionary
    signal, info_dict = unit.load_cmrr_data(fn, 'ECG', meta_info, True)
    assert info_dict['ECG']['n_channels'] == 4
    assert info_dict['ECG']['sample_rate'] == 1
    # test signal with scan sync
    assert np.shape(signal) == (221379, 4)
    assert (signal[0, 0], signal[1000, 1], signal[100000, 2], signal[221378, 3]) == (1306.0, 1756.0, 2111.0, 2068.0)

    # test signal without scan sync
    signal, info_dict = unit.load_cmrr_data(fn, 'ECG', meta_info, False)
    assert np.shape(signal) == (224121, 4)
    assert (signal[0, 0], signal[1000, 1], signal[100000, 2], signal[224120, 3]) == (2970.0, 2249.0, 2230.0, 2046.0)

def test_proc_input():
    """
    Test that proc_input generates the desired files, adds the correct
    frequency information, and assembles the signals
    """

    path = Path(__file__).parent.as_posix() + '/datasets/sample1/'
    info_file = 'Physio_sample1_Info.log'
    puls_file = 'Physio_sample1_PULS.log'
    resp_file = 'Physio_sample1_RESP.log'
    ecg_file = 'Physio_sample1_ECG.log'
    meta_file = 'test_meta.json'
    sig_file = 'test_signal.npy'
    unit.proc_input(path,
                    info_file,
                    puls_file,
                    resp_file,
                    ecg_file,
                    meta_filename=meta_file,
                    sig_filename=sig_file,
                    show_signals=False)

    # test presence of generated files
    assert os.path.isfile(meta_file)
    assert os.path.isfile(sig_file)
    # test attributes added to meat file by proc_input
    meta = json.load(open(meta_file))
    assert meta['frequency_info']['sampling_rate'] == 400
    assert meta['frequency_info']['cardiac_range'] == [0.75, 3.5]
    assert meta['frequency_info']['respiratory_range'] == [0.01, 0.5]
    # test signal
    signal = np.load(sig_file)
    assert np.shape(signal) == (221379, 6)
    assert (signal[0, 0], signal[10, 1], signal[100, 2], signal[1000, 3], signal[10000, 4], signal[221378, 5]) == (1306.0, 1724.0, 1956.0, 2056.0, 2304.5, 1025.625)
    # clean up
    os.remove(meta_file)
    os.remove(sig_file)


"""
Run tests
"""

test_get_lines()
test_load_cmrr_info()
test_load_cmrr_data()
test_proc_input()
