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
    signal, info_dict = unit.load_cmrr_data(fn, 'ECG', meta_info, True)
    signal, info_dict = unit.load_cmrr_data(fn, 'ECG', meta_info, False)

#def test_getData():
#
#  fn = '../data/sample1/Physio_sample1_ECG.log'
#  dat, nch, sr = unit.getData(fn, 21889410, 22113530, False)
#  assert nch == 4
#  assert sr == 1
#  assert dat.shape == (224121, 5)
#  assert (dat[0, 0], dat[10000, 2], dat[224120, 4]) == (0.0, 2002.0, 2046.0)
#
#def test_interpMissingData():
#
#  dat = np.zeros((10, 2))
#  dat[:, 0] = range(10)
#  dat[:, 1] = [1, 3, 0, 7, 6, 0, 4, 0, 4, 2]
#  dat = unit.interpMissingData(dat)
#  assert dat[2, 1] == 5
#  assert dat[5, 1] == 5
#  assert dat[7, 1] == 4
#
#def test_genJSON():
#
#  unit.genJSON(500, 1, 2, 10, 100, 30, 1000, 4, [0.75, 3.5], [0.01, 0.5])
#  assert os.path.isfile('meta.txt')
#  meta = json.load(open('meta.txt'))
#  assert meta['samplingRate'][0]['freq'] == 500
#  assert meta['samplingRate'][1]['rate'] == 1
#  assert meta['samplingRate'][2]['rate'] == 2
#  assert meta['samplingRate'][3]['rate'] == 10
#  assert meta['MRacquisition'][0]['Volumes'] == 100
#  assert meta['MRacquisition'][0]['Slices'] == 30
#  assert meta['MRacquisition'][0]['TR'] == 1000
#  assert meta['ECGground'][0]['Channel'] == 4
#  assert meta['frequencyRanges'][0]['Cardiac'] == [0.75, 3.5]
#  assert meta['frequencyRanges'][0]['Respiratory'] == [0.01, 0.5]
#  os.remove('meta.txt')
#
#def test_procInput():
#
#  path = '/Users/andrew/Fellowship/projects/brainhack-physio-project/data/sample2/'
#  infoFile = 'Physio_sample2_Info.log'
#  pulsFile = 'Physio_sample2_PULS.log'
#  respFile = 'Physio_sample2_RESP.log'
#  ecgFile = 'Physio_sample2_ECG.log'
#  unit.procInput(path, infoFile, pulsFile, respFile, ecgFile, showSignals=False)
#  assert os.path.isfile('meta.txt')
#  assert os.path.isfile('signal.npy')
#  os.remove('meta.txt')
#  os.remove('signal.npy')

################################################################################
# Run tests                                                                    #
################################################################################

test_get_lines()
test_load_cmrr_info()
test_load_cmrr_data()
#test_getData()
#test_interpMissingData()
#test_genJSON()
#test_procInput()
