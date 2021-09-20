import sys
# add path to code folder in first location
sys.path.insert(1, '../code')
import input as unit
import numpy as np
import os
import json

def test_getLines():

  fn = '../data/sample1/Physio_sample1_Info.log'
  lines = unit.getLines(fn)
  # Test first, random middle, and last line
  # TODO: may sure things like new line or whatever don't break this
  assert lines[0] == 'UUID        = 6ec4c7ab-b798-4eec-989c-9458617d425c'
  assert lines[10000] == '   312      19         21985776         21985799     0'
  assert lines[23627] == 'LastTime    = 22113530'

def test_getInfoData():

  fn = '../data/sample1/Physio_sample1_Info.log'
  Info, t0, tN, nVol, nSlice, TR = unit.getInfoData(fn, range(4))
  assert np.shape(Info) == (23616, 4)
  assert (Info[0,0], Info[1000,1], Info[10000,2], Info[-1,3]) == (0.0, 6.0, 21985886.0, 22113518.0)
  assert t0 == 21889410
  assert tN == 22113530
  assert nVol == 738
  assert nSlice == 32
  assert TR == 300

def test_getData():

  fn = '../data/sample1/Physio_sample1_ECG.log'
  dat, nch, sr = unit.getData(fn, 21889410, 22113530, False)
  assert nch == 4
  assert sr == 1
  assert dat.shape == (224121, 5)
  assert (dat[0, 0], dat[10000, 2], dat[224120, 4]) == (0.0, 2002.0, 2046.0)

def test_interpMissingData():

  dat = np.zeros((10, 2))
  dat[:, 0] = range(10)
  dat[:, 1] = [1, 3, 0, 7, 6, 0, 4, 0, 4, 2]
  dat = unit.interpMissingData(dat)
  assert dat[2, 1] == 5
  assert dat[5, 1] == 5
  assert dat[7, 1] == 4

def test_genJSON():

  unit.genJSON(500, 1, 2, 10, 100, 30, 1000, 4, [0.75, 3.5], [0.01, 0.5])
  assert os.path.isfile('meta.txt')
  meta = json.load(open('meta.txt'))
  assert meta['samplingRate'][0]['freq'] == 500
  assert meta['samplingRate'][1]['rate'] == 1
  assert meta['samplingRate'][2]['rate'] == 2
  assert meta['samplingRate'][3]['rate'] == 10
  assert meta['MRacquisition'][0]['Volumes'] == 100
  assert meta['MRacquisition'][0]['Slices'] == 30
  assert meta['MRacquisition'][0]['TR'] == 1000
  assert meta['ECGground'][0]['Channel'] == 4
  assert meta['frequencyRanges'][0]['Cardiac'] == [0.75, 3.5]
  assert meta['frequencyRanges'][0]['Respiratory'] == [0.01, 0.5]
  os.remove('meta.txt')

def test_procInput():

  path = '/Users/andrew/Fellowship/projects/brainhack-physio-project/data/sample2/'
  infoFile = 'Physio_sample2_Info.log'
  pulsFile = 'Physio_sample2_PULS.log'
  respFile = 'Physio_sample2_RESP.log'
  ecgFile = 'Physio_sample2_ECG.log'
  unit.procInput(path, infoFile, pulsFile, respFile, ecgFile, showSignals=False)
  assert os.path.isfile('meta.txt')
  assert os.path.isfile('signal.npy')
  os.remove('meta.txt')
  os.remove('signal.npy')

################################################################################
# Run tests                                                                    #
################################################################################

test_getLines()
test_getInfoData()
test_getData()
test_interpMissingData()
test_genJSON()
#test_procInput()
