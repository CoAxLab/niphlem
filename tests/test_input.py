import sys
# add path to code folder in first location
sys.path.insert(1, '../code')
import input as unit
import numpy as np

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

################################################################################
# Run tests                                                                    #
################################################################################

test_getLines()
test_getInfoData()
