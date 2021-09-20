import sys
import numpy as np
import json
import matplotlib.pyplot as mpl

###############################################################################
# imports lines from input file                                               #
# in:  fn - filename                                                          #
# out: lines - lines of file                                                  #
###############################################################################

def getLines(fn):

  lines = []
  try:
    fh = open(fn, 'r')
  except OSError:
    msg = 'Cannot open input file '+fn
    raise Warning(msg)
  else:
    # Get lines of file
    for l in fh:
      lines.append(l.rstrip('\n'))
    fh.close()

  return lines

###############################################################################
# imports data from info file                                                 #
# in:  fn - filename                                                          #
#      cols - list of columns to extract data from                            #
# out: x - array of data                                                      #
#      t0 - initial time                                                      #
#      tN - end time                                                          #
#      nVol - number of volumes acquired                                      #
#      nSlice - number of slices per acquisition                              #
#      TR - repetition time                                                   #
###############################################################################

def getInfoData(fn, cols):

  lines = getLines(fn)
  # Get parameters for meta file and lines containing data
  stt = 0; stp = 0
  for i in range(len(lines)):
    y = lines[i].split()
    if len(y) == 0:
      continue
    if y[0] == 'NumVolumes':
      nVol = int(y[2])
    if y[0] == 'NumSlices':
      nSlice = int(y[2])
    if y[0] == 'FirstTime':
      t0 = int(y[2])
    if y[0] == 'LastTime':
      tN = int(y[2])
    # Inherent assumption that all lines starting with a number are data
    if stt == 0:
      try:
        int(y[0])
        stt = i
      except ValueError:
        continue
    if stp == 0:
      try:
        int(y[0])
        continue
      except ValueError:
        stp = i

  # Pull data into numpy array
  x = np.zeros((stp-stt, len(cols)))
  for i in range(stt, stp):
    y = lines[i].split()
    for j in range(len(cols)):
      x[i-stt, j] = y[cols[j]]
  TR = round((x[-1, 3]-x[0, 2])/nVol)

  return x, t0, tN, nVol, nSlice, TR

###############################################################################
# imports data from file                                                      #
# in:  fn - filename                                                          #
#      t0 - initial time                                                      #
#      tN - end time                                                          #
#      interp - flag to interpolate missing values (upsamples to ECG freq.)   #
# out: x - array of data                                                      #
#      nch - number of channels                                               #
#      sr - sampling rate                                                     #
###############################################################################

def getData(fn, t0, tN, interp):

  lines = getLines(fn)

  # Get sampling rate and start of data
  stt = 0
  for i in range(len(lines)):
    y = lines[i].split()
    if len(y) == 0:
      continue
    if y[0] == 'SampleTime':
      sr = int(y[2])
    # Inherent assumption that all lines starting with a number are data
    if stt == 0:
      try:
        int(y[0])
        stt = i
      except ValueError:
        continue

  # Get number of channels (not particularly efficient, but thorough...)
  if y[1] == 'PULS' or y[1] == 'RESP':
    nch = 1
  else:
    nch = 0
    for i in range(stt, len(lines)):
      y = lines[i].split()
      j = int(y[1][-1])
      if j > nch:
        nch = j

  # Pull data into numpy array
  x = np.zeros((tN-t0+1, nch+1))
  x[:, 0] = range(0, tN-t0+1)
  if nch == 1:
    # Use separate loop for single channel to avoid repeated ifs for channel #
    for i in range(stt, len(lines)):
      y = lines[i].split()
      k = int(int(y[0]) - t0)
      x[k, 1] = float(y[2])
  else:
    for i in range(stt, len(lines)):
      y = lines[i].split()
      j = int(y[1][-1])
      k = int(int(y[0]) - t0)
      x[k, j] = float(y[2])
  if interp:
    x = interpMissingData(x)

  return x, nch, sr

###############################################################################
# interpolates the missing data points                                        #
# in/out:  dat - array of data                                                #
###############################################################################

def interpMissingData(dat):

  dat = dat.copy() # add copy
  nch = np.ma.size(dat, 1)-1        # number of channels
  for j in range(1, nch+1):

    # extrapolate at beginning of series if needed
    for i in range(0, len(dat)):
      # move on if no empty values at beginning of series
      if dat[i, j] != 0:
        break
      # find nearest non-zero neighbor above
      for k in range(1, len(dat)-1):
        y1 = dat[i+k, j]
        if y1 != 0:
          x1 = dat[i+k, 0]
          break
      # find next nearest non-zero neighbor above
      for kk in range(k+1, len(dat)-1):
        y2 = dat[i+kk, j]
        if y2 != 0:
          x2 = dat[i+kk, 0]
          break
      dat[i, j] = y1+(dat[i, 0]-x1)*(y2-y1)/(x2-x1)

    # extrapolate at end of series if needed
    for i in range(len(dat)-1, 0, -1):
      # move on to interpolating interior values if no empty values at end of series
      if dat[i, j] != 0:
        break
      # find nearest non-zero neighbor below
      for k in range(1, len(dat)-1):
        y1 = dat[i-k, j]
        if y1 != 0:
          x1 = dat[i-k, 0]
          break
      # find next nearest non-zero neighbor below
      for kk in range(k+1, len(dat)-1):
        y2 = dat[i-kk, j]
        if y2 != 0:
          x2 = dat[i-kk, 0]
          break
      dat[i, j] = y1+(dat[i, 0]-x1)*(y2-y1)/(x2-x1)

    for i in range(1, len(dat)-1):
      if dat[i, j] == 0:
        # find nearest non-zero neighbor below
        for k in range(1, i):
          y1 = dat[i-k, j]
          if y1 != 0:
            x1 = dat[i-k, 0]
            break
        # find nearest non-zero neighbor above
        for k in range(1, len(dat)-i):
          y2 = dat[i+k, j]
          if y2 != 0:
            x2 = dat[i+k, 0]
            break
        dat[i, j] = y1+(dat[i, 0]-x1)*(y2-y1)/(x2-x1)

  return dat

###############################################################################
# generates the json file                                                     #
# in:  sFreq - sampling frequency (1/tic in ms)                               #
#      srECG/PULS/RESP - sampling rates for ECG, PULS, and RESP               #
#      nVol,nSlice,TR - volumes, slices, and TR for fMRI acquisition          #
#      gCh - ECG ground channel                                               #
#      card/respRange - cardiac and respiratory range of frequencies          #
###############################################################################

def genJSON(sFreq, srECG, srPULS, srRESP, nVol, nSlice, TR, gCh, cardRange, respRange):

  meta = {}
  meta['samplingRate'] = []
  meta['samplingRate'].append({
    'freq': sFreq
  })
  meta['samplingRate'].append({
    'channel': 'ECG',
    'rate': srECG
  })
  meta['samplingRate'].append({
    'channel': 'PULS',
    'rate': srPULS
  })
  meta['samplingRate'].append({
    'channel': 'RESP',
    'rate': srRESP
  })
  meta['MRacquisition'] = []
  meta['MRacquisition'].append({
    'Volumes': nVol,
    'Slices': nSlice,
    'TR': TR,
  })
  meta['ECGground'] = []
  meta['ECGground'].append({
    'Channel': gCh
  })
  meta['frequencyRanges'] = []
  meta['frequencyRanges'].append({
    'Cardiac': cardRange,
    'Respiratory': respRange
  })

  with open('meta.txt', 'w') as outfile:
    json.dump(meta, outfile)

###############################################################################
# main code                                                                   #
# In:  path - path to folder containing physio files, assumes common location #
#      infoFile - Info file name                                              #
#      pulsFile - PULS file name                                              #
#      respFile - RESP file name                                              #
#      ecgFile - ECG file name                                                #
#      showSignals - flag to plot the signals (default False)                 #
# Out: (meta.txt) - json file containing meta data                            #
#      (signal.csv) - file containing condensed signal data                   #
###############################################################################

def procInput(path, infoFile, pulsFile, respFile, ecgFile, showSignals=False):

  cardiacRange = [0.75, 3.5]  # Hz
  respRange = [0.01, 0.5]     # Hz
  # TODO: Take this as an input or extract somehow
  sFreq = 400                 # Hz

  # ensure path ends in /
  if path[-1] != '/':
    path = path+'/'

  # get data from INFO file
  Info, t0, tN, nVol, nSlice, TR = getInfoData(path+infoFile, range(4))
  # get data from PULS file
  PULS, nch, srPULS = getData(path+pulsFile, t0, tN, True)
  # get data from RESP file
  RESP, nch, srRESP = getData(path+respFile, t0, tN, True)
  # get data from ECG file
  ECG, nch, srECG = getData(path+ecgFile, t0, tN, True)
  # generate JSON dictionary
  genJSON(sFreq, srECG, srPULS, srRESP, nVol, nSlice, TR, nch, cardiacRange, respRange)
  # store aligned signals in a single matrix, save to signal.npy
  signal = np.zeros((len(ECG), nch+3))
  signal[:, 0:(nch+1)] = ECG
  signal[:, nch+1] = PULS[:, 1]
  signal[:, nch+2] = RESP[:, 1]
  np.save('signal', signal)

  # plot signals if desired
  if showSignals:
    mpl.plot(PULS[:, 0], PULS[:, 1])
    mpl.show()
    mpl.plot(RESP[:, 0], RESP[:, 1])
    mpl.show()
    mpl.plot(ECG[:, 0], ECG[:, 1], 'b')
    mpl.plot(ECG[:, 0], ECG[:, 2], 'r')
    mpl.plot(ECG[:, 0], ECG[:, 3], 'g')
    mpl.plot(ECG[:, 0], ECG[:, 4], 'k')
    mpl.show()

###############################################################################

#path = '/Users/andrew/Fellowship/projects/brainhack-physio-project/data/sample2/'
#infoFile = 'Physio_sample2_Info.log'
#pulsFile = 'Physio_sample2_PULS.log'
#respFile = 'Physio_sample2_RESP.log'
#ecgFile = 'Physio_sample2_ECG.log'
#procInput(path, infoFile, pulsFile, respFile, ecgFile, showSignals=True)
