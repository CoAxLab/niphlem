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
    fh = open(fn,'r')
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

def getInfoData(fn,cols):

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
  x = np.zeros((stp-stt,len(cols)))
  for i in range(stt,stp):
    y = lines[i].split()
    for j in range(len(cols)):
      x[i-stt,j] = y[cols[j]]
  TR = round((x[-1,3]-x[0,2])/nVol)

  return x,t0,tN,nVol,nSlice,TR

###############################################################################
# imports data from input file                                                #
# in:  fn - filename                                                          #
#      cols - list of columns to extract data from                            #
#      t0 - initial time                                                      #
# out: x - array of data                                                      #
#      sr - sampling rate                                                     #
###############################################################################

def getData(fn,cols,t0):

  lines = getLines(fn)

  # Get sampling rate and start of data
  stt = 0
  for i in range(len(lines)):
    y = lines[i].split()
    if len(y) == 0:
      continue
    if y[0] == 'SampleTime':
      sr = int(y[2])
    if stt == 0:
      try:
        int(y[0])
        stt = i
      except ValueError:
        continue

  # Pull data into numpy array
  x = np.zeros((len(lines)-stt,len(cols)))
  for i in range(stt,len(lines)):
    y = lines[i].split()
    for j in range(len(cols)):
      x[i-stt,j] = y[cols[j]]
  #x[:,0] = x[:,0]-t0

  return x,sr

###############################################################################
# imports data from ECG file                                                  #
# in:  fn - filename                                                          #
#      t0 - initial time                                                      #
#      tN - end time                                                          #
# out: x - array of data                                                      #
#      sr - sampling rate                                                     #
###############################################################################

def getECGData(fn,t0,tN):

  lines = getLines(fn)

  # Get sampling rate and start of data
  stt = 0
  for i in range(len(lines)):
    y = lines[i].split()
    if len(y) == 0:
      continue
    if y[0] == 'SampleTime':
      sr = int(y[2])
    if stt == 0:
      try:
        int(y[0])
        stt = i
      except ValueError:
        continue

  # Get number of channels (not particularly efficient, but thorough...)
  nch = 0
  for i in range(stt,len(lines)):
    y = lines[i].split()
    j = int(y[1][-1])
    if j > nch:
      nch = j

  # Pull data into numpy array
  x = np.zeros((tN-t0+1,nch+1))
  x[:,0] = range(t0,tN+1)
  for i in range(stt,len(lines)):
    y = lines[i].split()
    j = int(y[1][-1])
    k = int(int(y[0]) - t0)
    x[k,j] = float(y[2])

  return x,nch,sr

###############################################################################
# interpolates the missing ECG data points                                    #
# in/out:  dat - array of ECG data                                            #
###############################################################################

def interpECGData(dat):

  dat = dat.copy() # add copy
  nch = np.ma.size(dat,1)-1        # number of channels
  for j in range(1,nch+1):
    for i in range(1,len(dat)-1):  # hoping no zeros in first or last positions!
      if dat[i,j] == 0:
        # find nearest non-zero neighbor below
        for k in range(1,i):
          y1 = dat[i-k,j]
          if y1 != 0:
            x1 = dat[i-k,0]
            break
        # find nearest non-zero neighbor above
        for k in range(1,len(dat)-i):
          y2 = dat[i+k,j]
          if y2 != 0:
            x2 = dat[i+k,0]
            break
        dat[i,j] = y1+(dat[i,0]-x1)*(y2-y1)/(x2-x1)

  return dat

###############################################################################
# generates the json file                                                     #
# in:  srECG/PULS/RESP - sampling rates for ECG, PULS, and RESP               #
#      nVol,nSlice,TR - volumes, slices, and TR for fMRI acquisition          #
#      gCh - ECG ground channel                                               #
#      card/respRange - cardiac and respiratory range of frequencies          #
###############################################################################

def genJSON(srECG,srPULS,srRESP,nVol,nSlice,TR,gCh,cardRange,respRange):

  meta = {}
  meta['samplingRate'] = []
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

  with open('meta.txt','w') as outfile:
    json.dump(meta,outfile)

###############################################################################
# main code                                                                   #
###############################################################################

# fold = sys.argv[1]
# cardiacRange = [0.75,3.5]  # Hz
# respRange = [0.01,0.5]     # Hz
#
# # get data
# Info,t0,tN,nVol,nSlice,TR = getInfoData('../data/'+fold+'/Physio_'+fold+'_Info.log',range(4))
# PULS,srPULS = getData('../data/'+fold+'/Physio_'+fold+'_PULS.log',(0,2),t0)
# RESP,srRESP = getData('../data/'+fold+'/Physio_'+fold+'_RESP.log',(0,2),t0)
# ECG,nch,srECG = getECGData('../data/'+fold+'/Physio_'+fold+'_ECG.log',t0,tN)
# ECG = interpECGData(ECG)
#
# mpl.plot(PULS[:,0]-t0,PULS[:,1])
# mpl.show()
# mpl.plot(RESP[:,0]-t0,RESP[:,1])
# mpl.show()
# mpl.plot(ECG[:,0]-t0,ECG[:,1],'b')
# mpl.plot(ECG[:,0]-t0,ECG[:,2],'r')
# mpl.plot(ECG[:,0]-t0,ECG[:,3],'g')
# mpl.plot(ECG[:,0]-t0,ECG[:,4],'k')
# mpl.show()
#
# genJSON(srECG,srPULS,srRESP,nVol,nSlice,TR,nch,cardiacRange,respRange)
