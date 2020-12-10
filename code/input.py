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
#      startTrim - lines to trim from beginning of file                       #
#      endTrim - lines to trim from end of file                               #
#      cols - list of columns to extract data from                            #
# out: x - array of data                                                      #
#      t0 - initial time                                                      #
#      tN - end time                                                          #
#      nVol - number of volumes acquired                                      #
#      nSlice - number of slices per acquisition                              #
#      TR - repetition time                                                   #
#      acqT - acquisition time                                                #
###############################################################################

def getInfoData(fn,startTrim,endTrim,cols):

  lines = getLines(fn)
  t0 = lines[len(lines)-2].split()
  t0 = int(t0[2])
  tN = lines[len(lines)-1].split()
  tN = int(tN[2])
  # Get number of volumes
  for i in range(startTrim):
    y = lines[i].split()
    if len(y) == 0:
      continue
    if y[0] == 'NumVolumes':
      nVol = float(y[2])
    if y[0] == 'NumSlices':
      nSlice = float(y[2])

  # Pull data into numpy array
  x = np.zeros((len(lines)-(startTrim+endTrim),len(cols)))
  for i in range(startTrim,len(lines)-endTrim):
    y = lines[i].split()
    for j in range(len(cols)):
      x[i-startTrim,j] = y[cols[j]]
  TR = round((x[-1,3]-x[0,2])/nVol)
  acqT = x[0,3]-x[0,2]

  return x,t0,tN,nVol,nSlice,TR,acqT

###############################################################################
# imports data from input file                                                #
# in:  fn - filename                                                          #
#      startTrim - lines to trim from beginning of file                       #
#      endTrim - lines to trim from end of file                               #
#      cols - list of columns to extract data from                            #
#      t0 - initial time                                                      #
# out: x - array of data                                                      #
#      sr - sampling rate                                                     #
###############################################################################

def getData(fn,startTrim,endTrim,cols,t0):

  lines = getLines(fn)

  # Get sampling rate
  for i in range(startTrim):
    y = lines[i].split()
    if y[0] == 'SampleTime':
      sr = float(y[2])
      break
  # Pull data into numpy array
  x = np.zeros((len(lines)-(startTrim+endTrim),len(cols)))
  for i in range(startTrim,len(lines)-endTrim):
    y = lines[i].split()
    for j in range(len(cols)):
      x[i-startTrim,j] = y[cols[j]]
  x[:,0] = x[:,0]-t0

  return x,sr

###############################################################################
# imports data from ECG file                                                  #
# in:  fn - filename                                                          #
#      startTrim - lines to trim from beginning of file                       #
#      endTrim - lines to trim from end of file                               #
#      nch - number of channels (should actually pull this from the data)     #
#      t0 - initial time                                                      #
#      tN - end time                                                          #
# out: x - array of data                                                      #
#      sr - sampling rate                                                     #
###############################################################################

def getECGData(fn,startTrim,endTrim,nch,t0,tN):

  lines = getLines(fn)

  # Get sampling rate
  for i in range(startTrim):
    y = lines[i].split()
    if y[0] == 'SampleTime':
      sr = float(y[2])
      break
  # Pull data into numpy array
  x = np.zeros((tN-t0+1,nch+1))
  x[:,0] = range(t0,tN+1)
  for i in range(startTrim,len(lines)-endTrim):
    y = lines[i].split()
    j = int(y[1][-1])
    k = int(int(y[0]) - t0)
    x[k,j] = float(y[2])

  return x,sr

###############################################################################
# interpolates the missing ECG data points                                    #
# in/out:  dat - array of ECG data                                            #
###############################################################################

def interpECGData(dat):

  nch = np.ma.size(dat,1)    # number of channels
  for j in range(1,nch):
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
# main code                                                                   #
###############################################################################

fold = sys.argv[1]
nch = 4

# get data
Info,t0,tN,nVol,nSlice,TR,acqT = getInfoData('../data/'+fold+'/Physio_'+fold+'_Info.log',10,2,[0,1,2,3])
PULS,srPULS = getData('../data/'+fold+'/Physio_'+fold+'_PULS.log',8,0,(0,2),t0)
RESP,srRESP = getData('../data/'+fold+'/Physio_'+fold+'_RESP.log',8,0,(0,2),t0)
ECG,srECG = getECGData('../data/'+fold+'/Physio_'+fold+'_ECG.log',8,0,nch,t0,tN)
ECG = interpECGData(ECG)

#mpl.plot(PULS[:,0]-t0,PULS[:,1])
#mpl.show()
#mpl.plot(RESP[:,0]-t0,RESP[:,1])
#mpl.show()
#mpl.plot(ECG[:,0]-t0,ECG[:,1],'b')
#mpl.plot(ECG[:,0]-t0,ECG[:,2],'r')
#mpl.plot(ECG[:,0]-t0,ECG[:,3],'g')
#mpl.plot(ECG[:,0]-t0,ECG[:,4],'k')
#mpl.show()

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
  'acqTime': acqT     # is this the quantity of interest?
})
meta['ECGground'] = []
meta['ECGground'].append({
  'Channel': 4        # what should this actually be?
})


with open('meta.txt','w') as outfile:
  json.dump(meta,outfile)
