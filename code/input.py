import sys
import numpy as np

###############################################################################
# imports data from input file                                                #
# in:  fn - filename                                                          #
#      startTrim - lines to trim from beginning of file                       #
#      endTrim - lines to trim from end of file                               #
#      cols - list of columns to extract data from                            #
#      t0 - initial time to adjust by (didn't actually do anything with this) #
# out: x - array of data                                                      #
#      t0 - initial time                                                      #
###############################################################################

def getLines(fn,startTrim,endTrim,nCol,t0):
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

  # Get t0 if not defined
  if t0 == 0:
    t0 = lines[len(lines)-2].split()
    t0 = t0[2]

  # Pull data into numpy array
  x = np.zeros((len(lines)-(startTrim+endTrim),len(nCol)))
  for i in range(startTrim,len(lines)-endTrim):
    y = lines[i].split()
    for j in range(len(nCol)):
      x[i-startTrim,j] = y[nCol[j]]

  return x,t0

###############################################################################
# main code                                                                   #
###############################################################################

fold = sys.argv[1]

Info,t0 = getLines('../data/'+fold+'/Physio_'+fold+'_Info.log',10,2,[0,1,2,3],0)
PULS,t0 = getLines('../data/'+fold+'/Physio_'+fold+'_PULS.log',8,0,(0,2),t0)
RESP,t0 = getLines('../data/'+fold+'/Physio_'+fold+'_RESP.log',8,0,(0,2),t0)
ECG,t0 = getLines('../data/'+fold+'/Physio_'+fold+'_ECG.log',8,0,(0,2),t0)
