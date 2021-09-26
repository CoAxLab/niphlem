import numpy as np
import json
import matplotlib.pyplot as mpl

###############################################################################
# imports lines from input file                                               #
# in:  fn - filename                                                          #
# out: lines - lines of file                                                  #
###############################################################################


def get_lines(fn):

    lines = []
    try:
        fh = open(fn, 'r')
    except OSError:
        msg = 'Cannot open input file ' + fn
        raise Warning(msg)
    else:
        # Get lines of file
        for line in fh:
            lines.append(line.rstrip('\n'))
        fh.close()

    return lines

###############################################################################
# imports data from info file                                                 #
# in:  fn - filename                                                          #
#      cols - list of columns to extract data from                            #
# out: x - array of data                                                      #
#      first_time - initial time                                              #
#      last_time - end time                                                   #
#      n_vols - number of volumes acquired                                    #
#      n_slices - number of slices per acquisition                            #
#      repetition_time - repetition time                                      #
###############################################################################


def load_cmrr_info(fn, cols):

    lines = get_lines(fn)
    # Get parameters for meta file and lines containing data
    stt = 0
    stp = 0
    for i in range(len(lines)):
        y = lines[i].split()
        if len(y) == 0:
            continue
        if y[0] == 'NumVolumes':
            n_vols = int(y[2])
        if y[0] == 'NumSlices':
            n_slices = int(y[2])
        if y[0] == 'FirstTime':
            first_time = int(y[2])
        if y[0] == 'LastTime':
            last_time = int(y[2])
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
    x = np.zeros((stp - stt, len(cols)))
    for i in range(stt, stp):
        y = lines[i].split()
        for j in range(len(cols)):
            x[i - stt, j] = y[cols[j]]
    repetition_time = round((x[-1, 3] - x[0, 2]) / n_vols)

    # TODO: Maybe return here just two objects, x for signal and the rest
    # in a dictionary as meta information?
    return x, first_time, last_time, n_vols, n_slices, repetition_time

###############################################################################
# imports data from file                                                      #
# in:  fn - filename                                                          #
#      first_time - initial time                                              #
#      last_time - end time                                                   #
#      interpolate - interpolate missing values (upsamples to ECG freq.)      #
# out: x - array of data                                                      #
#      n_channels - number of channels                                        #
#      sample_rate - sampling rate                                            #
###############################################################################


def load_cmrr_data(fn, first_time, last_time, interpolate=True):

    lines = get_lines(fn)

    # Get sampling rate and start of data
    stt = 0
    for i in range(len(lines)):
        y = lines[i].split()
        if len(y) == 0:
            continue
        if y[0] == 'SampleTime':
            sample_rate = int(y[2])
        # Inherent assumption that all lines starting with a number are data
        if stt == 0:
            try:
                int(y[0])
                stt = i
            except ValueError:
                continue

    # Get number of channels (not particularly efficient, but thorough...)
    if y[1] == 'PULS' or y[1] == 'RESP':
        n_channels = 1
    else:
        n_channels = 0
        for i in range(stt, len(lines)):
            y = lines[i].split()
            j = int(y[1][-1])
            if j > n_channels:
                n_channels = j

    # Pull data into numpy array
    x = np.zeros((last_time - first_time + 1, n_channels + 1))
    x[:, 0] = range(0, last_time - first_time + 1)
    if n_channels == 1:
        # Use separate loop for single channel to avoid repeated ifs for
        # channel #
        for i in range(stt, len(lines)):
            y = lines[i].split()
            k = int(int(y[0]) - first_time)
            x[k, 1] = float(y[2])
    else:
        for i in range(stt, len(lines)):
            y = lines[i].split()
            j = int(y[1][-1])
            k = int(int(y[0]) - first_time)
            x[k, j] = float(y[2])
    if interpolate:
        x = interpolate_missing_data(x)

    return x, n_channels, sample_rate

###############################################################################
# interpolates the missing data points                                        #
# in/out:  dat - array of data                                                #
###############################################################################


def interpolate_missing_data(dat):

    dat = dat.copy()  # add copy
    n_channels = np.ma.size(dat, 1) - 1        # number of channels
    for j in range(1, n_channels + 1):

        # extrapolate at beginning of series if needed
        for i in range(0, len(dat)):
            # move on if no empty values at beginning of series
            if dat[i, j] != 0:
                break
            # find nearest non-zero neighbor above
            for k in range(1, len(dat) - 1):
                y1 = dat[i + k, j]
                if y1 != 0:
                    x1 = dat[i + k, 0]
                    break
            # find next nearest non-zero neighbor above
            for kk in range(k + 1, len(dat) - 1):
                y2 = dat[i + kk, j]
                if y2 != 0:
                    x2 = dat[i + kk, 0]
                    break
            dat[i, j] = y1 + (dat[i, 0] - x1) * (y2 - y1) / (x2 - x1)

        # extrapolate at end of series if needed
        for i in range(len(dat) - 1, 0, -1):
            # move on to interpolating interior values if no empty values at
            # end of series
            if dat[i, j] != 0:
                break
            # find nearest non-zero neighbor below
            for k in range(1, len(dat) - 1):
                y1 = dat[i - k, j]
                if y1 != 0:
                    x1 = dat[i - k, 0]
                    break
            # find next nearest non-zero neighbor below
            for kk in range(k + 1, len(dat) - 1):
                y2 = dat[i - kk, j]
                if y2 != 0:
                    x2 = dat[i - kk, 0]
                    break
            dat[i, j] = y1 + (dat[i, 0] - x1) * (y2 - y1) / (x2 - x1)

        for i in range(1, len(dat) - 1):
            if dat[i, j] == 0:
                # find nearest non-zero neighbor below
                for k in range(1, i):
                    y1 = dat[i - k, j]
                    if y1 != 0:
                        x1 = dat[i - k, 0]
                        break
                # find nearest non-zero neighbor above
                for k in range(1, len(dat) - i):
                    y2 = dat[i + k, j]
                    if y2 != 0:
                        x2 = dat[i + k, 0]
                        break
                dat[i, j] = y1 + (dat[i, 0] - x1) * (y2 - y1) / (x2 - x1)

    return dat

###############################################################################
# generates the json file                                                     #
# in:  sFreq - sampling frequency (1/tic in ms)                               #
#      sample_rate_ecg/PULS/RESP - sampling rates for ECG, PULS, and RESP     #
#      n_vols - number of volumes                                             #
#      n_slices - number of slices                                            #
#      repetition_time - repetition_time for fMRI acquisition                 #
#      gCh - ECG ground channel                                               #
#      card/resp_range - cardiac and respiratory range of frequencies         #
###############################################################################


def gen_JSON(
        sFreq,
        sample_rate_ecg,
        sample_rate_puls,
        sample_rate_resp,
        n_vols,
        n_slices,
        repetition_time,
        gCh,
        cardiac_range,
        resp_range):

    meta = {}
    meta['samplingRate'] = []
    meta['samplingRate'].append({
        'freq': sFreq
    })
    meta['samplingRate'].append({
        'channel': 'ECG',
        'rate': sample_rate_ecg
    })
    meta['samplingRate'].append({
        'channel': 'PULS',
        'rate': sample_rate_puls
    })
    meta['samplingRate'].append({
        'channel': 'RESP',
        'rate': sample_rate_resp
    })
    meta['MRacquisition'] = []
    meta['MRacquisition'].append({
        'Volumes': n_vols,
        'Slices': n_slices,
        'repetition_time': repetition_time,
    })
    meta['ECGground'] = []
    meta['ECGground'].append({
        'Channel': gCh
    })
    meta['frequencyRanges'] = []
    meta['frequencyRanges'].append({
        'Cardiac': cardiac_range,
        'Respiratory': resp_range
    })

    with open('meta.txt', 'w') as outfile:
        json.dump(meta, outfile)

###############################################################################
# main code                                                                   #
# In:  path - path to folder containing physio files, assumes common location #
#      info_file - Info file name                                             #
#      puls_file - PULS file name                                             #
#      resp_file - RESP file name                                             #
#      ecg_file - ECG file name                                               #
#      show_signals - flag to plot the signals (default False)                #
# Out: (meta.txt) - json file containing meta data                            #
#      (signal.csv) - file containing condensed signal data                   #
###############################################################################


def proc_input(path,
               info_file,
               puls_file,
               resp_file,
               ecg_file,
               show_signals=False):

    cardiac_range = [0.75, 3.5]  # Hz
    resp_range = [0.01, 0.5]     # Hz
    # TODO: Take this as an input or extract somehow
    sFreq = 400                 # Hz

    # ensure path ends in /
    if path[-1] != '/':
        path = path + '/'

    # get data from INFO file
    Info, first_time, last_time, \
        n_vols, n_slices, repetition_time = load_cmmr_info(path + info_file,
                                                           range(4))
    # get data from PULS file
    PULS, n_channels, sample_rate_puls = load_cmrr_data(fn=path + puls_file,
                                                        first_time=first_time,
                                                        last_time=last_time,
                                                        interpolate=True)
    # get data from RESP file
    RESP, n_channels, sample_rate_resp = load_cmrr_data(fn=path + resp_file,
                                                        first_time=first_time,
                                                        last_time=last_time,
                                                        interpolate=True)
    # get data from ECG file
    ECG, n_channels, sample_rate_ecg = load_cmrr_data(fn=path + ecg_file,
                                                      first_time=first_time,
                                                      last_time=last_time,
                                                      interpolate=True)
    # generate JSON dictionary
    gen_JSON(
        sFreq,
        sample_rate_ecg,
        sample_rate_puls,
        sample_rate_resp,
        n_vols,
        n_slices,
        repetition_time,
        n_channels,
        cardiac_range,
        resp_range)
    # store aligned signals in a single matrix, save to signal.npy
    signal = np.zeros((len(ECG), n_channels + 3))
    signal[:, 0:(n_channels + 1)] = ECG
    signal[:, n_channels + 1] = PULS[:, 1]
    signal[:, n_channels + 2] = RESP[:, 1]
    np.save('signal', signal)

    # plot signals if desired
    if show_signals:
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
#info_file = 'Physio_sample2_Info.log'
#puls_file = 'Physio_sample2_PULS.log'
#resp_file = 'Physio_sample2_RESP.log'
#ecg_file = 'Physio_sample2_ECG.log'
#proc_input(path, info_file, puls_file, resp_file, ecg_file, show_signals=True)
