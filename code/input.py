import numpy as np
import json
import matplotlib.pyplot as mpl

###############################################################################
# imports lines from input file                                               #
# in:  filename - filename                                                    #
# out: lines - lines of file                                                  #
###############################################################################


def get_lines(filename):

    lines = []
    try:
        fh = open(filename, 'r')
    except OSError:
        msg = 'Cannot open input file ' + filename
        raise Warning(msg)
    else:
        # Get lines of file
        for line in fh:
            lines.append(line.rstrip('\n'))
        fh.close()

    return lines

###############################################################################
# imports data from info file                                                 #
# in:  filename - filename                                                    #
#      cols - list of columns to extract data from                            #
# out: x - array of data                                                      #
#      first_time - initial time                                              #
#      last_time - end time                                                   #
#      n_vols - number of volumes acquired                                    #
#      n_slices - number of slices per acquisition                            #
#      repetition_time - repetition time                                      #
###############################################################################


# Let's keep this function for comparisons
def load_cmrr_info_old(fn, cols):

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


def load_cmrr_info(filename):
    """
    Load information log files from CMRR sequences.

    Parameters
    ----------
    filename : str, pathlike
        Path to Information Log file.

    Returns
    -------
    traces : ndarray
        Time ticks of the scanner.
    meta_info : dict
        Dictionary with meta information about the info log file.

    """

    # TODO: Add function to validate input file. For example, it should be
    # a .log type file.

    lines = get_lines(filename)

    meta_info = dict()
    # Get parameters for meta file and lines containing data
    stt = 0
    stp = 0
    for i in range(len(lines)):
        y = lines[i].split()
        if len(y) == 0:
            continue
        elif y[0] == 'UUID':
            uuid = y[2]
            meta_info['uuid'] = uuid
        elif y[0] == 'ScanDate':
            scan_date = y[2]
            meta_info['scan_date'] = scan_date
        elif y[0] == 'LogVersion':
            log_version = y[2]
            meta_info['log_version'] = log_version
        elif y[0] == 'NumVolumes':
            n_vols = int(y[2])
            meta_info['n_vols'] = n_vols
        elif y[0] == 'NumSlices':
            n_slices = int(y[2])
            meta_info['n_slices'] = n_slices
        elif y[0] == 'NumEchoes':
            n_echoes = int(y[2])
            meta_info['n_echoes'] = n_echoes
        elif y[0] == 'FirstTime':
            first_time = int(y[2])
            meta_info['init_physio'] = first_time
        elif y[0] == 'LastTime':
            last_time = int(y[2])
            meta_info['end_physio'] = last_time

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
    # traces = np.zeros((stp - stt, len(cols)))
    traces = np.zeros((2, n_vols, n_slices, n_echoes), dtype=int)
    for i in range(stt, stp):
        y = lines[i].split()
        ivol = int(y[0])
        islice = int(y[1])
        iecho = int(y[-1])
        acq_start = int(y[2])
        acq_end = int(y[3])
        traces[:, ivol, islice, iecho] = [acq_start, acq_end]

    meta_info['init_scan'] = traces.min()
    meta_info['end_scan'] = traces.max()
    # TODO: Do we need this? The repetition is something usually knwon
    repetition_time = (meta_info['end_scan'] - meta_info['init_scan'])/n_vols
    meta_info['repetition_time'] = np.round(repetition_time)

    return traces, meta_info

###############################################################################
# imports data from file                                                      #
# in:  filename - filename                                                    #
#      first_time - initial time                                              #
#      last_time - end time                                                   #
#      interpolate - interpolate missing values (upsamples to ECG freq.)      #
# out: x - array of data                                                      #
#      n_channels - number of channels                                        #
#      sample_rate - sampling rate                                            #
###############################################################################


# Let's keep this function for comparisons
def load_cmrr_data_old(filename, first_time, last_time, interpolate=True):

    lines = get_lines(filename)

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


def load_cmrr_data(filename, info_dict, sync_scan=True):
    """

    Parameters
    ----------
    filename : str, pathlike
         Path to recording log file..
    info_dict : dict
        Dictionary with the meta information of the Info log file. It needs
        to be compute before by using the function load_cmrr_info.
    sync_scan : bool, optional
        Whether we want to resample the signal to be synchronized
        with the scanner times. The default is True.

    Returns
    -------
    signal : ndarray
        The recording signal, where the number of columns corresponds
        to the number of channels (ECG: 4, PULS: 1, RESP: 1) and the rows to
        observations.
    meta_info : dict
        Meta info of the physiological recording.

    """

    from scipy.interpolate import interp1d

    # TODO: Add checks of filename and info dict

    lines = get_lines(filename)

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
    n_samples = info_dict['end_physio'] - info_dict['init_physio'] + 1

    full_signal = np.zeros((n_samples, n_channels))
    time = np.arange(0, n_samples)
    if n_channels == 1:
        # Use separate loop for single channel to avoid repeated ifs for
        # channel #
        for i in range(stt, len(lines)):
            y = lines[i].split()
            k = int(int(y[0]) - info_dict['init_physio'])
            full_signal[k, 0] = float(y[2])
            time[k] = int(y[0])
    else:
        for i in range(stt, len(lines)):
            y = lines[i].split()
            j = int(int(y[1][-1])-1)
            k = int(int(y[0]) - info_dict['init_physio'])
            full_signal[k, j] = float(y[2])
            time[k] = int(y[0])

    if sync_scan:
        new_time = np.arange(info_dict['init_scan'],
                             info_dict['end_scan'] + 1)
    else:
        new_time = np.arange(info_dict['init_physio'],
                             info_dict['end_physio'] + 1)
    signal = []
    for s_channel in full_signal.T:
        mask = (s_channel != 0.)
        signal.append(interp1d(time[mask], s_channel[mask],
                               fill_value="extrapolate")(new_time))
    signal = np.column_stack(signal)

    meta_info = dict()
    meta_info['n_channels'] = n_channels
    meta_info['sample_rate'] = sample_rate

    return signal, meta_info


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
    Info, first_time, last_time, n_vols, n_slices, repetition_time = \
        load_cmrr_info_old(path + info_file, range(4))
    # get data from PULS file
    PULS, n_channels, sample_rate_puls = \
        load_cmrr_data_old(filename=path + puls_file,
                           first_time=first_time,
                           last_time=last_time,
                           interpolate=True)
    # get data from RESP file
    RESP, n_channels, sample_rate_resp = \
        load_cmrr_data(filename=path + resp_file,
                       first_time=first_time,
                       last_time=last_time,
                       interpolate=True)
    # get data from ECG file
    ECG, n_channels, sample_rate_ecg = \
        load_cmrr_data(filename=path + ecg_file,
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
