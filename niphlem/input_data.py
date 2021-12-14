import numpy as np
import json
import matplotlib.pyplot as mpl


def get_lines(filename):
    """
    Read in lines from file, stripping new line markers

    Parameters
    ----------
    filename : str, pathlike
        Path to file.

    Returns
    -------
    lines : list
        List containing each line.
    """

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
            meta_info['uuid'] = y[2]
        elif y[0] == 'ScanDate':
            meta_info['scan_date'] = y[2]
        elif y[0] == 'LogVersion':
            meta_info['log_version'] = y[2]
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
            meta_info['init_physio'] = int(y[2])
        elif y[0] == 'LastTime':
            meta_info['end_physio'] = int(y[2])

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

    meta_info['init_scan'] = int(traces.min())
    meta_info['end_scan'] = int(traces.max())
    # TODO: Do we need this? The repetition is something usually knwon
    repetition_time = (meta_info['end_scan'] - meta_info['init_scan'])/n_vols
    meta_info['repetition_time'] = np.round(repetition_time)

    return traces, meta_info


def load_cmrr_data(filename, sig_type, info_dict, sync_scan=True):
    """
    Load data log files from CMRR sequences.

    Parameters
    ----------
    filename : str, pathlike
        Path to recording log file..
    sig_type : str
        Type of signal for use in dictionary
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
    info_dict : dict
        Updated meta info of the physiological recording.

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

    info_dict[sig_type] = {}
    info_dict[sig_type]['n_channels'] = n_channels
    info_dict[sig_type]['sample_rate'] = sample_rate

    return signal, info_dict


def proc_input(path,
               info_file,
               puls_file,
               resp_file,
               ecg_file,
               meta_filename='meta.json',
               sig_filename='signal',
               show_signals=False):
    """
    Extract relevant data from info, PULS, RESP, and ECG files; creates meta
        file with info and .npy file with signal array

    Parameters
    ----------
    path : str, pathlike
        Path to directories containing files.
    info_file : str, pathlike
        Info file name.
    puls_file : str, pathlike
        PULS file name.
    resp_file : str, pathlike
        RESP file name.
    ecg_file : str, pathlike
        ECG file name.
    meta_filename : str, pathlike, optional
        Filename to store meta info, default 'meta.json'
    sig_filename : str, pathlike, optional
        Filename to store signal array, default 'signal'
    show_signals : bool, optional
        Flag to show plots of signals, default False.
    """

    cardiac_range = [0.75, 3.5]  # Hz
    respiratory_range = [0.01, 0.5]     # Hz
    # TODO: Take this as an input or extract somehow
    sampling_frequency = 400    # Hz

    # ensure path ends in /
    if path[-1] != '/':
        path = path + '/'

    # get data from INFO file
    traces, meta_info = load_cmrr_info(filename=path + info_file)
    meta_info['frequency_info'] = {}
    meta_info['frequency_info']['sampling_rate'] = sampling_frequency
    meta_info['frequency_info']['cardiac_range'] = cardiac_range
    meta_info['frequency_info']['respiratory_range'] = respiratory_range

    # get data from PULS file
    PULS, meta_info = \
        load_cmrr_data(filename=path + puls_file,
                       sig_type='puls',
                       info_dict=meta_info,
                       sync_scan=True)
    # get data from RESP file
    RESP, meta_info = \
        load_cmrr_data(filename=path + resp_file,
                       sig_type='resp',
                       info_dict=meta_info,
                       sync_scan=True)
    # get data from ECG file
    ECG, meta_info = \
        load_cmrr_data(filename=path + ecg_file,
                       sig_type='ecg',
                       info_dict=meta_info,
                       sync_scan=True)

    # store aligned signals in a single matrix, save to signal.npy
    n_channels = meta_info['ecg']['n_channels']
    signal = np.zeros((len(ECG), n_channels + 2))
    signal[:, 0:n_channels] = ECG
    signal[:, [n_channels]] = PULS
    signal[:, [n_channels + 1]] = RESP
    np.save(sig_filename, signal)

    with open(meta_filename, 'w') as outfile:
        json.dump(meta_info, outfile)

    # plot signals if desired
    if show_signals:
        mpl.plot(PULS)
        mpl.show()
        mpl.plot(RESP)
        mpl.show()
        mpl.plot(ECG[:, 0], 'b')
        mpl.plot(ECG[:, 1], 'r')
        mpl.plot(ECG[:, 2], 'g')
        mpl.plot(ECG[:, 3], 'k')
        mpl.show()


def load_bids_physio():
    """
    Load physiological data in BIDS format
    """
    # TODO
    return NotImplementedError


###############################################################################

#path = '/Users/andrew/Fellowship/projects/brainhack-physio-project/data/sample2/'
#info_file = 'Physio_sample2_Info.log'
#puls_file = 'Physio_sample2_PULS.log'
#resp_file = 'Physio_sample2_RESP.log'
#ecg_file = 'Physio_sample2_ECG.log'
#proc_input(path, info_file, puls_file, resp_file, ecg_file, show_signals=True)
