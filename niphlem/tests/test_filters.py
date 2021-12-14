import numpy as np
from niphlem.clean import _transform_filter


def test_transform():

    rng = np.random.RandomState(1234)
    n_samples = 200

    low_pass = None
    high_pass = None
    sampling_rate = None

    eps = 100*np.finfo(np.float64).eps
    # Compare output for different options.
    # single timeseries
    data = 3.1 + 2.5*rng.standard_normal(size=n_samples)

    transform = None
    data_transform = _transform_filter(data,
                                       transform=transform,
                                       high_pass=high_pass,
                                       low_pass=low_pass,
                                       sampling_rate=sampling_rate)
    assert abs(np.mean(data_transform)) < eps

    transform = "abs"
    data_transform = _transform_filter(data,
                                       transform=transform,
                                       high_pass=high_pass,
                                       low_pass=low_pass,
                                       sampling_rate=sampling_rate)
    assert abs(np.mean(data_transform)) < eps

    transform = "zscore"
    data_transform = _transform_filter(data,
                                       transform=transform,
                                       high_pass=high_pass,
                                       low_pass=low_pass,
                                       sampling_rate=sampling_rate)
    assert abs(np.mean(data_transform)) < eps
    assert np.allclose(np.std(data_transform), 1.0)


def test_filter():

    from scipy.signal import periodogram

    # Create signal with sampling 50 Hz, that has
    # a frequency signal of 5, 10 and 15 Hz.
    sampling_rate = 50
    times = np.arange(1000)/sampling_rate
    signal = np.sin(2*np.pi*5*times) +\
        np.sin(2*np.pi*10*times) + np.sin(2*np.pi*15*times)

    # band pass filter betweenn 6 and 12
    low_pass = 12
    high_pass = 6

    signal_transform = _transform_filter(signal,
                                         high_pass=high_pass,
                                         low_pass=low_pass,
                                         sampling_rate=sampling_rate)
    freqs, Pxx = periodogram(signal_transform, fs=sampling_rate)
    #  Uncomment to see the plot and how the filtered worked
    # plt.plot(freqs, Pxx)

    # Verify that the filtered frequencies are removed with respect
    # to passed frequencies
    Pxx_passed = np.sum(Pxx[(freqs < low_pass * 2.) &
                            (freqs > high_pass / 2.)])
    Pxx_filtered = np.sum(Pxx[(freqs >= low_pass * 2.) |
                              (freqs <= high_pass / 2)])
    assert Pxx_filtered < 1e-3*Pxx_passed

    # low pass filter below 12 Hz
    low_pass = 12
    high_pass = None

    signal_transform = _transform_filter(signal,
                                         high_pass=high_pass,
                                         low_pass=low_pass,
                                         sampling_rate=sampling_rate)

    freqs, Pxx = periodogram(signal_transform, fs=sampling_rate)

    #  Uncomment to see the plot and how the filtered worked
    # plt.plot(freqs, Pxx)

    Pxx_passed = np.sum(Pxx[freqs < low_pass * 2.])
    Pxx_filtered = np.sum(Pxx[freqs >= low_pass * 2.])
    assert Pxx_filtered < 1e-3*Pxx_passed

    # high pass filter above 6 Hz
    low_pass = None
    high_pass = 6

    signal_transform = _transform_filter(signal,
                                         high_pass=high_pass,
                                         low_pass=low_pass,
                                         sampling_rate=sampling_rate)

    freqs, Pxx = periodogram(signal_transform, fs=sampling_rate)

    #  Uncomment to see the plot and how the filtered worked
    # plt.plot(freqs, Pxx)

    Pxx_passed = np.sum(Pxx[freqs > high_pass / 2.])
    Pxx_filtered = np.sum(Pxx[freqs <= high_pass / 2])
    assert Pxx_filtered < 1e-3*Pxx_passed
