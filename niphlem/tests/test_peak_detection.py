import numpy as np

from niphlem.events import compute_max_events


def test_peak_detection():
    """
    Function that tests that the peak detection algorithm
    identifies the correct number of peaks in signal
    """
    rng = np.random.RandomState(1234)
    N = 1000  # Time points
    p = 0.5

    signal = rng.binomial(1, p=p, size=N)
    signal = (np.diff(signal) == -1).astype(float)
    signal[0], signal[-1] = 0, 0
    print(f"the number of peaks is: {sum(signal==1)}")

    peaks_idxs = compute_max_events(signal, peak_rise=0.75, delta=1)
    peaks_idxs = peaks_idxs.astype(int)
    assert len(peaks_idxs) == sum(signal == 1)

    # Now let's test the behaviour of peak rise hyperparameter, which controls
    # how tall events need to be in order to be consider peaks. This variable
    # multiplies the height of the 20th maximum event signal and set this
    # as the baseline.

    # Set 20% of original peaks a bit below 1.0 (0.7, for example)
    sub_peaks = rng.choice(peaks_idxs,
                           replace=False,
                           size=int(0.2*len(peaks_idxs)))
    signal_2 = signal.copy()
    signal_2[sub_peaks] = 0.7

    print(f"the number of leading peaks is: {sum(signal_2==1)}")
    print(f"the number of sub-leading peaks is: {sum(signal_2==0.7)}")

    # As peak_rise here is 0.75, and the 20 th maximum event signal is 1,
    # then any signal below 0.75x1 will not be marked as peak.
    peaks_idxs_2 = compute_max_events(signal_2, peak_rise=0.75, delta=1)
    assert len(peaks_idxs_2) == sum(signal_2 == 1)

    # If we want then to also pick the sub-leading peaks, we need to lower
    # the parameter peak_rise (for instance, to 0.5 in this toy case)
    peaks_idxs_2 = compute_max_events(signal_2, peak_rise=0.5, delta=1)
    assert len(peaks_idxs_2) == sum(signal_2 > 0)

    # Finally, let's test the behaviour of delta, which controls how separable
    # peaks need to be.

    signal_3 = signal.copy()
    delta = 5
    # We can know the very easily how many peaks are separated by at least
    # this chosen delta (the +1 here is to include the first peak).
    n_peaks_by_delta = sum(np.diff(np.where(signal_3 == 1.0)[0]) > delta) + 1
    # And using the compute events function passing the explicit delta
    peaks_idxs = compute_max_events(signal_3, peak_rise=0.75, delta=delta)

    assert len(peaks_idxs) == n_peaks_by_delta
