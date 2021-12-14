import numpy as np
import pytest

from niphlem.models import RetroicorPhysio


class TestPhysio(RetroicorPhysio):
    "This is the RetroicorPhysio class that here computes no regressors."
    " It is only for testing previous transformations and the several"
    " checkings ond the input parameters."

    def _process_regressors(self,
                            signal,
                            time_physio,
                            time_scan):
        return signal


def test_initial_checks():
    rng = np.random.RandomState(1234)
    signal = rng.randn(100, 3)

    # add a fake time scan. It doesn't matter. It will not be used, but
    # we have to pass one with the same observations of the signal
    time_scan = rng.randn(100)

    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200)

    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan[:50])
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")

    # This should give an error because a not numeric low_pass freq is used
    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                             low_pass="a string")
    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan)
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")

    # This should give an error because a not numeric high_pass freq is used
    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                             high_pass="a string")
    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan)
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")

    # Error because high_pass freq is greater than low_pass frequency
    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                             high_pass=2, low_pass=1)
    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan)
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")

    # This should give an error because we are passing a not numeric column
    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                             columns=[1, 2, "dog"])
    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan)
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")

    # This should give an error because we are not passing the right
    # number of columns.
    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                             columns=[1, 2])
    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan)
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")

    # these should give errors because the order has to be positive > 0
    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200, order=0)
    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan)
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")

    # these should give errors because the order has to be positive > 0
    test_physio = TestPhysio(physio_rate=400, t_r=2.0, delta=200, order=-1)
    with pytest.raises(ValueError) as exc_info:
        test_physio.compute_regressors(signal,
                                       time_scan,
                                       time_physio=time_scan)
    assert exc_info.type is ValueError
    print(f"we have passed the test: {exc_info.value.args[0]}")


def test_columns_param():
    "Function to test columns argument"

    rng = np.random.RandomState(1234)
    signal = rng.randn(100, 3)

    mean_signal = np.mean(signal, axis=1)
    mean_signal -= mean_signal.mean()

    signal12 = signal[:, 0]-signal[:, 1]
    signal12 -= signal12.mean()

    signal13 = signal[:, 0]-signal[:, 2]
    signal13 -= signal13.mean()

    # add a fake time scan. It doesn't matter. It will not be used, but
    # we have to pass one with the same number of observations of the signal.
    time_scan = rng.randn(100)

    test_mean = TestPhysio(physio_rate=400, t_r=2.0,  delta=200,
                           columns="mean")
    assert np.allclose(test_mean.compute_regressors(signal, time_scan)[:, 0],
                       mean_signal)

    # Passing a list of weight columns should also work
    test_list_mean = TestPhysio(physio_rate=400, t_r=2.0,  delta=200,
                                columns=[1/3, 1/3, 1/3])
    # add a fake time scan. It doesn't matter. It will not be used, but
    # we have to pass one with the same number of observations of the signal.
    assert np.allclose(
        test_list_mean.compute_regressors(signal, time_scan)[:, 0],
        mean_signal
        )

    test_list_12 = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                              columns=[1, -1, 0])
    # add a fake time scan. It doesn't matter. It will not be used, but
    # we have to pass one with the same observations of the signal
    assert np.allclose(
        test_list_12.compute_regressors(signal, time_scan)[:, 0],
        signal12
        )

    test_list_13 = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                              columns=[1, 0, -1])
    # add a fake time scan. It doesn't matter. It will not be used, but
    # we have to pass one with the same observations of the signal
    assert np.allclose(
        test_list_13.compute_regressors(signal, time_scan)[:, 0],
        signal13
        )

    # Test now just one single channel (the first one, for example)
    test_single = TestPhysio(physio_rate=400, t_r=2.0, delta=200,
                             columns=[1, 0, 0])
    assert np.allclose(
        test_single.compute_regressors(signal, time_scan)[:, 0],
        signal[:, 0] - signal[:, 0].mean()
        )
