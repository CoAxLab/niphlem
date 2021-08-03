# content of test_example.py (adapted from https://docs.pytest.org)

def inc(x):
    """Functionality that we want to test"""
    return x + 1


def test_inc():
    # This will give an error
    assert inc(3) == 5
