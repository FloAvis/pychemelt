import pytest

from pychemelt.utils.math import (
    first_derivative_savgol
)


def test_first_derivative_savgol_error():

    x = [1, 3, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    # Raise value error if x is not evenly spaced
    with pytest.raises(ValueError):
        first_derivative_savgol(x,y)

def test_first_derivative_savgol_polyorder_error():

    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]

    # Raise value error if the window is too short
    with pytest.raises(ValueError):
        first_derivative_savgol(x,y,window_length=1)