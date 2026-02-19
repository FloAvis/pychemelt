import pytest
import numpy as np

from pychemelt.utils.math import (
    first_derivative_savgol,
    solve_one_root_quadratic,
    solve_one_root_depressed_cubic
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

def test_solve_one_root_quadratic():

    assert solve_one_root_quadratic(3, 2, -1) == pytest.approx(1/3)

    #division by 0
    assert np.isnan(solve_one_root_quadratic(2, -2, 0))

    assert solve_one_root_quadratic(2, 5, 1.125) == pytest.approx(-0.25)


def test_solve_one_root_depressed_cubic():

    assert solve_one_root_depressed_cubic(2, 2) == pytest.approx(-0.77092, abs=1e-4)



