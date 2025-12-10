import numpy as np
import pytest

from pychemelt.utils.files import (
    guess_Tm_from_derivative,
    get_colors_from_numeric_values,
    fit_local_thermal_unfolding_to_signal_lst_exponential,
    fit_local_thermal_unfolding_to_signal_lst
)

from pychemelt.utils.palette import VIRIDIS

def test_error_not_enough_data():

    temp_lst = [[1,2,3]]
    deriv_lst = [[1,2,3]]

    x1 = 0
    x2 = 5

    with pytest.raises(ValueError):

        guess_Tm_from_derivative(temp_lst, deriv_lst, x1, x2)

def test_get_colors_from_numeric_values():

    y = [1, 2, 3, 4, 5]

    colors = get_colors_from_numeric_values(y, 1, 5,use_log_scale=True)

    assert colors[0] == VIRIDIS[0]
    assert colors[-1] == VIRIDIS[-1]

    colors = get_colors_from_numeric_values(y, 1, 5,use_log_scale=False)

    assert colors[0] == VIRIDIS[0]
    assert colors[-1] == VIRIDIS[-1]

def test_trigger_exception_fit_local_thermal_unfolding_to_signal_lst():

    signal_lst = [[np.nan for _ in range(5)]]
    temp_lst = [[x for x in range(5)]]

    Tms, dHs, predicted_lst =  fit_local_thermal_unfolding_to_signal_lst_exponential(
        signal_lst, temp_lst, [100],
        [1], [1], [1], [1], [1], [1]
    )

    assert Tms == []

    signal_lst = [[np.nan for _ in range(5)]]
    temp_lst = [[x for x in range(5)]]

    Tms, dHs, predicted_lst =  fit_local_thermal_unfolding_to_signal_lst(
        signal_lst, temp_lst, [100],
        [1], [1], [1], [1], [1], [1]
    )

    assert Tms == []