import numpy as np
import pytest

from pychemelt.utils.processing import (
    guess_Tm_from_derivative,
    get_colors_from_numeric_values,
    fit_local_thermal_unfolding_to_signal_lst,
    adjust_value_to_interval,
    parse_number,
    are_all_strings_numeric,
    is_float
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

    Tms, dHs, predicted_lst =  fit_local_thermal_unfolding_to_signal_lst(
        signal_lst, temp_lst, [100],
        [1], [1], [1], [1], [1], [1],
        baseline_native_fx = lambda x: x,
        baseline_unfolded_fx = lambda x: x
    )

    assert Tms == []

def test_adjust_value_to_interval():

    assert adjust_value_to_interval(10, 0, 100,1) == 10

    assert adjust_value_to_interval(-1, 0, 100,0.5) == 0.5

    assert adjust_value_to_interval(110, 0, 100,0.5) == 99.5

def test_parse_number():

    assert parse_number('10') == 10

    assert parse_number('10.5') == 10.5

    assert parse_number('1e-3') == 0.001

    with pytest.raises(ValueError):
        parse_number('not_a_number')

def test_are_all_strings_numeric():

    assert are_all_strings_numeric(['10', '20', '30']) == True

    assert are_all_strings_numeric(['10', '20.5', '1e-3']) == True

    assert are_all_strings_numeric(['10', 'not_a_number', '30']) == False

def test_is_float():

    assert is_float('10') == True

    assert is_float('10.5') == True

    assert is_float('1e-3') == True

    assert is_float('not_a_number') == False