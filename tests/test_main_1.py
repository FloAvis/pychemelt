"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np
import pytest

from pychemelt import Sample

sample = Sample()

def test_load_data():
    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')

    assert len(sample.conditions) == 48
    assert len(sample.labels) == 48


def test_set_denaturant_concentrations():

    sample.set_denaturant_concentrations()

    assert np.min(sample.denaturant_concentrations_pre) == 0
    assert np.max(sample.denaturant_concentrations_pre) == 8.24

def test_select_signal():

    sample.set_signal(['350nm'])

    assert sample.signal_names == ['350nm']

def test_select_conditions():

    # Select without scaling
    sample.select_conditions(
        [False for _ in range(24)] + [True for _ in range(8)] + [False for _ in range(16)],
        normalise_to_global_max=False
    )

    assert len(sample.signal_lst_multiple) == 1
    assert len(sample.signal_lst_multiple[0]) == 8
    assert np.max(sample.signal_lst_multiple[0]) != 1.0

    # Select with scaling
    sample.select_conditions(
        [False for _ in range(24)] + [True for _ in range(8)] + [False for _ in range(16)],
        normalise_to_global_max=True
    )

    assert len(sample.signal_lst_multiple) == 1
    assert len(sample.signal_lst_multiple[0]) == 8
    assert np.max(sample.signal_lst_multiple[0]) == 100


def test_set_temperature_range():

    # Raise error if t_max < t_min
    with pytest.raises(ValueError):
        sample.set_temperature_range(80, 30)

    sample.set_temperature_range(30, 80)

    assert np.min(sample.temp_lst_multiple[0]) >= 30

    sample.set_temperature_range(5, 100)

    assert sample.user_min_temp == 5
    assert sample.user_max_temp == 100

def test_expand_signal():

    sample.expand_multiple_signal()

    assert sample.signal_lst_expanded is not None
    assert sample.temp_lst_expanded is not None

def test_set_signal_id():

    sample.set_signal_id()

    assert sample.signal_ids == [0 for _ in range(8)]

def test_estimate_derivative():

    sample.estimate_derivative()

    assert len(sample.deriv_lst_multiple[0]) == 8

def test_guess_Tm():

    sample.guess_Tm()

    # assert almost
    np.testing.assert_allclose(sample.t_melting_init_multiple[0][-1],63.82,rtol=0.01)

def test_init_slope_dic():

    # Raise error if self.poly_order_native or self.poly_order_unfolded is None
    with pytest.raises(ValueError):
        sample.init_slope_dic()

def test_guess_initial_parameters_ratio():

    sample.n_residues = 130
    sample.guess_initial_parameters(poly_order_native=1,poly_order_unfolded=1)

    np.testing.assert_allclose(sample.thermodynamic_params_guess[0],71.8,rtol=0.1)

def test_different_format_data():

    file2 = './test_files/MX3005P.txt'

    sample.read_multiple_files([file2])

    assert len(sample.conditions) == 48

    assert  'Fluorescence' not in sample.signals
    assert '350nm' in sample.signals

def test_read_same_format_data():

    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')

    assert len(sample.conditions) == 48 * 2
