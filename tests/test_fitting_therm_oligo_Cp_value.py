"""
Module for testing fitting of thermal unfolding curves with different Cp values.
"""
"""
import numpy as np
import pytest

from pychemelt.utils.fitting import (
    fit_oligomer_unfolding_single_slopes,
    fit_oligomer_unfolding_three_states_single_slopes,
)

from pychemelt.utils.math import constant_baseline

from pychemelt.utils.signals import (
    map_two_state_model_to_signal_fx,
    map_three_state_model_to_signal_fx
)

# Centralized test constants
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 95.0
N_TEMPS = 100
CONCS = np.array([20, 50, 100]) * 1e-6

# Two-state ground-truth parameters
DHm_VAL = 200
Tm_VAL = 65
CP_TRUE = 2.5

# Three-state ground-truth parameters
DH1_VAL = 250
DH2_VAL = 250
T1_VAL = 45
T2_VAL = 75
CP1_TRUE = 0.5
CPTH_TRUE = 1.5

INTERCEPT_N = 50
INTERCEPT_I = 80
INTERCEPT_U = 100

rng = np.random.default_rng(RNG_SEED)

def get_two_state_data(model_name, cp_true):
    signal_fx = map_two_state_model_to_signal_fx(model_name)
    temp_range = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
    temp_range_K = temp_range + 273.15
    
    params = {
        'dHm': DHm_VAL,
        'Tm': Tm_VAL + 273.15,
        'Cp': cp_true,
        'p1_N': 0, 'p2_N': INTERCEPT_N, 'p3_N': 0, 'p4_N': 0,
        'p1_U': 0, 'p2_U': INTERCEPT_U, 'p3_U': 0, 'p4_U': 0,
        'baseline_N_fx': constant_baseline,
        'baseline_U_fx': constant_baseline,
    }
    
    signal_list = []
    temp_list = []
    for C in CONCS:
        y = signal_fx(temp_range_K, C, **params)
        #y += rng.normal(0, 0.002, len(y))
        signal_list.append(y)
        temp_list.append(temp_range)
    
    return temp_list, signal_list, signal_fx

def get_three_state_data(model_name, cp1_true, cp_th_true):
    signal_fx = map_three_state_model_to_signal_fx(model_name)
    temp_range = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
    temp_range_K = temp_range + 273.15
    
    params = {
        'DH1': DH1_VAL, 'DH2': DH2_VAL,
        'T1': T1_VAL + 273.15, 'T2': T2_VAL + 273.15,
        'bI': INTERCEPT_I,
        'p1_N': 0, 'p2_N': INTERCEPT_N, 'p3_N': 0, 'p4_N': 0,
        'p1_U': 0, 'p2_U': INTERCEPT_U, 'p3_U': 0, 'p4_U': 0,
        'baseline_N_fx': constant_baseline,
        'baseline_U_fx': constant_baseline,
        'Cp1': cp1_true,
        'CpTh': cp_th_true,
    }
    
    signal_list = []
    temp_list = []
    for C in CONCS:
        y = signal_fx(temp_range_K, C, **params)
        #y += rng.normal(0, 0.002, len(y))
        signal_list.append(y)
        temp_list.append(temp_range)
        
    return temp_list, signal_list, signal_fx

@pytest.mark.parametrize("model_name", ["Monomer", "Dimer"])
@pytest.mark.parametrize("cp_true, cp_fit", [
    (2.0, 2.0),   # True Cp matches given Cp
    (2.0, 0.0),   # True Cp differs from given Cp
    (0.0, 2.0),   # True Cp is zero, fit with non-zero Cp
])
def test_fitting_therm_oligo_cp_impact(model_name, cp_true, cp_fit):
    temp_list, signal_list, signal_fx = get_two_state_data(model_name, cp_true)
    
    # p0 = [Tm, dHm, Cp, baseline_params...]
    p0 = [Tm_VAL, DHm_VAL, cp_fit] + [INTERCEPT_N] * len(CONCS) + [INTERCEPT_U] * len(CONCS)
    low_bounds = [TEMP_START, 50, 0] + [10] * (2 * len(CONCS))
    high_bounds = [TEMP_STOP, 500, 5] + [200] * (2 * len(CONCS))
    
    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': CONCS,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'cp_value': cp_fit
    }
    

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )
    # If it converges, check it's within bounds
    assert TEMP_START < global_fit_params[0] < TEMP_STOP
    assert global_fit_params[1] > 0


@pytest.mark.parametrize("model_name", ["Monomer_monomeric_intermediate", "Dimer_monomeric_intermediate"])
@pytest.mark.parametrize("cp_th_true, cp_th_fit", [
    (1.5, 1.5),   # True CpTh matches given CpTh
    (1.5, 0.0),   # True CpTh differs from given CpTh
])
def test_fitting_therm_oligo_three_state_cp_impact(model_name, cp_th_true, cp_th_fit):
    cp1_true = 0.5 # A non-zero true Cp1
    temp_list, signal_list, signal_fx = get_three_state_data(model_name, cp1_true, cp_th_true)
    
    # When CpTh_value is NOT None, the parameters are:
    # [Tm1, DHm1, Tm2, DHm2, Cp1, p1N...p1U...bI...]
    # When CpTh_value IS None, Cp1 is fixed to 0.0 and NOT in the parameters.
    
    if cp_th_fit != 0: # Providing a non-zero CpTh_value
        p0 = [T1_VAL, DH1_VAL, T2_VAL, DH2_VAL, cp1_true] + \
             [INTERCEPT_N] * len(CONCS) + [INTERCEPT_U] * len(CONCS) + [INTERCEPT_I] * len(CONCS)
        low_bounds = [TEMP_START, 50, TEMP_START, 50, 0] + [10] * (3 * len(CONCS))
        high_bounds = [TEMP_STOP, 500, TEMP_STOP, 500, 5] + [200] * (3 * len(CONCS))
        cp_th_arg = cp_th_fit
    else:
        # If cp_th_fit is 0, we don't pass it as CpTh_value to the fitter, 
        # so it defaults to None (which sets Cp1=0, CpTh=0)
        p0 = [T1_VAL, DH1_VAL, T2_VAL, DH2_VAL] + \
             [INTERCEPT_N] * len(CONCS) + [INTERCEPT_U] * len(CONCS) + [INTERCEPT_I] * len(CONCS)
        low_bounds = [TEMP_START, 50, TEMP_START, 50] + [10] * (3 * len(CONCS))
        high_bounds = [TEMP_STOP, 500, TEMP_STOP, 500] + [200] * (3 * len(CONCS))
        cp_th_arg = None
        
    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': CONCS,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'CpTh_value': cp_th_arg
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    # We check the fit stays within physical bounds
    assert TEMP_START < global_fit_params[0] < TEMP_STOP
    assert TEMP_START < global_fit_params[2] < TEMP_STOP
    assert global_fit_params[1] > 0
    assert global_fit_params[3] > 0
"""

"""
Module for fitting of thermal unfolding curves.

Tests the fitting methods used by ThermalOligomer for their capabilities of fitting to simulated data
"""

import numpy as np

from pychemelt.utils.fitting import (
    fit_oligomer_unfolding_single_slopes,
    fit_oligomer_unfolding_three_states_single_slopes
)

from pychemelt.utils.math import constant_baseline

from pychemelt.utils.signals import (
    map_two_state_model_to_signal_fx,
    map_three_state_model_to_signal_fx
)

# Centralized test constants
RNG_SEED = 2
TEMP_START = 30.0
TEMP_STOP = 90.0
N_TEMPS = 80
CONCS = np.arange(10, 100, 10)*1e-6

# Two state
# Model / ground-truth parameters
DHm_VAL = 250
Tm_VAL = 70
CP0_VAL = 1.8


INTERCEPT_N = 50
C_N_VAL = 0
INTERCEPT_U = 100
C_U_VAL = 0

rng = np.random.default_rng(RNG_SEED)

def_params = {
    'dHm': DHm_VAL,
    'Tm': Tm_VAL+273.15,
    'Cp': CP0_VAL,
    'p1_N': C_N_VAL,
    'p2_N': INTERCEPT_N,
    'p3_N': 0,
    'p4_N': 0,
    'p1_U': C_U_VAL,
    'p2_U': INTERCEPT_U,
    'p3_U': 0,
    'p4_U': 0,
    'baseline_N_fx':constant_baseline,
    'baseline_U_fx':constant_baseline,
}

# Three state
# Model / ground-truth parameters
DHm_VAL_1 = 300
DHm_VAL_2 = 300
Tm_VAL_1 = 50
Tm_VAL_2 = 70
CP1_VAL = 1.0
CPTH_VAL = 2.0


INTERCEPT_I = 100

INTERCEPT_N = 80
PRE_EXP_N = 0
C_N_VAL = 0
ALPHA_N_VAL = 0


INTERCEPT_U = 110
PRE_EXP_U = 0
C_U_VAL = 0
ALPHA_U_VAL = 0

def_params_three_state = {
    'DH1': DHm_VAL_1,
    'DH2': DHm_VAL_2,
    'T1': Tm_VAL_1+273.15,
    'T2': Tm_VAL_2+273.15,
    'bI': INTERCEPT_I,
    'p1_N': C_N_VAL,
    'p2_N': INTERCEPT_N,
    'p3_N': PRE_EXP_N,
    'p4_N': ALPHA_N_VAL,
    'p1_U': C_U_VAL,
    'p2_U': INTERCEPT_U,
    'p3_U': PRE_EXP_U,
    'p4_U': ALPHA_U_VAL,
    'baseline_N_fx':constant_baseline,
    'baseline_U_fx':constant_baseline,
    "Cp1": 0,
    'CpTh': 0,
}


concs = CONCS

# Calculate signal range for proper y-axis scaling
temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
temp_range_K = temp_range + 273.15

def fitting_data_two_state(oligomer, cp_true, cp_fit, params=def_params):
    signal_fx = map_two_state_model_to_signal_fx(oligomer)

    signal_list = []
    temp_list = []

    params['Cp'] = cp_true

    for C in concs:
        y = signal_fx(temp_range_K, C, **params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL - 100] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL + 100] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'cp_value': cp_fit,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    return global_fit_params, cov, predicted_lst

def fitting_data_three_state(oligomer, cp1_true, cpth_true, cpth_fit, cp1_fit=1.0, params=def_params_three_state):
    signal_fx = map_three_state_model_to_signal_fx(oligomer)

    signal_list = []
    temp_list = []

    params['Cp1'] = cp1_true
    params['CpTh'] = cpth_true

    for C in concs:
        y = signal_fx(temp_range_K, C, **params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    if cpth_fit is None:
        p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
            INTERCEPT_I] * len(concs)
        low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
        high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))
    else:
        p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, cp1_fit] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
            INTERCEPT_I] * len(concs)
        low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100, 0.1] + [1e-5] * (3 * len(concs))
        high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100, cpth_fit] + [1e3] * (3 * len(concs))


    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'CpTh_value': cpth_fit,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    return global_fit_params, cov, predicted_lst

def test_fit_monomer_unfolding_single_slopes_Cp_values():

    oligomer = "Monomer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)


    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)


def test_fit_dimer_unfolding_single_slopes_Cp_values():
    oligomer = "Dimer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)


def test_fit_trimer_unfolding_single_slopes_Cp_values():
    oligomer = "Trimer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)

def test_fit_tetramer_unfolding_single_slopes_Cp_values():
    oligomer = "Tetramer"

    expected = [Tm_VAL, DHm_VAL]

    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 0.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_two_state(oligomer, 2.0, 0.0, params=def_params)

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.2, atol=0)

# Testing three state fitting

def test_fit_monomer_unfolding_three_states_single_slopes_Cp_values():

    oligomer = "Monomer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_dimer_unfolding_three_states_single_slopes_Cp_values():
    oligomer = "Dimer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_trimer_unfolding_three_states_single_slopes_Cp_values():

    oligomer = "Trimer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

def test_fit_tetramer_unfolding_three_states_single_slopes_Cp_values():
    oligomer = "Tetramer_monomeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_dimer_dimeric_unfolding_three_states_single_slopes_Cp_values():
    oligomer = "Dimer_dimeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_trimer_trimeric_unfolding_three_states_single_slopes_Cp_values():

    oligomer = "Trimer_trimeric_intermediate"

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    # True == expected Cp0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, 2.0, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


    # True and expected Cp0 == 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, None, params=def_params_three_state)


    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 2 and assumed is 0

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 1.0, 2.0, None, params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # True cp is 0 and assumed is 2

    global_fit_params, cov, predicted_lst = fitting_data_three_state(oligomer, 0.0, 0.0, 2.0,
                                                                     params=def_params_three_state)

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)
