"""
Module for fitting of thermal unfolding curves with three states.

Tests the fitting methods used by ThermalOligomer for their capabilities of fitting to simulated data
"""

import numpy as np

from pychemelt.utils.fitting import (
    fit_oligomer_unfolding_three_states_single_slopes,
    fit_oligomer_unfolding_three_states_shared_slopes_many_signals,
    fit_oligomer_unfolding_three_states_many_signals,
    evaluate_need_to_refit_three_state
)

from pychemelt.utils.math import constant_baseline

from pychemelt.utils.signals import (
    map_three_state_model_to_signal_fx
)

# Centralized test constants
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 90.0
N_TEMPS = 150
CONCS = np.arange(10, 80, 10)*1e-6

# Model / ground-truth parameters
DHm_VAL_1 = 300
DHm_VAL_2 = 300
Tm_VAL_1 = 50
Tm_VAL_2 = 70

INTERCEPT_I = 100

INTERCEPT_N = 80
PRE_EXP_N = 0
C_N_VAL = 0
ALPHA_N_VAL = 0


INTERCEPT_U = 110
PRE_EXP_U = 0
C_U_VAL = 0
ALPHA_U_VAL = 0

rng = np.random.default_rng(RNG_SEED)

def_params = {
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
    "Cp1": 0.5,
    'CpTh': 1.0,
}

concs = CONCS

# Calculate signal range for proper y-axis scaling
temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
temp_range_K = temp_range + 273.15


def test_fit_monomer_unfolding_three_states_single_slopes_constant():
    signal_fx = map_three_state_model_to_signal_fx("Monomer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100]   + [1e-5]*(3*len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3]*(3*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':constant_baseline,
        'baseline_unfolded_fx':constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm1 and Tm2
    
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()
    
    
    
    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm


    # Fit with fixed dH1 and dh2
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()


    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH


def test_fit_dimer_unfolding_three_states_single_slopes_constant():
    signal_fx = map_three_state_model_to_signal_fx("Dimer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

    


def test_fit_trimer_unfolding_three_states_single_slopes_constant():
    signal_fx = map_three_state_model_to_signal_fx("Trimer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

    

def test_fit_tetramer_unfolding_three_states_single_slopes_constant():
    signal_fx = map_three_state_model_to_signal_fx("Tetramer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()


    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH


def test_fit_dimer_dimeric_unfolding_three_states_single_slopes_constant():
    signal_fx = map_three_state_model_to_signal_fx("Dimer_dimeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH


def test_fit_trimer_trimeric_unfolding_three_states_single_slopes_constant():
    signal_fx = map_three_state_model_to_signal_fx("Trimer_trimeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH


# Test fitting global slope

def test_fit_monomer_unfolding_three_states_shared_slopes_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Monomer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100]   + [1e-5]*(3*len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3]*(3*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':constant_baseline,
        'baseline_unfolded_fx':constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH
    
def test_fit_dimer_unfolding_three_states_shared_slopes_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Dimer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

def test_fit_trimer_unfolding_three_states_shared_slopes_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Trimer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

def test_fit_tetramer_unfolding_three_states_shared_slopes_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Tetramer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

def test_fit_dimer_dimeric_unfolding_three_states_shared_slopes_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Dimer_dimeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

def test_fit_trimer_trimeric_unfolding_three_states_shared_slopes_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Trimer_trimeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] *len(concs) + [INTERCEPT_U] * len(concs) + [INTERCEPT_I] * len(concs)
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()



    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        t1=Tm_VAL_1,
        t2=Tm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_shared_slopes_many_signals(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh1=DHm_VAL_1,
        dh2=DHm_VAL_2,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

# Test fitting global slope and baselines

def test_fit_monomer_unfolding_three_states_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Monomer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N, INTERCEPT_U, INTERCEPT_I]
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [-1e2] * 3
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * 3

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids = scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    

def test_fit_dimer_unfolding_three_states_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Dimer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N, INTERCEPT_U, INTERCEPT_I]
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [-1e2] * 3
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * 3

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    

def test_fit_trimer_unfolding_three_states_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Trimer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N, INTERCEPT_U, INTERCEPT_I]
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [-1e2] * 3
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * 3

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    

def test_fit_tetramer_unfolding_three_states_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Tetramer_monomeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)
        # Add gaussian error to signal
        y += rng.normal(0, 0.002*1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N, INTERCEPT_U, INTERCEPT_I]
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [-1e2] * 3
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * 3

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_dimer_dimeric_unfolding_three_states_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Dimer_dimeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N, INTERCEPT_U, INTERCEPT_I]
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [-1e2] * 3
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * 3

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)


def test_fit_trimer_trimeric_unfolding_three_states_many_signals_constant():
    signal_fx = map_three_state_model_to_signal_fx("Trimer_trimeric_intermediate")

    signal_list = []
    temp_list = []

    for C in concs:
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.002 * 1e-3, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N, INTERCEPT_U, INTERCEPT_I]
    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [-1e2] * 3
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * 3

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_three_states_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

def test_refit_three_state_model_constant():

    p0 = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
        INTERCEPT_I] * len(concs)

    # Test refitting of three state model Tm1 and Tm2 overlapping and DH1 adn DH2 too close

    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    global_fit_params = p0.copy()

    global_fit_params[0] = -5
    global_fit_params[2] = -5

    low_bounds[1] = DHm_VAL_1 - 5
    high_bounds[1] = DHm_VAL_1 + 5

    low_bounds[3] = DHm_VAL_2 - 5
    high_bounds[3] = DHm_VAL_2 + 5

    re_fit, _, _, _ = evaluate_need_to_refit_three_state(
        global_fit_params,
        high_bounds,
        low_bounds,
        p0,
        check_dh=True,
        check_tm=True,
    )

    assert re_fit == True

    # Test refitting of three state model Tm1 and Tm2 too close

    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    global_fit_params = p0.copy()

    global_fit_params[0] = 80
    global_fit_params[2] = 60

    low_bounds[0] = 80 - 5
    high_bounds[0] = 80 + 5

    low_bounds[2] = 60 - 5
    high_bounds[2] = 60 + 5

    re_fit, _, _, _ = evaluate_need_to_refit_three_state(
        global_fit_params,
        high_bounds,
        low_bounds,
        p0,
        check_dh=True,
        check_tm=True,
    )

    assert re_fit == True

    # Test refitting of three state model baseline parameter too close

    low_bounds = [TEMP_START, DHm_VAL_1 - 100, TEMP_START, DHm_VAL_2 - 100] + [1e-5] * (3 * len(concs))
    high_bounds = [TEMP_STOP, DHm_VAL_1 + 100, TEMP_STOP, DHm_VAL_2 + 100] + [1e3] * (3 * len(concs))

    global_fit_params = p0.copy()

    low_bounds[4] = INTERCEPT_N - 1
    high_bounds[4] = INTERCEPT_N + 1

    re_fit, _, _, _ = evaluate_need_to_refit_three_state(
        global_fit_params,
        high_bounds,
        low_bounds,
        p0,
        check_dh=True,
        check_tm=True,
    )

    assert re_fit == True
