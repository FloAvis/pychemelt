import numpy as np

from pychemelt.utils.fitting import (
    fit_oligomer_unfolding_single_slopes,
    fit_oligomer_unfolding_shared_slopes_many_signals,
    fit_oligomer_unfolding_many_signals,
)

from pychemelt.utils.math import constant_baseline

from pychemelt.utils.signals import (
    map_two_state_model_to_signal_fx
)

# Centralized test constants
RNG_SEED = 2
TEMP_START = 30.0
TEMP_STOP = 90.0
N_TEMPS = 80
CONCS = np.arange(10, 100, 10)*1e-6

# Model / ground-truth parameters
DHm_VAL = 150
Tm_VAL = 70
CP0_VAL = 1.8


INTERCEPT_N = 100
C_N_VAL = 0
INTERCEPT_U = 110
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
    'baseline_U_fx':constant_baseline

}

concs = CONCS

# Calculate signal range for proper y-axis scaling
temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
temp_range_K = temp_range + 273.15


def test_fit_monomer_unfolding_single_slopes_constant():
    signal_fx = map_two_state_model_to_signal_fx("Monomer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1]*(2*len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1]   + [1e-5]*(2*len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3]*(2*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':constant_baseline,
        'baseline_unfolded_fx':constant_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm


    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    p0_dh.pop(1)
    low_bounds_dh.pop(1)
    high_bounds_dh.pop(1)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh_value=DHm_VAL,
        **kwargs
    )

    expected = [Tm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

    # Fit with fixed Cp
    p0_cp = p0.copy()
    low_bounds_cp = low_bounds.copy()
    high_bounds_cp = high_bounds.copy()

    p0_cp.pop(2)
    low_bounds_cp.pop(2)
    high_bounds_cp.pop(2)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_cp,
        low_bounds=low_bounds_cp,
        high_bounds=high_bounds_cp,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Cp

def test_fit_dimer_unfolding_single_slopes_constant():
    signal_fx = map_two_state_model_to_signal_fx("Dimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    p0_dh.pop(1)
    low_bounds_dh.pop(1)
    high_bounds_dh.pop(1)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh_value=DHm_VAL,
        **kwargs
    )

    expected = [Tm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

    # Fit with fixed Cp
    p0_cp = p0.copy()
    low_bounds_cp = low_bounds.copy()
    high_bounds_cp = high_bounds.copy()

    p0_cp.pop(2)
    low_bounds_cp.pop(2)
    high_bounds_cp.pop(2)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_cp,
        low_bounds=low_bounds_cp,
        high_bounds=high_bounds_cp,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Cp


def test_fit_trimer_unfolding_single_slopes_constant():
    signal_fx = map_two_state_model_to_signal_fx("Trimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    p0_dh.pop(1)
    low_bounds_dh.pop(1)
    high_bounds_dh.pop(1)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh_value=DHm_VAL,
        **kwargs
    )

    expected = [Tm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

    # Fit with fixed Cp
    p0_cp = p0.copy()
    low_bounds_cp = low_bounds.copy()
    high_bounds_cp = high_bounds.copy()

    p0_cp.pop(2)
    low_bounds_cp.pop(2)
    high_bounds_cp.pop(2)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_cp,
        low_bounds=low_bounds_cp,
        high_bounds=high_bounds_cp,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Cp

def test_fit_tetramer_unfolding_single_slopes_constant():
    signal_fx = map_two_state_model_to_signal_fx("Tetramer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed dH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    p0_dh.pop(1)
    low_bounds_dh.pop(1)
    high_bounds_dh.pop(1)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh_value=DHm_VAL,
        **kwargs
    )

    expected = [Tm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed dH

    # Fit with fixed Cp
    p0_cp = p0.copy()
    low_bounds_cp = low_bounds.copy()
    high_bounds_cp = high_bounds.copy()

    p0_cp.pop(2)
    low_bounds_cp.pop(2)
    high_bounds_cp.pop(2)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0_cp,
        low_bounds=low_bounds_cp,
        high_bounds=high_bounds_cp,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Cp


# Test fitting global slope

def test_fit_monomer_unfolding_shared_slopes_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Monomer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1]*(2*len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1]   + [1e-5]*(2*len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3]*(2*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':constant_baseline,
        'baseline_unfolded_fx':constant_baseline,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

def test_fit_dimer_unfolding_shared_slopes_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Dimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm


def test_fit_trimer_unfolding_shared_slopes_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Trimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm


def test_fit_tetramer_unfolding_shared_slopes_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Tetramer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm


# Test fitting global slope and baselines

def test_fit_monomer_unfolding_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Monomer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START, 1] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * 4)

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'fit_native_olig_slope' : False,
        'fit_unfolded_olig_slope' : False,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids = scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit cp value set

    p0 = [Tm_VAL, DHm_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200] + [1e3] * (2 * 4)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

def test_fit_dimer_unfolding_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Dimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START, 1] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * 4)

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'fit_native_olig_slope': False,
        'fit_unfolded_olig_slope': False,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit cp value set

    p0 = [Tm_VAL, DHm_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200] + [1e3] * (2 * 4)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

def test_fit_trimer_unfolding_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Trimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START, 1] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * 4)

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'fit_native_olig_slope': False,
        'fit_unfolded_olig_slope': False,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit cp value set

    p0 = [Tm_VAL, DHm_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200] + [1e3] * (2 * 4)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

def test_fit_tetramer_unfolding_many_signals_constant():
    signal_fx = map_two_state_model_to_signal_fx("Tetramer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START, 1] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * 4)

    kwargs = {
        'list_of_temperatures': temp_list,
        'list_of_signals': signal_list,
        'signal_ids': [0 for _ in range(len(signal_list))],
        'oligomer_concentrations': concs,
        'signal_fx': signal_fx,
        'baseline_native_fx': constant_baseline,
        'baseline_unfolded_fx': constant_baseline,
        'fit_native_olig_slope': False,
        'fit_unfolded_olig_slope': False,
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit scale factor

    scale_factors = [1 for _ in range(len(signal_list) - 1)]
    scale_factors_low = [0.5882 for _ in range(len(signal_list) - 1)]
    scale_factors_high = [1.7 for _ in range(len(signal_list) - 1)]

    p0 = np.concatenate([p0, scale_factors])
    low_bounds = np.concatenate([low_bounds, scale_factors_low])
    high_bounds = np.concatenate([high_bounds, scale_factors_high])

    scale_factor_exclude_ids = [len(signal_list) - 1]

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        model_scale_factor=True,
        scale_factor_exclude_ids=scale_factor_exclude_ids,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fit cp value set

    p0 = [Tm_VAL, DHm_VAL] + [100, 110] + [0] * (2 * 3)
    low_bounds = [TEMP_START, TEMP_START] + [-1e2] * (2 * 4)
    high_bounds = [TEMP_STOP, 200] + [1e3] * (2 * 4)

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        cp_value=CP0_VAL,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)