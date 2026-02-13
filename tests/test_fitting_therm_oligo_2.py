import numpy as np

from pychemelt.utils.fitting import (
    fit_oligomer_unfolding_single_slopes,
    fit_oligomer_unfolding_shared_slopes_many_signals,
    fit_oligomer_unfolding_many_signals,
)

from pychemelt.utils.math import exponential_baseline

from pychemelt.utils.signals import (
    map_two_state_model_to_signal_fx
)

# Centralized test constants
RNG_SEED = 2
TEMP_START = 27.0
TEMP_STOP = 90.0
N_TEMPS = 80
CONCS = np.arange(10, 100, 10)*1e-6

# Model / ground-truth parameters
DHm_VAL = 150
Tm_VAL = 70
CP0_VAL = 1.8

INTERCEPT_N = 100
PRE_EXP_N = 1
C_N_VAL = 0
ALPHA_N_VAL = 0.1
INTERCEPT_U = 110
PRE_EXP_U = 1
C_U_VAL = 0
ALPHA_U_VAL = 0.2

rng = np.random.default_rng(RNG_SEED)

def_params = {
    'dHm': DHm_VAL,
    'Tm': Tm_VAL+273.15,
    'Cp': CP0_VAL,
    'p1_N': C_N_VAL,
    'p2_N': INTERCEPT_N,
    'p3_N': PRE_EXP_N,
    'p4_N': ALPHA_N_VAL,
    'p1_U': C_U_VAL,
    'p2_U': INTERCEPT_U,
    'p3_U': PRE_EXP_U,
    'p4_U': ALPHA_U_VAL,
    'baseline_N_fx':exponential_baseline,
    'baseline_U_fx':exponential_baseline

}

concs = CONCS

# Calculate signal range for proper y-axis scaling
temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
temp_range_K = temp_range + 273.15

def test_fit_monomer_unfolding_single_slopes_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Monomer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1]*(6*len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1]   + [1e-5]*(6*len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3]*(6*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

def test_fit_dimer_unfolding_single_slopes_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Dimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)
    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1]*(6*len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1]   + [1e-5]*(6*len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3]*(6*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

def test_fit_trimer_unfolding_single_slopes_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Trimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)
    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1]*(6*len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1]   + [1e-5]*(6*len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3]*(6*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

def test_fit_tetramer_unfolding_single_slopes_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Tetramer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1]*(6*len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1]   + [1e-5]*(6*len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3]*(6*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)



def test_fit_monomer_unfolding_shared_slopes_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Monomer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
        PRE_EXP_N] + [PRE_EXP_U] + [ALPHA_N_VAL] + [ALPHA_U_VAL]

    low_bounds = [-1      for _ in p0]
    high_bounds = [np.inf for _ in p0]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline,
        'signal_ids' : [0 for _ in range(len(signal_list))]
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

def test_fit_dimer_unfolding_shared_slopes_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Dimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
        PRE_EXP_N] + [PRE_EXP_U] + [ALPHA_N_VAL] + [ALPHA_U_VAL]

    low_bounds = [-1      for _ in p0]
    high_bounds = [np.inf for _ in p0]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline,
        'signal_ids' : [0 for _ in range(len(signal_list))]
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

def test_fit_trimer_unfolding_shared_slopes_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Trimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
        PRE_EXP_N] + [PRE_EXP_U] + [ALPHA_N_VAL] + [ALPHA_U_VAL]

    low_bounds = [-1      for _ in p0]
    high_bounds = [np.inf for _ in p0]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline,
        'signal_ids' : [0 for _ in range(len(signal_list))]
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

def test_fit_tetramer_unfolding_shared_slopes_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Tetramer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [INTERCEPT_N] * len(concs) + [INTERCEPT_U] * len(concs) + [
        PRE_EXP_N] + [PRE_EXP_U] + [ALPHA_N_VAL] + [ALPHA_U_VAL]
    low_bounds = [-1      for _ in p0]
    high_bounds = [np.inf for _ in p0]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline,
        'signal_ids' : [0 for _ in range(len(signal_list))]
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)
    
    

def test_fit_monomer_unfolding_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Monomer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)


    p0 = [Tm_VAL,DHm_VAL]
    p0 += [INTERCEPT_N, INTERCEPT_U]
    p0 += [ALPHA_N_VAL, ALPHA_U_VAL]
    p0 += [PRE_EXP_N, PRE_EXP_U]
    p0 += [C_N_VAL, C_U_VAL]
    p0 += [ALPHA_N_VAL,ALPHA_U_VAL]

    low_bounds = [-0.1 for _ in p0]
    high_bounds = [1e3 for _ in p0]


    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'model_scale_factor': False,
        'cp_value' : CP0_VAL,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    np.testing.assert_allclose(global_fit_params[:2], [Tm_VAL,DHm_VAL], rtol=0.1, atol=1e-2)

def test_fit_dimer_unfolding_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Dimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)


    p0 = [Tm_VAL,DHm_VAL]
    p0 += [INTERCEPT_N, INTERCEPT_U]
    p0 += [ALPHA_N_VAL, ALPHA_U_VAL]
    p0 += [PRE_EXP_N, PRE_EXP_U]
    p0 += [C_N_VAL, C_U_VAL]
    p0 += [ALPHA_N_VAL,ALPHA_U_VAL]

    low_bounds = [-0.1 for _ in p0]
    high_bounds = [1e3 for _ in p0]


    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'model_scale_factor': False,
        'cp_value' : CP0_VAL,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    np.testing.assert_allclose(global_fit_params[:2], [Tm_VAL,DHm_VAL], rtol=0.1, atol=1e-2)

def test_fit_trimer_unfolding_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Trimer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)


    p0 = [Tm_VAL,DHm_VAL]
    p0 += [INTERCEPT_N, INTERCEPT_U]
    p0 += [ALPHA_N_VAL, ALPHA_U_VAL]
    p0 += [PRE_EXP_N, PRE_EXP_U]
    p0 += [C_N_VAL, C_U_VAL]
    p0 += [ALPHA_N_VAL,ALPHA_U_VAL]

    low_bounds = [-0.1 for _ in p0]
    high_bounds = [1e3 for _ in p0]


    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'model_scale_factor': False,
        'cp_value' : CP0_VAL,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    np.testing.assert_allclose(global_fit_params[:2], [Tm_VAL,DHm_VAL], rtol=0.1, atol=1e-2)

def test_fit_tetramer_unfolding_many_signals_exponential():
    signal_fx = map_two_state_model_to_signal_fx("Tetramer")

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL,DHm_VAL]
    p0 += [INTERCEPT_N, INTERCEPT_U]
    p0 += [ALPHA_N_VAL, ALPHA_U_VAL]
    p0 += [PRE_EXP_N, PRE_EXP_U]
    p0 += [C_N_VAL, C_U_VAL]
    p0 += [ALPHA_N_VAL,ALPHA_U_VAL]

    low_bounds = [-0.1 for _ in p0]
    high_bounds = [1e3 for _ in p0]


    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'oligomer_concentrations' : concs,
        'signal_fx' : signal_fx,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'model_scale_factor': False,
        'cp_value' : CP0_VAL,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_oligomer_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    np.testing.assert_allclose(global_fit_params[:2], [Tm_VAL,DHm_VAL], rtol=0.1, atol=1e-2)
