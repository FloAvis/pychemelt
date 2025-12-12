import numpy as np

from pychemelt.utils.fitting import (
    fit_thermal_unfolding,
    fit_tc_unfolding_single_slopes,
    fit_tc_unfolding_shared_slopes_many_signals,
    fit_tc_unfolding_many_signals
)

from pychemelt.utils.math import exponential_baseline

from pychemelt.utils.signals import (
    signal_two_state_t_unfolding,
    signal_two_state_tc_unfolding
)

# Centralized test constants
RNG_SEED = 2
TEMP_START = 30.0
TEMP_STOP = 90.0
N_TEMPS = 80
CONCS = [0.01, 1, 2, 2.6, 3, 4, 5]

# Model / ground-truth parameters
DHm_VAL = 120
Tm_VAL = 65
CP0_VAL = 1.8
M0_VAL = 2.6
M1_VAL = 0

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
    'DHm': DHm_VAL,
    'Tm': Tm_VAL,
    'Cp0': CP0_VAL,
    'm0': M0_VAL,
    'm1': M1_VAL,
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
signal_list = []
temp_list   = []

for D in concs:

    y = signal_two_state_tc_unfolding(temp_range, D, **def_params)

    # Add gaussian error to signal
    y += rng.normal(0, 0.005, len(y))

    signal_list.append(y)
    temp_list.append(temp_range)


def test_fit_thermal_unfolding_exponential():

    p0 = [Tm_VAL - 5, INTERCEPT_N] + [1]*6
    low_bounds = [TEMP_START, TEMP_START]   + [-np.inf]*6
    high_bounds = [TEMP_STOP, 200] + [np.inf]*6

    global_fit_params, cov, predicted_lst = fit_thermal_unfolding(
        list_of_temperatures=temp_list[:1],
        list_of_signals=signal_list[:1],
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        signal_fx=signal_two_state_t_unfolding,
        baseline_native_fx=exponential_baseline,
        baseline_unfolded_fx=exponential_baseline,
        Cp=0,
    )

    expected = [Tm_VAL, DHm_VAL]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)


def test_fit_tc_unfolding_single_slopes_exponential():

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL, M0_VAL] + [0.1]*(6*len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1, 1]   + [1e-5]*(6*len(concs))
    high_bounds = [TEMP_STOP, 200, 5, 5] + [1e3]*(6*len(concs))

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [Tm_VAL, DHm_VAL, CP0_VAL, M0_VAL]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=Tm_VAL,
        **kwargs
    )

    expected = [DHm_VAL, CP0_VAL, M0_VAL]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed DH and fixed Tm and fixed Cp - allow fit of m1
    p0_2 = p0_tm[2:]
    low_bounds_2 = low_bounds_tm[2:]
    high_bounds_2 = high_bounds_tm[2:]

    p0_2.insert(1, 0)
    low_bounds_2.insert(1, -0.5)
    high_bounds_2.insert(1, 0.5)

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
        initial_parameters=p0_2,
        low_bounds=low_bounds_2,
        high_bounds=high_bounds_2,
        dh_value=DHm_VAL,
        tm_value=Tm_VAL,
        cp_value=CP0_VAL,
        fit_m1=True,
        **kwargs
    )

    np.testing.assert_allclose(global_fit_params[0], M0_VAL, rtol=0.1, atol=0)
    # End of - Fit with fixed DH


def test_fit_tc_unfolding_shared_slopes_many_signals_exponential():

    # p0 includes initial parameters for m, m1 and the shared intercepts
    # The pre-exp and exp coefficients are fitted separately
    p0 = [M0_VAL, M1_VAL, INTERCEPT_N, INTERCEPT_U] + [1,1]*len(temp_list) + [0,0]*len(temp_list) + [ALPHA_N_VAL, ALPHA_U_VAL]*len(temp_list)

    low_bounds = [-1      for _ in p0]
    high_bounds = [np.inf for _ in p0]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline,
        'fit_m1' : True,
        'tm_value' : Tm_VAL,
        'dh_value' : DHm_VAL,
        'cp_value' : CP0_VAL,
        'signal_ids' : [0 for _ in range(len(signal_list))]
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_shared_slopes_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    np.testing.assert_allclose(global_fit_params[0], M0_VAL, rtol=0.1, atol=0)


def test_fit_tc_unfolding_many_signals_exponential():

    # p0 includes initial parameters for m, m1 and the shared intercepts
    # The pre-exp and exp coefficients are fitted separately
    p0 = [Tm_VAL,DHm_VAL,M0_VAL,M1_VAL]
    p0 += [INTERCEPT_N, INTERCEPT_U]
    p0 += [ALPHA_N_VAL, ALPHA_U_VAL]
    p0 += [PRE_EXP_N, PRE_EXP_U]
    p0 += [C_N_VAL, C_U_VAL]
    p0 += [ALPHA_N_VAL,ALPHA_U_VAL]

    low_bounds = [-0.1 for _ in p0]
    high_bounds = [1e3 for _ in p0]

    M1_VAL_index = 3
    low_bounds[M1_VAL_index] = -0.05
    high_bounds[M1_VAL_index] = 0.05

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'model_scale_factor': False,
        'cp_value' : CP0_VAL,
        'fit_m1' : True,
        'baseline_native_fx':exponential_baseline,
        'baseline_unfolded_fx':exponential_baseline
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_many_signals(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    np.testing.assert_allclose(global_fit_params[:4], p0[:4], rtol=0.1, atol=1e-2)
