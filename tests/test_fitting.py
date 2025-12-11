import numpy as np

from pychemelt.utils.fitting import (
    fit_line_robust,
    fit_quadratic_robust,
    fit_exponential_robust,
    fit_thermal_unfolding,
    fit_tc_unfolding_single_slopes,
    fit_tc_unfolding_shared_slopes_many_signals,
    fit_tc_unfolding_many_signals
)

from pychemelt.utils.signals import (
    signal_two_state_tc_unfolding_monomer,
    signal_two_state_t_unfolding_monomer
)

# Centralized test constants (so the file has no scattered hardcoded values)
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 80.0
N_TEMPS = 80
CONCS = [1e-8, 1, 1.5, 2, 2.6, 3, 4, 5]

# Model / ground-truth parameters used across tests
DHm_VAL = 100.0
Tm_VAL = 60.0
CP0_VAL = 1.6
M0_VAL = 2.6
M1_VAL = 0.0
A_N_VAL = 1.5
B_N_VAL = -0.015
C_N_VAL = -0.1
D_N_VAL = 0.0001
A_U_VAL = 2.5
B_U_VAL = -0.025
C_U_VAL = -0.005
D_U_VAL = 0.0002

# Small test-specific constants
LINE_M = 30.0
LINE_B = 10.0
QUAD_A = 1.0
QUAD_B = 2.0
QUAD_C = 3.0
EXP_A = 10.0
EXP_C = 1.0
EXP_ALPHA = 0.01

rng = np.random.default_rng(RNG_SEED)

### Create datasets for the tests ###

params = {
    'DHm': DHm_VAL,
    'Tm': Tm_VAL,
    'Cp0': CP0_VAL,
    'm0': M0_VAL,
    'm1': M1_VAL,
    'a_N': A_N_VAL,
    'b_N': B_N_VAL,  # Negative temperature dependence for native state
    'c_N': C_N_VAL,
    'd_N': D_N_VAL,
    'a_U': A_U_VAL,
    'b_U': B_U_VAL,  # Negative temperature dependence for unfolded state
    'c_U': C_U_VAL,
    'd_U': D_U_VAL
}

params_no_temp_slopes = params.copy()
params_no_temp_slopes['b_N'] = 0
params_no_temp_slopes['b_U'] = 0
params_no_temp_slopes['d_N'] = 0
params_no_temp_slopes['d_U'] = 0

params_no_den_slopes = params.copy()
params_no_den_slopes['c_N'] = 0
params_no_den_slopes['c_U'] = 0

concs = CONCS

# Calculate signal range for proper y-axis scaling
temp_range = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
signal_list = []
signal_list_2 = [] # Signal where the temperature dependence is removed
signal_list_3 = [] # Signal where the denaturant concentration dependence is removed
temp_list   = []

for D in concs:

    y = signal_two_state_tc_unfolding_monomer(temp_range, D, **params)
    y2 = signal_two_state_tc_unfolding_monomer(temp_range,D,**params_no_temp_slopes)
    y3 = signal_two_state_tc_unfolding_monomer(temp_range,D,**params_no_den_slopes)

    # Add gaussian error to signal
    y += rng.normal(0, 0.01, len(y))
    y2 += rng.normal(0, 0.01, len(y2))
    y3 += rng.normal(0, 0.01, len(y3))

    # Add gaussian error to PROTEIN concentration
    y *= rng.normal(1, 0.001)
    y2 *= rng.normal(1, 0.001)
    y3 *= rng.normal(1, 0.001)

    signal_list.append(y)
    signal_list_2.append(y2)
    signal_list_3.append(y3)
    temp_list.append(temp_range)

### End of create datasets for the tests ###
###
###

def test_fit_line_robust():

    m = LINE_M
    b = LINE_B

    x = np.linspace(0,10,100)
    y = m * x + b
    y = y + rng.normal(0,0.1,100)

    m_fit, b_fit = fit_line_robust(x, y)

    try:
        np.testing.assert_allclose(m_fit, m, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_line_robust FAILED for slope: expected {m!r}, got {m_fit!r}")
        raise

    try:
        np.testing.assert_allclose(b_fit, b, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_line_robust FAILED for intercept: expected {b!r}, got {b_fit!r}")
        raise


def test_fit_quadratic_robust():

    a = QUAD_A
    b = QUAD_B
    c = QUAD_C
    x = np.linspace(0,10,100)

    y = a * x ** 2 + b * x + c
    y = y + rng.normal(0,0.1,100)

    a_fit, b_fit, c_fit = fit_quadratic_robust(x, y)

    try:
        np.testing.assert_allclose([a_fit, b_fit, c_fit], [a, b, c], rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_quadratic_robust FAILED: expected {[a,b,c]!r}, got {[a_fit,b_fit,c_fit]!r}")
        raise


def test_fit_exponential_robust():

    a = EXP_A
    c = EXP_C
    alpha = EXP_ALPHA
    x = np.linspace(0,100,100)

    y = a + c * np.exp(-alpha * x)

    y = y + rng.normal(0,0.01,100)

    a_fit, c_fit, alpha_fit = fit_exponential_robust(x, y)

    try:
        np.testing.assert_allclose([a_fit, c_fit, alpha_fit], [a, c, alpha], rtol=0.1, atol=0)
    except AssertionError:
        print(f"test_fit_exponential_robust FAILED: expected {[a,c,alpha]!r}, got {[a_fit,c_fit,alpha_fit]!r}")
        raise


def test_fit_thermal_unfolding():

    initial_parameters = [Tm_VAL,DHm_VAL] + [1]*6
    low_bounds = [TEMP_START,TEMP_START]          + [-np.inf]*6
    high_bounds = [TEMP_STOP,200]        + [np.inf]*6

    # Fit only the lowest concentration
    global_fit_params, cov, predicted_lst = fit_thermal_unfolding(
            temp_list[:1],
            signal_list[:1],
            initial_parameters,
            low_bounds,
            high_bounds,
            signal_two_state_t_unfolding_monomer,
            CP0_VAL,
            fit_slopes = {
                'fit_slope_native' : True,
                'fit_slope_unfolded' : True,
                'fit_quadratic_native' : True,
                'fit_quadratic_unfolded' : True
            },
            list_of_oligomer_conc=None)

    # The Tm at a concentration close to zero should be the Tm
    try:
        np.testing.assert_allclose(global_fit_params[0], Tm_VAL, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_thermal_unfolding FAILED for Tm: expected {Tm_VAL}, got {global_fit_params[0]!r}")
        raise

    # Same for DH
    try:
        np.testing.assert_allclose(global_fit_params[1], DHm_VAL, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_thermal_unfolding FAILED for DH: expected {DHm_VAL}, got {global_fit_params[1]!r}")

    # Same for the intercept bN
    try:
        np.testing.assert_allclose(global_fit_params[2], A_N_VAL, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_thermal_unfolding FAILED for bN: expected {A_N_VAL}, got {global_fit_params[2]!r}")
        raise

def test_fit_tc_unfolding_single_slopes():

    # Tm, Dh, Cp, m0
    initial_parameters = [Tm_VAL,DHm_VAL,CP0_VAL,M0_VAL] + [1]*(len(concs)*6) # Times six, because of bN, bU, kN, kU, qN, qU
    low_bounds = [TEMP_START,TEMP_START,0,0] + [-np.inf]*(len(concs)*6)
    high_bounds = [TEMP_STOP,200,5,5] + [np.inf]*(len(concs)*6)

    fit_slopes_dic = {
        'fit_slope_native': True,
        'fit_slope_unfolded': True,
        'fit_quadratic_native': True,
        'fit_quadratic_unfolded': True
    }

    kwargs = {

        'fit_slopes' : fit_slopes_dic,
        'list_of_temperatures':temp_list,
        'list_of_signals':signal_list,
        'denaturant_concentrations':concs,
        'signal_fx':signal_two_state_tc_unfolding_monomer,
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
            initial_parameters=initial_parameters,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            **kwargs
    )

    expected = [Tm_VAL,DHm_VAL,CP0_VAL,M0_VAL]

    # Verify the fitting
    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Now do fitting with fixed Tm
    initial_parameters_tm = initial_parameters.copy()[1:]
    low_bounds_tm = low_bounds.copy()[1:]
    high_bounds_tm = high_bounds.copy()[1:]

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
            initial_parameters=initial_parameters_tm,
            low_bounds=low_bounds_tm,
            high_bounds=high_bounds_tm,
            tm_value=Tm_VAL,
            **kwargs
    )

    expected = [DHm_VAL,CP0_VAL,M0_VAL]

    # Verify the fitting
    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # Fitting with fixed DH
    initial_parameters_dh = initial_parameters.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    # Remove the Dh parameter at index 1
    initial_parameters_dh.pop(1)
    low_bounds_dh.pop(1)
    high_bounds_dh.pop(1)

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
            initial_parameters=initial_parameters_dh,
            low_bounds=low_bounds_dh,
            high_bounds=high_bounds_dh,
            dh_value=DHm_VAL,
            **kwargs
    )

    expected = [Tm_VAL,CP0_VAL,M0_VAL]

    # Verify the fitting
    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)


def test_fit_tc_unfolding_shared_slopes_many_signals():

    # Tm, Dh, Cp, m0
    initial_parameters = [M0_VAL,M1_VAL] + [1]*(len(concs)*6) # Times six, because of bN, bU, kN, kU, qN, qU
    low_bounds = [0,-1] + [-np.inf]*(len(concs)*6)
    high_bounds = [5,1] + [np.inf]*(len(concs)*6)

    fit_slopes_dic = {
        'fit_slope_native': True,
        'fit_slope_unfolded': True,
        'fit_quadratic_native': True,
        'fit_quadratic_unfolded': True
    }

    kwargs = {

        'fit_slopes' : fit_slopes_dic,
        'list_of_temperatures':temp_list,
        'list_of_signals':signal_list,
        'denaturant_concentrations':concs,
        'signal_fx':signal_two_state_tc_unfolding_monomer,
        'tm_value':Tm_VAL,
        'dh_value':DHm_VAL,
        'cp_value':CP0_VAL,
        'signal_ids':[0 for _ in range(len(signal_list))],
        'fit_m1':True
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_shared_slopes_many_signals(
            initial_parameters=initial_parameters,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            **kwargs
    )

    # Verify the fitting
    np.testing.assert_allclose(global_fit_params[0], M0_VAL, rtol=0.1, atol=0)


def test_fit_tc_unfolding_shared_slopes_many_signals_no_slopes():

    # Tm, Dh, Cp, m0
    initial_parameters = [M0_VAL,M1_VAL] + [1]*(len(concs)*2) # Times six, because of bN, bU, kN, kU, qN, qU
    low_bounds = [0,-1] + [-np.inf]*(len(concs)*2)
    high_bounds = [5,1] + [np.inf]*(len(concs)*2)

    fit_slopes_dic = {
        'fit_slope_native': False,
        'fit_slope_unfolded': False,
        'fit_quadratic_native': False,
        'fit_quadratic_unfolded': False
    }

    kwargs = {

        'fit_slopes' : fit_slopes_dic,
        'list_of_temperatures':temp_list,
        'list_of_signals':signal_list_2, # No dependence on temperature
        'denaturant_concentrations':concs,
        'signal_fx':signal_two_state_tc_unfolding_monomer,
        'tm_value':Tm_VAL,
        'dh_value':DHm_VAL,
        'cp_value':CP0_VAL,
        'signal_ids':[0 for _ in range(len(signal_list))],
        'fit_m1':True
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_shared_slopes_many_signals(
            initial_parameters=initial_parameters,
            low_bounds=low_bounds,
            high_bounds=high_bounds,
            **kwargs
    )

    # Verify the fitting
    np.testing.assert_allclose(global_fit_params[0], M0_VAL, rtol=0.1, atol=0)

def test_fit_tc_unfolding_many_signals():

    expected = [Tm_VAL,DHm_VAL,CP0_VAL,M0_VAL,A_N_VAL,A_U_VAL,B_N_VAL,B_U_VAL,C_N_VAL,C_U_VAL,D_N_VAL,D_U_VAL]

    p0 = [Tm_VAL-10,Tm_VAL+20,1,3,A_N_VAL,A_U_VAL,B_N_VAL,B_U_VAL,C_N_VAL,C_U_VAL,D_N_VAL,D_U_VAL] + [1] * len(temp_list)

    low_bounds = [-1e3 for _ in range(len(p0))]
    high_bounds = [1e3 for _ in range(len(p0))]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding_monomer,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'fit_slope_native_temp': True,
        'fit_slope_unfolded_temp': True,
        'fit_quadratic_native': True,
        'fit_quadratic_unfolded': True,
        'initial_parameters' : p0,
        'low_bounds' : low_bounds,
        'high_bounds' : high_bounds,
        'model_scale_factor': True
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_many_signals(**kwargs)

    np.testing.assert_allclose(global_fit_params[:4], expected[:4], rtol=0.1, atol=0)

    # Test with fixed params

    expected = [Tm_VAL,DHm_VAL,M0_VAL,M1_VAL,A_N_VAL,A_U_VAL,B_N_VAL,B_U_VAL,C_N_VAL,C_U_VAL,D_N_VAL,D_U_VAL]

    p0 = expected + [0.95] * len(temp_list)
    low_bounds = [-1e3 for _ in range(len(p0))]
    high_bounds = [1e3 for _ in range(len(p0))]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding_monomer,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'fit_slope_native_temp': True,
        'fit_slope_unfolded_temp': True,
        'fit_quadratic_native': True,
        'fit_quadratic_unfolded': True,
        'initial_parameters' : p0,
        'low_bounds' : low_bounds,
        'high_bounds' : high_bounds,
        'model_scale_factor': True,
        'cp_value':CP0_VAL,
        'fit_m1':True
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_many_signals(**kwargs)

    np.testing.assert_allclose(global_fit_params[:3], expected[:3], rtol=0.1, atol=0)

def test_fit_tc_unfolding_many_signals_no_slopes():

    expected = [Tm_VAL,DHm_VAL,CP0_VAL,M0_VAL,A_N_VAL,A_U_VAL,C_N_VAL,C_U_VAL]

    p0 = expected + [1.3] * len(temp_list)
    p0[0] = Tm_VAL - 10
    p0[1] = DHm_VAL + 80

    low_bounds = [-1e3 for _ in range(len(p0))]
    high_bounds = [1e3 for _ in range(len(p0))]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list_2,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding_monomer,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'fit_slope_native_temp': False,
        'fit_slope_unfolded_temp': False,
        'fit_quadratic_native': False,
        'fit_quadratic_unfolded': False,
        'initial_parameters' : p0,
        'low_bounds' : low_bounds,
        'high_bounds' : high_bounds,
        'model_scale_factor': True
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_many_signals(**kwargs)

    np.testing.assert_allclose(global_fit_params[:4], expected[:4], rtol=0.1, atol=0)

    expected = [Tm_VAL,DHm_VAL,CP0_VAL,M0_VAL,A_N_VAL,A_U_VAL,B_N_VAL,B_U_VAL,D_N_VAL,D_U_VAL]

    p0 = expected + [1.3] * len(temp_list)
    p0[0] = Tm_VAL - 10
    p0[1] = DHm_VAL + 80

    low_bounds = [-1e3 for _ in range(len(p0))]
    high_bounds = [1e3 for _ in range(len(p0))]

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list_3,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding_monomer,
        'signal_ids' : [0 for _ in range(len(signal_list))],
        'fit_slope_native_temp': True,
        'fit_slope_unfolded_temp': True,
        'fit_quadratic_native': True,
        'fit_quadratic_unfolded': True,
        'fit_slope_native_den' : False,
        'fit_slope_unfolded_den' : False,
        'initial_parameters' : p0,
        'low_bounds' : low_bounds,
        'high_bounds' : high_bounds,
        'model_scale_factor': True
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_many_signals(**kwargs)

    np.testing.assert_allclose(global_fit_params[:4], expected[:4], rtol=0.1, atol=0)
