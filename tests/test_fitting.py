import numpy as np

from pychemelt.utils.fitting import (
    fit_line_robust,
    fit_quadratic_robust,
    fit_exponential_robust,
    fit_thermal_unfolding,
    fit_tc_unfolding_single_slopes
)

from pychemelt.utils.signals import (
    signal_two_state_tc_unfolding_monomer,
    signal_two_state_t_unfolding_monomer
)

def test_fit_line_robust():

    m = 30
    b = 10

    x = np.linspace(0,10,100)
    y = m * x + b
    y = y + np.random.normal(0,0.1,100)

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

    a = 1
    b = 2
    c = 3
    x = np.linspace(0,10,100)

    y = a * x ** 2 + b * x + c
    y = y + np.random.normal(0,0.1,100)

    a_fit, b_fit, c_fit = fit_quadratic_robust(x, y)

    try:
        np.testing.assert_allclose([a_fit, b_fit, c_fit], [a, b, c], rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_quadratic_robust FAILED: expected {[a,b,c]!r}, got {[a_fit,b_fit,c_fit]!r}")
        raise


def test_fit_exponential_robust():

    a = 10
    c = 1
    alpha = 0.01
    x = np.linspace(0,100,100)

    y = a + c * np.exp(-alpha * x)

    y = y + np.random.normal(0,0.01,100)

    a_fit, c_fit, alpha_fit = fit_exponential_robust(x, y)

    try:
        np.testing.assert_allclose([a_fit, c_fit, alpha_fit], [a, c, alpha], rtol=0.1, atol=0)
    except AssertionError:
        print(f"test_fit_exponential_robust FAILED: expected {[a,c,alpha]!r}, got {[a_fit,c_fit,alpha_fit]!r}")
        raise

params = {
    'DHm': 100,
    'Tm': 60,
    'Cp0': 1.6,
    'm0': 2.6,
    'm1': 0,
    'a_N': 1.5,
    'b_N': -0.015,  # Negative temperature dependence for native state
    'c_N': -0.1,
    'd_N': 0.0001,
    'a_U': 2.5,
    'b_U': -0.025,  # Negative temperature dependence for unfolded state
    'c_U': -0.005,
    'd_U': 0.0002
}

concs = [1e-8,1,1.5,2,2.6,3,4,5]

# Calculate signal range for proper y-axis scaling
temp_range = np.linspace(20, 80, 80)
signal_list = []
temp_list   = []

for D in concs:

    y = signal_two_state_tc_unfolding_monomer(temp_range, D, **params)

    # Add gaussian error to signal
    y += np.random.normal(0, 0.005, len(y))

    # Add gaussian error to PROTEIN concentration
    y *= np.random.normal(1, 0.001)

    signal_list.append(y)
    temp_list.append(temp_range)

def test_fit_thermal_unfolding():

    initial_parameters = [60,100] + [1]*6
    low_bounds = [30,30]          + [-np.inf]*6
    high_bounds = [70,200]        + [np.inf]*6

    # Fit only the lowest concentration
    global_fit_params, cov, predicted_lst = fit_thermal_unfolding(
            temp_list[:1],
            signal_list[:1],
            initial_parameters,
            low_bounds,
            high_bounds,
            signal_two_state_t_unfolding_monomer,
            1.6,
            fit_slopes = {
                'fit_slope_native' : True,
                'fit_slope_unfolded' : True,
                'fit_quadratic_native' : True,
                'fit_quadratic_unfolded' : True
            },
            list_of_oligomer_conc=None)

    # The Tm at a concentration close to zero should be the Tm
    try:
        np.testing.assert_allclose(global_fit_params[0], 60, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_thermal_unfolding FAILED for Tm: expected 60, got {global_fit_params[0]!r}")
        raise

    # Same for DH
    try:
        np.testing.assert_allclose(global_fit_params[1], 100, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_thermal_unfolding FAILED for DH: expected 100, got {global_fit_params[1]!r}")

    # Same for the intercept bN
    try:
        np.testing.assert_allclose(global_fit_params[2], 1.5, rtol=0.05, atol=0)
    except AssertionError:
        print(f"test_fit_thermal_unfolding FAILED for bN: expected 1.5, got {global_fit_params[2]!r}")
        raise

def test_fit_tc_unfolding_single_slopes():

    # Tm, Dh, Cp, m0
    initial_parameters = [60,100,1.6,2.6] + [1]*(len(concs)*6) # Times six, because of bN, bU, kN, kU, qN, qU
    low_bounds = [30,30,0,0] + [-np.inf]*(len(concs)*6)
    high_bounds = [80,200,5,5] + [np.inf]*(len(concs)*6)

    # Fit only the lowest concentration
    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
            temp_list,
            signal_list,
            concs,
            initial_parameters,
            low_bounds,
            high_bounds,
            signal_two_state_tc_unfolding_monomer,
            fit_slopes = {
                'fit_slope_native' : True,
                'fit_slope_unfolded' : True,
                'fit_quadratic_native' : True,
                'fit_quadratic_unfolded' : True
            },
            list_of_oligomer_conc=None)

    # Verify the Tm fitting
    np.testing.assert_allclose(global_fit_params[0], 60, rtol=0.1, atol=0)

    # Verify DH fitting
    np.testing.assert_allclose(global_fit_params[1], 100, rtol=0.1, atol=0)

    # Verify Cp fitting
    np.testing.assert_allclose(global_fit_params[2], 1.6, rtol=0.1, atol=0)

    # Verify m0 fitting
    np.testing.assert_allclose(global_fit_params[3], 2.6, rtol=0.1, atol=0)

    # Now do fitting with fixed Tm
    initial_parameters_tm = initial_parameters.copy()[1:]
    low_bounds_tm = low_bounds.copy()[1:]
    high_bounds_tm = high_bounds.copy()[1:]

    # Fit only the lowest concentration
    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes(
            temp_list,
            signal_list,
            concs,
            initial_parameters_tm,
            low_bounds_tm,
            high_bounds_tm,
            signal_two_state_tc_unfolding_monomer,
            fit_slopes = {
                'fit_slope_native' : True,
                'fit_slope_unfolded' : True,
                'fit_quadratic_native' : True,
                'fit_quadratic_unfolded' : True
            },
            tm_value=60,
            list_of_oligomer_conc=None)

    # Verify DH fitting
    np.testing.assert_allclose(global_fit_params[0], 100, rtol=0.1, atol=0)

    # Verify Cp fitting
    np.testing.assert_allclose(global_fit_params[1], 1.6, rtol=0.1, atol=0)

    # Verify m0 fitting
    np.testing.assert_allclose(global_fit_params[2], 2.6, rtol=0.1, atol=0)