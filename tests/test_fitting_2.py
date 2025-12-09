import numpy as np

from pychemelt.utils.fitting import (
    fit_thermal_unfolding_exponential,
    fit_tc_unfolding_single_slopes_exponential
)

from pychemelt.utils.signals import (
    signal_two_state_t_unfolding_monomer_exponential,
    signal_two_state_tc_unfolding_monomer_exponential
)

rng = np.random.default_rng(2)

def_params = {
    'DHm': 120,
    'Tm': 65,
    'Cp0': 1.8,
    'm0': 2.6,
    'm1': 0,
    'intercept_n': 100,
    'pre_exp_n':1,
    'c_N':0,
    'alpha_N':0.1,
    'intercept_u':110,
    'pre_exp_u':1,
    'c_U':0,
    'alpha_U':0.2
}

concs = [0.01,1,2,2.6,3,4,5]

# Calculate signal range for proper y-axis scaling
temp_range  = np.linspace(30, 90, 80)
signal_list = []
temp_list   = []

for D in concs:

    y = signal_two_state_tc_unfolding_monomer_exponential(temp_range, D, **def_params)

    # Add gaussian error to signal
    y += rng.normal(0, 0.005, len(y))

    # Add gaussian error to PROTEIN concentration
    y *= rng.normal(1, 0.001)

    signal_list.append(y)
    temp_list.append(temp_range)


def test_fit_thermal_unfolding_exponential():

    p0 = [60,100] + [1]*6
    low_bounds = [30,30]   + [-np.inf]*6
    high_bounds = [70,200] + [np.inf]*6

    global_fit_params, cov, predicted_lst = fit_thermal_unfolding_exponential(
        temp_list[:1],
        signal_list[:1],
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        signal_fx=signal_two_state_t_unfolding_monomer_exponential,
    )

    expected = [65,120]

    np.testing.assert_allclose(global_fit_params[:2], expected, rtol=0.1, atol=0)

def test_fit_tc_unfolding_single_slopes_exponential():

    p0 = [65,120,1.8,2.6]      + [1]*(6*7)
    low_bounds = [30,30,1,1]   + [1e-5]*(6*7)
    high_bounds = [70,200,5,5] + [1e3]*(6*7)

    kwargs = {
        'list_of_temperatures' : temp_list,
        'list_of_signals' : signal_list,
        'denaturant_concentrations' : concs,
        'signal_fx' : signal_two_state_tc_unfolding_monomer_exponential,
    }

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes_exponential(
        initial_parameters=p0,
        low_bounds=low_bounds,
        high_bounds=high_bounds,
        **kwargs
    )

    expected = [65,120,1.8,2.6]

    np.testing.assert_allclose(global_fit_params[:4], expected, rtol=0.1, atol=0)

    # Fit with fixed Tm
    p0_tm = p0.copy()
    low_bounds_tm = low_bounds.copy()
    high_bounds_tm = high_bounds.copy()

    p0_tm.pop(0)
    low_bounds_tm.pop(0)
    high_bounds_tm.pop(0)

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes_exponential(
        initial_parameters=p0_tm,
        low_bounds=low_bounds_tm,
        high_bounds=high_bounds_tm,
        tm_value=65,
        **kwargs
    )

    expected = [120,1.8,2.6]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)

    # End of - Fit with fixed Tm

    # Fit with fixed DH
    p0_dh = p0.copy()
    low_bounds_dh = low_bounds.copy()
    high_bounds_dh = high_bounds.copy()

    p0_dh.pop(1)
    low_bounds_dh.pop(1)
    high_bounds_dh.pop(1)

    global_fit_params, cov, predicted_lst = fit_tc_unfolding_single_slopes_exponential(
        initial_parameters=p0_dh,
        low_bounds=low_bounds_dh,
        high_bounds=high_bounds_dh,
        dh_value=120,
        **kwargs
    )

    expected = [65,1.8,2.6]

    np.testing.assert_allclose(global_fit_params[:3], expected, rtol=0.1, atol=0)
    # End of - Fit with fixed DH

    # Fit with fixed Cp
    p0_cp = p0.copy()
    low_bounds_cp = low_bounds.copy()
    high_bounds_cp = high_bounds.copy()