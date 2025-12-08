"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np

from pychemelt import Sample
from pychemelt.utils.signals import signal_two_state_tc_unfolding_monomer

def_params = {
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

def_concs = [1e-8,1,1.5,2,2.6,3,4,5]

def aux_create_pychem_sim(params,concs):

    # Calculate signal range for proper y-axis scaling
    temp_range = np.linspace(20, 80, 60)
    signal_list = []
    temp_list   = []

    for D in concs:

        y = signal_two_state_tc_unfolding_monomer(temp_range, D, **params)

        rng = np.random.default_rng(2)

        # Add gaussian error to signal
        y += rng.normal(0, 0.0005, len(y)) # Small error

        # Add gaussian error to PROTEIN concentration
        y *= rng.normal(1, 0.001)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = Sample()

    pychem_sim.signal_dic['Fluo'] = signal_list
    pychem_sim.temp_dic['Fluo']   = [temp_range for _ in range(len(concs))]

    pychem_sim.conditions = concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_denaturant_concentrations()

    pychem_sim.set_signal('Fluo')

    pychem_sim.select_conditions(normalise_to_global_max=False)
    pychem_sim.expand_multiple_signal()

    return pychem_sim

def test_estimate_baseline_parameters():

    params = def_params.copy()

    # Set fluorescence dependence on temperature and denaturant concentration to zero
    params['b_N'] = 0
    params['b_U'] = 0
    params['c_N'] = 0
    params['c_U'] = 0
    params['d_N'] = 0
    params['d_U'] = 0

    pychem_sim = aux_create_pychem_sim(params,def_concs)

    pychem_sim.estimate_baseline_parameters(poly_order_native=0, poly_order_unfolded=0)

    np.testing.assert_allclose(pychem_sim.bNs_per_signal[0][0],params['a_N'], rtol=0.01, atol=0)

    # Reset fittings results
    sample.reset_fittings_results()
    assert len(sample.bNs_per_signal) == 0


    # ------------ #
    params = def_params.copy()

    params['c_N'] = 0
    params['c_U'] = 0
    params['d_N'] = 0
    params['d_U'] = 0

    pychem_sim = aux_create_pychem_sim(params,def_concs)

    pychem_sim.estimate_baseline_parameters(poly_order_native=1, poly_order_unfolded=1)

    np.testing.assert_allclose(pychem_sim.kNs_per_signal[0][0],params['b_N'], rtol=0.1, atol=0)
    np.testing.assert_allclose(pychem_sim.kUs_per_signal[0][-1],params['b_U'], rtol=0.1, atol=0)

    # ------------ #
    params = def_params.copy()
    params['c_N'] = 0
    params['c_U'] = 0

    pychem_sim = aux_create_pychem_sim(params,def_concs)

    pychem_sim.estimate_baseline_parameters(20,poly_order_native=2, poly_order_unfolded=2)

    np.testing.assert_allclose(pychem_sim.qNs_per_signal[0][0],params['d_N'], rtol=0.1, atol=0)

# --------- #  Create global pychem_sim object for the rest of tests  # --------- #
sample = aux_create_pychem_sim(def_params,def_concs)
sample.estimate_derivative()
sample.guess_Tm()
sample.n_residues = 130

def test_fit_thermal_unfolding_local():

    sample.estimate_baseline_parameters(16,poly_order_native=2, poly_order_unfolded=2)
    sample.fit_thermal_unfolding_local()

    np.testing.assert_allclose(sample.Tms_multiple[0][0],60.2,rtol=0.05)

def test_guess_Cp():

    sample.guess_Cp()

    np.testing.assert_allclose(sample.Cp0,1.7,rtol=0.05)

def test_guess_initial_parameters():

    sample.guess_initial_parameters(poly_order_native=2, poly_order_unfolded=2)

    np.testing.assert_allclose(sample.Cp0,1.7,rtol=0.2)

def test_fit_thermal_unfolding_global():

    sample.fit_thermal_unfolding_global()

    assert sample.params_df is not None

    assert sample.params_df.shape[0] == 52

    expected = [60.2, 100, 1.7, 2.6]
    actual   = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

    args_dic = {
        'fit_m_dep': True,
        'dh_limits': [50,200],
        'tm_limits': [40,80],
        'cp_limits': [0.5,4]
    }

    for key,val in args_dic.items():

        sample.fit_thermal_unfolding_global(**{key:val})
        actual = sample.params_df.iloc[:4,1]
        np.testing.assert_allclose(actual,expected,rtol=0.1)

    # -- Fit with fixed Cp -- #
    sample.fit_thermal_unfolding_global(cp_value=1.7)

    expected = [60.2, 100, 2.6]
    actual = sample.params_df.iloc[:3,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

def test_fit_thermal_unfolding_global_global():

    sample.set_signal_id()

    # Re-do fit without constraints
    sample.fit_thermal_unfolding_global()

    sample.fit_thermal_unfolding_global_global()

    expected = [60.2, 100, 1.7, 2.6]
    actual = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

def test_fit_thermal_unfolding_global_global_global():

    for model_scale_factor in [True,False]:

        sample.fit_thermal_unfolding_global() # Needs to be done firsts
        sample.fit_thermal_unfolding_global_global() # Needs to be done first
        sample.fit_thermal_unfolding_global_global_global(model_scale_factor=model_scale_factor)

        expected = [60.2, 100, 1.7, 2.6]
        actual = sample.params_df.iloc[:4,1]

        np.testing.assert_allclose(actual,expected,rtol=0.1)

def test_signal_to_df():

    signal_type_options = ['raw','derivative']

    for signal_type in signal_type_options:

        df = sample.signal_to_df(signal_type=signal_type, scaled=False)

        assert len(df) == 480

    signal_type_options = ['raw','fitted']

    for signal_type in signal_type_options:

        df = sample.signal_to_df(signal_type=signal_type, scaled=True)

        assert len(df) == 480
        assert np.max(df['Signal']) <= 100
