"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np

from pychemelt import Sample
from pychemelt.utils.signals import signal_two_state_tc_unfolding

from pychemelt.utils.math import quadratic_baseline

def_params = {
    'DHm': 100,
    'Tm': 60,
    'Cp0': 1.6,
    'm0': 2.6,
    'm1': 0,
    'p1_N': -0.1,
    'p2_N': 1.5,
    'p3_N': -0.015,  # Negative temperature dependence for native state
    'p4_N': 0.0001,
    'p1_U': -0.005,
    'p2_U': 2.5,
    'p3_U': -0.025,  # Negative temperature dependence for unfolded state
    'p4_U': 0.0002,
    'baseline_N_fx':quadratic_baseline,
    'baseline_U_fx':quadratic_baseline
}

def_concs = [1e-8,1,1.5,2,2.6,3,4,5]

scalings_factors = np.array([1,0.95,1,1.1,1,1,1,0.96])

def aux_create_pychem_sim(params,concs):

    # Calculate signal range for proper y-axis scaling
    temp_range = np.linspace(20, 80, 60)
    signal_list = []
    temp_list   = []

    for i,D in enumerate(concs):

        y = signal_two_state_tc_unfolding(temp_range, D, **params)

        rng = np.random.default_rng(2)

        # Add gaussian error to signal
        y += rng.normal(0, 0.0005, len(y)) # Small error

        # Add gaussian error to PROTEIN concentration
        y *= scalings_factors[i]

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
    params['p1_N'] = 0
    params['p1_U'] = 0
    params['p3_N'] = 0
    params['p3_U'] = 0
    params['p4_N'] = 0
    params['p4_N'] = 0

    pychem_sim = aux_create_pychem_sim(params,def_concs)

    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='constant',
        unfolded_baseline_type='constant'
    )

    np.testing.assert_allclose(
        pychem_sim.first_param_Ns_per_signal[0][0],
        1.5,
        rtol=0.01,
        atol=0)

    # Reset fittings results
    sample.reset_fittings_results()
    assert len(sample.first_param_Ns_per_signal) == 0


    # ------------ #
    params = def_params.copy()

    params['p1_N'] = 0
    params['p1_U'] = 0
    params['p4_N'] = 0
    params['p4_U'] = 0

    pychem_sim = aux_create_pychem_sim(params,def_concs)

    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='linear'
    )

    np.testing.assert_allclose(pychem_sim.second_param_Ns_per_signal[0][0], params['p3_N'], rtol=0.1, atol=0)
    np.testing.assert_allclose(pychem_sim.second_param_Us_per_signal[0][-1], params['p3_U'], rtol=0.1, atol=0)

    # ------------ #
    params = def_params.copy()
    params['p1_N'] = 0
    params['p1_U'] = 0

    pychem_sim = aux_create_pychem_sim(params,def_concs)

    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='quadratic',
        unfolded_baseline_type='quadratic',
        window_range_native=20,
        window_range_unfolded=20
        )

    np.testing.assert_allclose(pychem_sim.third_param_Ns_per_signal[0][0], params['p4_N'], rtol=0.1, atol=0)

# --------- #  Create global pychem_sim object for the rest of tests  # --------- #
sample = aux_create_pychem_sim(def_params,def_concs)
sample.estimate_derivative()
sample.guess_Tm()
sample.n_residues = 130

def test_fit_thermal_unfolding_local():

    sample.estimate_baseline_parameters(
        native_baseline_type='quadratic',
        unfolded_baseline_type='quadratic',
        window_range_native=16,
        window_range_unfolded=16
    )
    sample.fit_thermal_unfolding_local()

    np.testing.assert_allclose(sample.Tms_multiple[0][0],60.2,rtol=0.05)

def test_guess_Cp():

    sample.guess_Cp()

    np.testing.assert_allclose(sample.Cp0,1.7,rtol=0.05)

def test_guess_initial_parameters():

    sample.guess_initial_parameters(
        native_baseline_type='quadratic',
        unfolded_baseline_type='quadratic',
        window_range_native=16,
        window_range_unfolded=16
    )

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

    sample.global_fit_done = False # Force re-fitting

    sample.fit_thermal_unfolding_global_global()

    expected = [60.2, 100, 1.7, 2.6]
    actual = sample.params_df.iloc[:4,1]

    np.testing.assert_allclose(actual,expected,rtol=0.1)

def test_fit_thermal_unfolding_global_global_global():

    sample.fit_thermal_unfolding_global() # Needs to be done firsts
    sample.global_global_fit_done = False # Force re-fitting clause
    sample.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

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
