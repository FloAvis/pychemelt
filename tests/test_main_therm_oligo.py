"""
Test file for thermal unfolding of oligomeres based difference in concentrations
"""

import numpy as np

import pytest

from pychemelt.thermal_oligomer import ThermalOligomer

from pychemelt.utils.math import linear_baseline, exponential_baseline, quadratic_baseline

from pychemelt.utils.signals import (
    map_two_state_model_to_signal_fx
)


# Centralized test constants
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 90.0
N_TEMPS = 100
CONCS = [0.01, 1, 2, 2.6, 3, 4, 5]

# Model / ground-truth parameters
DHm_VAL = 100
Tm_VAL = 50
CP0_VAL = 1.8


INTERCEPT_N = 24
SLOPE_N = -0.27
C_N_VAL = 0
INTERCEPT_U = -4
SLOPE_U = 80.5
EXPONENT_U = 0.0224
C_U_VAL = 0

rng = np.random.default_rng(RNG_SEED)

def_params = {
    'dHm': DHm_VAL,
    'Tm': Tm_VAL+273.15,
    'Cp': CP0_VAL,
    'p1_N': C_N_VAL,
    'p2_N': INTERCEPT_N,
    'p3_N': SLOPE_N,
    'p4_N': 0,
    'p1_U': C_U_VAL,
    'p2_U': INTERCEPT_U,
    'p3_U': SLOPE_U,
    'p4_U': EXPONENT_U,
    'baseline_N_fx':linear_baseline,
    'baseline_U_fx':exponential_baseline

}

concs = CONCS

def aux_create_pychem_sim(params,concs, model):

    signal_fx = map_two_state_model_to_signal_fx(model)

    # Calculate signal range for proper y-axis scaling
    temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
    temp_range_K = temp_range + 273.15

    signal_list = []
    temp_list   = []

    # Use a seeded Generator for reproducible noise in tests
    rng = np.random.default_rng(2)

    for D in concs:

        y = signal_fx(temp_range_K, D, **params)

        y += rng.normal(0, 0.0005, len(y)) # Small error (seeded)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = ThermalOligomer()

    pychem_sim.set_model(model)

    pychem_sim.signal_dic['Fluo'] = signal_list
    pychem_sim.temp_dic['Fluo']   = [temp_range for _ in range(len(concs))]

    pychem_sim.conditions = concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_concentrations()

    pychem_sim.set_signal(['Fluo'])

    pychem_sim.select_conditions(normalise_to_global_max=False)
    pychem_sim.expand_multiple_signal()



    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    pychem_sim.n_residues = 150  # only for cp initial guess
    pychem_sim.guess_Cp()

    return pychem_sim


sample = ThermalOligomer()
sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')


def test_set_model():

    sample.set_model("Monomer")
    assert sample.model == "Monomer"

    sample.set_model("Dimer")
    assert sample.model == "Dimer"

    sample.set_model("dimer")
    assert sample.model == "Dimer"

    pytest.raises(ValueError, sample.set_model, "test_false")


def test_set_concentrations():

    sample.set_concentrations()

    assert np.min(sample.oligomer_concentrations_pre) == 0
    assert np.max(sample.oligomer_concentrations_pre) == 8.24

def test_select_conditions():
    sample.set_signal(['350nm'])

    # Select without scaling
    sample.select_conditions(
        [False for _ in range(24)] + [True for _ in range(8)] + [False for _ in range(16)],
        normalise_to_global_max=False
    )

    assert len(sample.signal_lst_multiple) == 1
    assert len(sample.signal_lst_multiple[0]) == 8
    assert np.max(sample.signal_lst_multiple[0]) != 1.0

    # Select with scaling
    sample.select_conditions(
        [False for _ in range(24)] + [True for _ in range(8)] + [False for _ in range(16)],
        normalise_to_global_max=True
    )

    assert len(sample.signal_lst_multiple) == 1
    assert len(sample.signal_lst_multiple[0]) == 8
    assert np.max(sample.signal_lst_multiple[0]) == 100

def test_guess_Cp():

    sample.n_residues = 150
    sample.guess_Cp()

    assert sample.Cp0 == pytest.approx(2.0933, rel=1e-2)

    sample.n_residues = 0
    pytest.raises(ValueError, sample.guess_Cp)

    sample.n_residues = 5

    sample.guess_Cp()
    assert sample.Cp0 == pytest.approx(0, rel=1e-2)


def test_fit_thermal_unfolding_global_global_global_failure():

    sample = ThermalOligomer()
    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')

    pytest.raises(ValueError, sample.fit_thermal_unfolding_global_global_global)

def test_fit_thermal_unfolding_global_global_global_scaling():
    model = "Monomer"
    rng = np.random.default_rng(RNG_SEED)

    #Using concentrations close to each other in order to trigger non-scaling
    scale_concs = [0.999999999999999, 1.00000000000000000001]

    temp_range = np.linspace(20, 90, 100)
    temp_range_K = temp_range + 273.15

    signal_list = []
    temp_list = []

    signal_fx = map_two_state_model_to_signal_fx(model)

    for i, C in enumerate(scale_concs):
        y = signal_fx(temp_range_K, C, **def_params)

        # Add gaussian error to simulated signal
        y += rng.normal(0, 0.02, len(y))

        # Add error to the initial signal to model variance across positions
        y *= rng.uniform(0.9, 1.1)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = ThermalOligomer()

    pychem_sim.signal_dic['Simulated signal'] = signal_list
    pychem_sim.temp_dic['Simulated signal'] = [temp_range for _ in range(len(scale_concs))]

    pychem_sim.set_model(model)
    pychem_sim.conditions = scale_concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_concentrations()

    pychem_sim.set_signal('Simulated signal')

    pychem_sim.select_conditions(normalise_to_global_max=True)
    pychem_sim.expand_multiple_signal()

    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    pychem_sim.n_residues = 130  # only for cp initial guess
    pychem_sim.guess_Cp()

    pychem_sim.fit_thermal_unfolding_global()
    pychem_sim.fit_thermal_unfolding_global_global()
    pychem_sim.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

    assert pychem_sim.params_df is not None

# Testing Monomer model

monomer_sim = aux_create_pychem_sim(def_params, concs, "Monomer")

def test_fit_thermal_unfolding_global_monomer():

    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    monomer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3,1], expected, rtol=0.1, atol=0)

    # fixed Tm limits

    monomer_sim.fit_thermal_unfolding_global(tm_limits=[Tm_VAL-12, Tm_VAL+20])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed dh limits

    monomer_sim.fit_thermal_unfolding_global(dh_limits=[10, 500])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp limits

    monomer_sim.fit_thermal_unfolding_global(cp_limits=[0.1, 5])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp

    expected = [Tm_VAL, DHm_VAL]

    monomer_sim.fit_thermal_unfolding_global(cp_value=CP0_VAL)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:2, 1], expected, rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_monomer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, SLOPE_N, SLOPE_U, EXPONENT_U]

    monomer_sim.fit_thermal_unfolding_global()

    monomer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[[0, 1, 2, 17, 18, 19], 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_monomer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, INTERCEPT_N, INTERCEPT_U, SLOPE_N, SLOPE_U, EXPONENT_U]

    monomer_sim.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[[0, 1, 2, 3, 4, 5, 6, 9], 1], expected,
                               rtol=0.1, atol=0)

# Testing Dimer model

dimer_sim = aux_create_pychem_sim(def_params, concs, "Dimer")

def test_fit_thermal_unfolding_global_dimer():
    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    dimer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed Tm limits

    monomer_sim.fit_thermal_unfolding_global(tm_limits=[Tm_VAL-12, Tm_VAL+20])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed dh limits

    monomer_sim.fit_thermal_unfolding_global(dh_limits=[10, 500])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp limits

    monomer_sim.fit_thermal_unfolding_global(cp_limits=[0.1, 5])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp

    expected = [Tm_VAL, DHm_VAL]

    monomer_sim.fit_thermal_unfolding_global(cp_value=CP0_VAL)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:2, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_global_dimer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, SLOPE_N, SLOPE_U, EXPONENT_U]

    monomer_sim.fit_thermal_unfolding_global()

    dimer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[[0, 1, 2, 17, 18, 19], 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_dimer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, INTERCEPT_N, INTERCEPT_U, SLOPE_N, SLOPE_U, EXPONENT_U]

    dimer_sim.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(dimer_sim.params_df.iloc[[0, 1, 2, 3, 4, 5, 6, 9], 1], expected,
                               rtol=0.1, atol=0)


# Testing Trimer model

trimer_sim = aux_create_pychem_sim(def_params, concs, "Trimer")

def test_fit_thermal_unfolding_global_trimer():
    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    trimer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed Tm limits

    monomer_sim.fit_thermal_unfolding_global(tm_limits=[Tm_VAL-12, Tm_VAL+20])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed dh limits

    monomer_sim.fit_thermal_unfolding_global(dh_limits=[10, 500])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp limits

    monomer_sim.fit_thermal_unfolding_global(cp_limits=[0.1, 5])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp

    expected = [Tm_VAL, DHm_VAL]

    monomer_sim.fit_thermal_unfolding_global(cp_value=CP0_VAL)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:2, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_global_trimer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, SLOPE_N, SLOPE_U, EXPONENT_U]

    trimer_sim.fit_thermal_unfolding_global()

    trimer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[[0, 1, 2, 17, 18, 19], 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_trimer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, INTERCEPT_N, INTERCEPT_U, SLOPE_N, SLOPE_U, EXPONENT_U]

    trimer_sim.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(trimer_sim.params_df.iloc[[0, 1, 2, 3, 4, 5, 6, 9], 1], expected,
                               rtol=0.1, atol=0)


# Testing Tetramer model

tetramer_sim = aux_create_pychem_sim(def_params, concs, "Tetramer")

def test_fit_thermal_unfolding_global_tetramer():
    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    tetramer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed Tm limits

    monomer_sim.fit_thermal_unfolding_global(tm_limits=[Tm_VAL - 12, Tm_VAL + 20])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed dh limits

    monomer_sim.fit_thermal_unfolding_global(dh_limits=[10, 500])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp limits

    monomer_sim.fit_thermal_unfolding_global(cp_limits=[0.1, 5])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)

    # fixed cp

    expected = [Tm_VAL, DHm_VAL]

    monomer_sim.fit_thermal_unfolding_global(cp_value=CP0_VAL)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:2, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_global_tetramer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, SLOPE_N, SLOPE_U, EXPONENT_U]

    tetramer_sim.fit_thermal_unfolding_global()

    tetramer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[[0, 1, 2, 17, 18, 19], 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_tetramer():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL, INTERCEPT_N, INTERCEPT_U, SLOPE_N, SLOPE_U, EXPONENT_U]

    tetramer_sim.fit_thermal_unfolding_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[[0, 1, 2, 3, 4, 5, 6, 9], 1], expected,
                               rtol=0.1, atol=0)

def test_signal_to_df():

    signal_type_options = ['raw','derivative']

    for signal_type in signal_type_options:

        df = monomer_sim.signal_to_df(signal_type=signal_type, scaled=False)

        assert len(df) == 700

    signal_type_options = ['raw','fitted']

    for signal_type in signal_type_options:

        df = monomer_sim.signal_to_df(signal_type=signal_type, scaled=True)

        assert len(df) == 700
        assert np.max(df['Signal']) <= 100


        monomer_sim.max_points = 200

        df = monomer_sim.signal_to_df(signal_type=signal_type, scaled=False)

        assert df is not None