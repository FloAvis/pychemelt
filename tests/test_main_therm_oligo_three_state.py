"""
Module for thermal unfolding simulation and testing with given concentration and model parameters.

This module includes functionalities for simulating thermal unfolding using the ThermalOligomer class,
applying two-state models for signal mapping, baseline estimation, and curve fitting. Various test
functions are provided to validate the implementation, including parameter fitting for monomer
models and scaling behaviors.

Classes:
    ThermalOligomer: Represents the system for thermal unfolding simulations and parameter fitting.

Notes
-----
- The tests rely on seeded random number generators for reproducible results.
- Thermal unfolding behaviors can accommodate different baselines (linear, exponential).
- The module utilizes the `pytest` library for test development.

"""

import numpy as np

import pytest

from pychemelt.thermal_oligomer import ThermalOligomer

from pychemelt.utils.math import linear_baseline, exponential_baseline

from pychemelt.utils.signals import (
    map_three_state_model_to_signal_fx
)


# Centralized test constants
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 90.0
N_TEMPS = 150
CONCS = np.arange(10, 60, 10)*1e-6
MAX_POINTS = 400
slope_position = 4 + 3 * len(CONCS)


# Model / ground-truth parameters
DHm_VAL_1 = 250
DHm_VAL_2 = 250
Tm_VAL_1 = 60
Tm_VAL_2 = 70

INTERCEPT_I = 15

INTERCEPT_N = 24
SLOPE_N = -0.27
C_N_VAL = 0
INTERCEPT_U = -4
SLOPE_U = 80.5
EXPONENT_U = 0.0224
C_U_VAL = 0

rng = np.random.default_rng(RNG_SEED)

def_params = {
    'DH1': DHm_VAL_1,
    'DH2': DHm_VAL_2,
    'T1': Tm_VAL_1+273.15,
    'T2': Tm_VAL_2+273.15,
    'bI': INTERCEPT_I,
    'p1_N': C_N_VAL,
    'p2_N': INTERCEPT_N,
    'p3_N': SLOPE_N,
    'p4_N': 0,
    'p1_U': C_U_VAL,
    'p2_U': INTERCEPT_U,
    'p3_U': SLOPE_U,
    'p4_U': EXPONENT_U,
    'baseline_N_fx':linear_baseline,
    'baseline_U_fx':exponential_baseline,
    "Cp1": 0.5,
    'CpTh': 1.0,
}

concs = CONCS

def aux_create_pychem_sim(params,concs, model, intermediate):

    signal_fx = map_three_state_model_to_signal_fx(model + "_" + intermediate + "_intermediate")

    # Calculate signal range for proper y-axis scaling
    temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
    temp_range_K = temp_range + 273.15

    signal_list = []
    temp_list   = []

    # Use a seeded Generator for reproducible noise in tests
    rng = np.random.default_rng(2)

    for C in concs:

        y = signal_fx(temp_range_K, C, **params)

        # Add gaussian error to signal
        #y += rng.normal(0, 0.002*1e-3, len(y)) # Small error (seeded)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = ThermalOligomer()

    pychem_sim.signal_dic['Fluo'] = signal_list
    pychem_sim.temp_dic['Fluo']   = [temp_range for _ in range(len(concs))]

    pychem_sim.set_model(model, intermediate)

    pychem_sim.conditions = concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_concentrations()

    pychem_sim.set_signal('Fluo')

    pychem_sim.select_conditions()
    pychem_sim.expand_multiple_signal()



    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    return pychem_sim

# Test scaling case for close signals

# Using concentrations close to each other in order to trigger non-scaling
scale_concs = [0.999999999999999, 1.00000000000000000001]

pychem_sim_scaling = aux_create_pychem_sim(def_params, scale_concs, "Monomer", "monomeric")


def test_fit_thermal_unfolding_three_state_global_global_global_scaling():

    pychem_sim_scaling.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    assert pychem_sim_scaling.params_df is not None


# Test not scaling for one concentration

# Using concentrations close to each other in order to trigger non-scaling
conc = np.array([10])*1e-6

pychem_sim_one_conc = aux_create_pychem_sim(def_params, conc, "Monomer", "monomeric")


def test_fit_thermal_unfolding_three_state_global_global_global_scaling_one_conc():

    pychem_sim_one_conc.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    assert pychem_sim_one_conc.params_df is not None

# Testing Monomer_monomeric model

monomer_sim = aux_create_pychem_sim(def_params, concs, "Monomer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_monomer_monomeric():

    # local slopes and baselines
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    monomer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4,1], expected, rtol=0.2)

    # Given T1 and T2

    monomer_sim.fit_thermal_unfolding_three_state_global(t1_init=Tm_VAL_1, t2_init=Tm_VAL_2)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    #Only given T1

    monomer_sim.fit_thermal_unfolding_three_state_global(t1_init=Tm_VAL_1)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # Only given T2

    monomer_sim.fit_thermal_unfolding_three_state_global(t2_init=Tm_VAL_2)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed Tm limits

    monomer_sim.fit_thermal_unfolding_three_state_global(tm_limits=[Tm_VAL_1-12, Tm_VAL_1+20, Tm_VAL_2-12, Tm_VAL_2+20])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed dh limits

    monomer_sim.fit_thermal_unfolding_three_state_global(dh_limits=[10, 500, 10, 500])

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

def test_fit_thermal_unfolding_three_state_global_global_monomer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, SLOPE_N, SLOPE_U, EXPONENT_U]

    monomer_sim.fit_thermal_unfolding_three_state_global()

    monomer_sim.fit_thermal_unfolding_three_state_global_global()
    
    np.testing.assert_allclose(monomer_sim.params_df.iloc[[0, 1, 2, 3, slope_position, slope_position + 1, slope_position + 2], 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_global_monomer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U, EXPONENT_U]

    monomer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:10, 1], expected, rtol=0.2)

# Testing Dimer_monomeric model

dimer_sim = aux_create_pychem_sim(def_params, concs, "Dimer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_dimer_monomeric():
    # local slopes and baselines

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    dimer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # Given T1 and T2

    dimer_sim.fit_thermal_unfolding_three_state_global(t1_init=Tm_VAL_1, t2_init=Tm_VAL_2)

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed Tm limits

    dimer_sim.fit_thermal_unfolding_three_state_global(tm_limits=[Tm_VAL_1 - 12, Tm_VAL_1 + 20, Tm_VAL_2 - 12, Tm_VAL_2 + 20])

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed dh limits

    dimer_sim.fit_thermal_unfolding_three_state_global(dh_limits=[10, 500, 10, 500])

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_dimer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, SLOPE_N, SLOPE_U, EXPONENT_U]

    dimer_sim.fit_thermal_unfolding_three_state_global()

    dimer_sim.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[[0, 1, 2, 3, slope_position, slope_position + 1, slope_position + 2], 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_global_dimer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U, EXPONENT_U]

    dimer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:10, 1], expected, rtol=0.2)


# Testing Trimer_monomeric model

trimer_sim = aux_create_pychem_sim(def_params, concs, "Trimer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_trimer_monomeric():
    # local slopes and baselines

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    trimer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # Given T1 and T2

    trimer_sim.fit_thermal_unfolding_three_state_global(t1_init=Tm_VAL_1, t2_init=Tm_VAL_2)

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed Tm limits

    trimer_sim.fit_thermal_unfolding_three_state_global(
        tm_limits=[Tm_VAL_1 - 12, Tm_VAL_1 + 20, Tm_VAL_2 - 12, Tm_VAL_2 + 20])

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed dh limits

    trimer_sim.fit_thermal_unfolding_three_state_global(dh_limits=[10, 500, 10, 500])

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_trimer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, SLOPE_N, SLOPE_U, EXPONENT_U]

    trimer_sim.fit_thermal_unfolding_three_state_global()

    trimer_sim.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[[0, 1, 2, 3, slope_position, slope_position + 1, slope_position + 2], 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_global_trimer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U, EXPONENT_U]

    trimer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:10, 1], expected, rtol=0.2)


# Testing Tetramer_monomeric model

tetramer_sim = aux_create_pychem_sim(def_params, concs, "Tetramer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_tetramer_monomeric():
    # local slopes and baselines

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    tetramer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # Given T1 and T2

    tetramer_sim.fit_thermal_unfolding_three_state_global(t1_init=Tm_VAL_1, t2_init=Tm_VAL_2)

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed Tm limits

    tetramer_sim.fit_thermal_unfolding_three_state_global(
        tm_limits=[Tm_VAL_1 - 12, Tm_VAL_1 + 20, Tm_VAL_2 - 12, Tm_VAL_2 + 20])

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed dh limits

    tetramer_sim.fit_thermal_unfolding_three_state_global(dh_limits=[10, 500, 10, 500])

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_tetramer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    tetramer_sim.fit_thermal_unfolding_three_state_global()

    tetramer_sim.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_global_tetramer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    tetramer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)


# Testing Dimer_dimeric model

dimer_sim_dimeric = aux_create_pychem_sim(def_params, concs, "Dimer", "dimeric")

def test_fit_thermal_unfolding_three_state_global_dimer_dimeric():
    # local slopes and baselines

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:4, 1], expected, rtol=0.2)

    # Given T1 and T2

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global(t1_init=Tm_VAL_1, t2_init=Tm_VAL_2)

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed Tm limits

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global(tm_limits=[Tm_VAL_1 - 12, Tm_VAL_1 + 20, Tm_VAL_2 - 12, Tm_VAL_2 + 20])

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed dh limits

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global(dh_limits=[10, 500, 10, 500])

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:4, 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_dimer_dimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, SLOPE_N, SLOPE_U, EXPONENT_U]

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global()

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[[0, 1, 2, 3, slope_position, slope_position + 1, slope_position + 2], 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_global_dimer_dimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U, EXPONENT_U]

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:10, 1], expected, rtol=0.2)


# Testing Trimer_trimeric model

trimer_sim_trimeric = aux_create_pychem_sim(def_params, concs, "Trimer", "trimeric")

def test_fit_thermal_unfolding_three_state_global_trimer_trimeric():
    # local slopes and baselines

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(trimer_sim_trimeric.params_df.iloc[:4, 1], expected, rtol=0.2)

    # Given T1 and T2

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global(t1_init=Tm_VAL_1, t2_init=Tm_VAL_2)

    np.testing.assert_allclose(trimer_sim_trimeric.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed Tm limits

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global(
        tm_limits=[Tm_VAL_1 - 12, Tm_VAL_1 + 20, Tm_VAL_2 - 12, Tm_VAL_2 + 20])

    np.testing.assert_allclose(trimer_sim_trimeric.params_df.iloc[:4, 1], expected, rtol=0.2)

    # fixed dh limits

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global(dh_limits=[10, 500, 10, 500])

    np.testing.assert_allclose(trimer_sim_trimeric.params_df.iloc[:4, 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_trimer_trimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, SLOPE_N, SLOPE_U, EXPONENT_U]

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global()

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(trimer_sim_trimeric.params_df.iloc[[0, 1, 2, 3, slope_position, slope_position + 1, slope_position + 2], 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_global_trimer_trimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U, EXPONENT_U]

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(trimer_sim_trimeric.params_df.iloc[:10, 1], expected, rtol=0.2)

