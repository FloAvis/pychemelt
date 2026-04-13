"""
Tests to ensure that the main functionalities of the pychemelt ThermalOligomer class work as expected.
The order of the tests is important, as some functions depend on the previous ones.

This file tests the fitting of the thermal unfolding model to exponential native and constant unfolded baseline data.
"""
import numpy as np
import pytest

from pychemelt.thermal_oligomer import ThermalOligomer
from pychemelt.utils.signals import (
    map_three_state_model_to_signal_fx
)
from pychemelt.utils.math import exponential_baseline, constant_baseline, linear_baseline

# Centralized test constants
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 90.0
N_TEMPS = 150
CONCS = np.arange(10, 60, 10)*1e-6
MAX_POINTS = 400

# Model / ground-truth parameters
DHm_VAL_1 = 250
DHm_VAL_2 = 250
Tm_VAL_1 = 60
Tm_VAL_2 = 70

INTERCEPT_I = 107

INTERCEPT_N = 80
PRE_EXP_N = 1
C_N_VAL = 0
ALPHA_N_VAL = 0.2


INTERCEPT_U = 110
PRE_EXP_U = 0
C_U_VAL = 0
ALPHA_U_VAL = 0


rng = np.random.default_rng(RNG_SEED)

def_params = {
    'DH1': DHm_VAL_1,
    'DH2': DHm_VAL_2,
    'T1': Tm_VAL_1+273.15,
    'T2': Tm_VAL_2+273.15,
    'bI': INTERCEPT_I,
    'p1_N': C_N_VAL,
    'p2_N': INTERCEPT_N,
    'p3_N': PRE_EXP_N,
    'p4_N': ALPHA_N_VAL,
    'p1_U': C_U_VAL,
    'p2_U': INTERCEPT_U,
    'p3_U': PRE_EXP_U,
    'p4_U': ALPHA_U_VAL,
    'baseline_N_fx':exponential_baseline,
    'baseline_U_fx':constant_baseline,
    "Cp1":0.5,
    'CpTh':1.0,
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
        y += rng.normal(0, 0.002*1e-3, len(y)) # Small error (seeded)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = ThermalOligomer()

    pychem_sim.set_model(model, intermediate)

    pychem_sim.signal_dic['Fluo'] = signal_list
    pychem_sim.temp_dic['Fluo']   = [temp_range for _ in range(len(concs))]

    pychem_sim.conditions = concs

    pychem_sim.global_min_temp = np.min(temp_range)
    pychem_sim.global_max_temp = np.max(temp_range)

    pychem_sim.set_concentrations()

    pychem_sim.set_signal(['Fluo'])

    pychem_sim.select_conditions()
    pychem_sim.expand_multiple_signal()

    pychem_sim.estimate_baseline_parameters(
        native_baseline_type='exponential',
        unfolded_baseline_type='constant'
    )


    return pychem_sim

# Testing Monomer model

monomer_sim = aux_create_pychem_sim(def_params, concs, "Monomer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_monomer_exponential_baseline_monomeric():

    monomer_sim.max_points = MAX_POINTS

    # local slopes and baselines
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    monomer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4,1], expected, rtol=0.1)

def test_fit_thermal_unfolding_three_state_global_global_monomer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    monomer_sim.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.1)

def test_fit_thermal_unfolding_three_state_global_global_global_monomer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    monomer_sim.fit_thermal_unfolding_three_state_global_global_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.1)

# Testing Dimer model

dimer_sim = aux_create_pychem_sim(def_params, concs, "Dimer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_dimer_exponential_baseline_monomeric():
    # local slopes and baselines
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    dimer_sim.max_points = MAX_POINTS

    dimer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:4, 1], expected, rtol=0.1)


def test_fit_thermal_unfolding_three_state_global_global_dimer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]


    dimer_sim.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.1)

def test_fit_thermal_unfolding_three_state_global_global_global_dimer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    dimer_sim.fit_thermal_unfolding_three_state_global_global_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.1)


# Testing Trimer model

trimer_sim = aux_create_pychem_sim(def_params, concs, "Trimer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_trimer_exponential_baseline_monomeric():
    # local slopes and baselines
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    trimer_sim.max_points = MAX_POINTS

    trimer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:4, 1], expected, rtol=0.1)


def test_fit_thermal_unfolding_three_state_global_global_trimer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]


    trimer_sim.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.1)

def test_fit_thermal_unfolding_three_state_global_global_global_trimer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    trimer_sim.fit_thermal_unfolding_three_state_global_global_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.1)


# Testing Tetramer model

tetramer_sim = aux_create_pychem_sim(def_params, concs, "Tetramer", "monomeric")

def test_fit_thermal_unfolding_three_state_global_tetramer_exponential_baseline_monomeric():
    # local slopes and baselines
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    tetramer_sim.max_points = MAX_POINTS

    tetramer_sim.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected, rtol=0.2)


def test_fit_thermal_unfolding_three_state_global_global_tetramer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    tetramer_sim.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.2)

def test_fit_thermal_unfolding_three_state_global_global_global_tetramer_exponential_baseline_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    tetramer_sim.fit_thermal_unfolding_three_state_global_global_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:4, 1], expected,
                               rtol=0.2)

# Testing Dimer model

dimer_sim_dimeric = aux_create_pychem_sim(def_params, concs, "Dimer", "dimeric")

def test_fit_thermal_unfolding_three_state_global_dimer_exponential_baseline_dimeric():
    # local slopes and baselines
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    dimer_sim_dimeric.max_points = MAX_POINTS

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:4, 1], expected, rtol=0.1)


def test_fit_thermal_unfolding_three_state_global_global_dimer_exponential_baseline_dimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]


    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:4, 1], expected,
                               rtol=0.1)

def test_fit_thermal_unfolding_three_state_global_global_global_dimer_exponential_baseline_dimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global_global_global()

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:4, 1], expected,
                               rtol=0.1)


# Testing Trimer model

trimer_sim_trimeric = aux_create_pychem_sim(def_params, concs, "Trimer", "trimeric")

def test_fit_thermal_unfolding_three_state_global_trimer_exponential_baseline_trimeric():
    # local slopes and baselines
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    trimer_sim_trimeric .max_points = MAX_POINTS

    trimer_sim_trimeric .fit_thermal_unfolding_three_state_global()

    np.testing.assert_allclose(trimer_sim_trimeric .params_df.iloc[:4, 1], expected, rtol=0.3)


def test_fit_thermal_unfolding_three_state_global_global_trimer_exponential_baseline_trimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]


    trimer_sim_trimeric .fit_thermal_unfolding_three_state_global_global()

    np.testing.assert_allclose(trimer_sim_trimeric .params_df.iloc[:4, 1], expected,
                               rtol=0.3)

def test_fit_thermal_unfolding_three_state_global_global_global_trimer_exponential_baseline_trimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2]

    trimer_sim_trimeric .fit_thermal_unfolding_three_state_global_global_global()

    np.testing.assert_allclose(trimer_sim_trimeric .params_df.iloc[:4, 1], expected,
                               rtol=0.3)

