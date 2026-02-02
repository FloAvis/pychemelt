"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import numpy as np
import pytest

from pychemelt.thermal_oligomer import ThermalOligomer
from pychemelt.utils.signals import (
    map_two_state_model_to_signal_fx
)
from pychemelt.utils.math import exponential_baseline, constant_baseline, linear_baseline

# Centralized test constants
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 90.0
N_TEMPS = 100
CONCS = np.arange(1,6.0,0.5)
MAX_POINTS = 400

# Model / ground-truth parameters
DHm_VAL = 120
Tm_VAL = 65
CP0_VAL = 1.8

INTERCEPT_N = 100
PRE_EXP_N = 1
C_N_VAL = 0
ALPHA_N_VAL = 0.1


INTERCEPT_U = 110
PRE_EXP_U = 0
C_U_VAL = 0
ALPHA_U_VAL = 0


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
    'baseline_U_fx':constant_baseline

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
        native_baseline_type='exponential',
        unfolded_baseline_type='constant'
    )

    pychem_sim.n_residues = 130 # only for cp initial guess
    pychem_sim.guess_Cp()

    return pychem_sim

# Testing Monomer model

monomer_sim = aux_create_pychem_sim(def_params, concs, "Monomer")

def test_fit_thermal_unfolding_global_monomer_exponential_baseline():

    monomer_sim.max_points = MAX_POINTS

    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    monomer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3,1], expected, rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_monomer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    monomer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_monomer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    monomer_sim.fit_thermal_unfolding_global_global_global()

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)

# Testing Dimer model

dimer_sim = aux_create_pychem_sim(def_params, concs, "Dimer")

def test_fit_thermal_unfolding_global_dimer_exponential_baseline():
    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    dimer_sim.max_points = MAX_POINTS

    dimer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_global_dimer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]


    dimer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_dimer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    dimer_sim.fit_thermal_unfolding_global_global_global()

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)


# Testing Trimer model

trimer_sim = aux_create_pychem_sim(def_params, concs, "Trimer")

def test_fit_thermal_unfolding_global_trimer_exponential_baseline():
    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    trimer_sim.max_points = MAX_POINTS

    trimer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_global_trimer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]


    trimer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_trimer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    trimer_sim.fit_thermal_unfolding_global_global_global()

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)


# Testing Tetramer model

tetramer_sim = aux_create_pychem_sim(def_params, concs, "Tetramer")

def test_fit_thermal_unfolding_global_tetramer_exponential_baseline():
    # local slopes and baselines
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    tetramer_sim.max_points = MAX_POINTS

    tetramer_sim.fit_thermal_unfolding_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_global_tetramer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    tetramer_sim.fit_thermal_unfolding_global_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)

def test_fit_thermal_unfolding_global_global_global_tetramer_exponential_baseline():
    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    tetramer_sim.fit_thermal_unfolding_global_global_global()

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:3, 1], expected,
                               rtol=0.1, atol=0)
