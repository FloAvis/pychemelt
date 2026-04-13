"""
Module for thermal unfolding simulation and testing of oligomers with given concentration and model parameters.
CpTh is given in the tests

Classes:
    ThermalOligomer: Represents the system for thermal unfolding simulations and parameter fitting.

Notes
-----
- The tests rely on seeded random number generators for reproducible results.
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
CONCS = np.arange(10, 60, 10) * 1e-6
MAX_POINTS = 400

CP1 = 1.0
CPTH = 1.8

# Model / ground-truth parameters
DHm_VAL_1 = 300
DHm_VAL_2 = 300
Tm_VAL_1 = 70
Tm_VAL_2 = 80

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
    'T1': Tm_VAL_1 + 273.15,
    'T2': Tm_VAL_2 + 273.15,
    'bI': INTERCEPT_I,
    'p1_N': C_N_VAL,
    'p2_N': INTERCEPT_N,
    'p3_N': SLOPE_N,
    'p4_N': 0,
    'p1_U': C_U_VAL,
    'p2_U': INTERCEPT_U,
    'p3_U': SLOPE_U,
    'p4_U': EXPONENT_U,
    'baseline_N_fx': linear_baseline,
    'baseline_U_fx': exponential_baseline,
    "Cp1": CP1,
    'CpTh': CPTH,
}

concs = CONCS


def aux_create_pychem_sim(params, concs, model, intermediate):

    signal_fx = map_three_state_model_to_signal_fx(model + "_" + intermediate + "_intermediate")

    # Calculate signal range for proper y-axis scaling
    temp_range = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
    temp_range_K = temp_range + 273.15

    signal_list = []
    temp_list = []

    # Use a seeded Generator for reproducible noise in tests
    rng = np.random.default_rng(2)

    for C in concs:
        y = signal_fx(temp_range_K, C, **params)

        # Add gaussian error to signal
        # y += rng.normal(0, 0.002*1e-3, len(y)) # Small error (seeded)

        signal_list.append(y)
        temp_list.append(temp_range)

    pychem_sim = ThermalOligomer()

    pychem_sim.signal_dic['Fluo'] = signal_list
    pychem_sim.temp_dic['Fluo'] = [temp_range for _ in range(len(concs))]

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


# Needs Cp value for fitting

def test_fit_thermal_unfolding_three_state_global_failure():
    sample = aux_create_pychem_sim(def_params, concs, "Monomer", "monomeric")

    pytest.raises(ValueError, sample.fit_thermal_unfolding_three_state_global, CpTh=0.0)


# Testing Monomer_monomeric model

monomer_sim = aux_create_pychem_sim(def_params, concs, "Monomer", "monomeric")


def test_fit_thermal_unfolding_three_state_given_CpTh_monomer_monomeric():

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, CP1, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U,
                EXPONENT_U]

    monomer_sim.n_residues = 100
    monomer_sim.guess_Cp()

    monomer_sim.fit_thermal_unfolding_three_state_global(CpTh=CPTH)
    monomer_sim.fit_thermal_unfolding_three_state_global_global()

    monomer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(monomer_sim.params_df.iloc[:11, 1], expected, rtol=0.2, atol=1.5)





# Testing Dimer_monomeric model

dimer_sim = aux_create_pychem_sim(def_params, concs, "Dimer", "monomeric")


def test_fit_thermal_unfolding_three_state_given_CpTh_dimer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, CP1, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U,
                EXPONENT_U]

    dimer_sim.n_residues = 100
    dimer_sim.guess_Cp()

    dimer_sim.fit_thermal_unfolding_three_state_global(CpTh=CPTH)
    dimer_sim.fit_thermal_unfolding_three_state_global_global()

    dimer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(dimer_sim.params_df.iloc[:11, 1], expected, rtol=0.2, atol=1.5)


# Testing Trimer_monomeric model

trimer_sim = aux_create_pychem_sim(def_params, concs, "Trimer", "monomeric")


def test_fit_thermal_unfolding_three_state_given_CpTh_trimer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, CP1, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U,
                EXPONENT_U]

    trimer_sim.n_residues = 100
    trimer_sim.guess_Cp()

    trimer_sim.fit_thermal_unfolding_three_state_global(CpTh=CPTH)
    trimer_sim.fit_thermal_unfolding_three_state_global_global()

    trimer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(trimer_sim.params_df.iloc[:11, 1], expected, rtol=0.2, atol=1.5)


# Testing Tetramer_monomeric model

tetramer_sim = aux_create_pychem_sim(def_params, concs, "Tetramer", "monomeric")


def test_fit_thermal_unfolding_three_state_given_CpTh_tetramer_monomeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, CP1, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U,
                EXPONENT_U]

    tetramer_sim.n_residues = 100
    tetramer_sim.guess_Cp()

    tetramer_sim.fit_thermal_unfolding_three_state_global(CpTh=CPTH)
    tetramer_sim.fit_thermal_unfolding_three_state_global_global()

    tetramer_sim.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(tetramer_sim.params_df.iloc[:11, 1], expected, rtol=0.2, atol=1.5)


# Testing Dimer_dimeric model

dimer_sim_dimeric = aux_create_pychem_sim(def_params, concs, "Dimer", "dimeric")


def test_fit_thermal_unfolding_three_state_given_CpTh_dimer_dimeric():
    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, CP1, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U,
                EXPONENT_U]

    dimer_sim_dimeric.n_residues = 100
    dimer_sim_dimeric.guess_Cp()

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global(CpTh=CPTH)
    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global_global()

    dimer_sim_dimeric.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(dimer_sim_dimeric.params_df.iloc[:11, 1], expected, rtol=0.2, atol=1.5)


# Testing Trimer_trimeric model

trimer_sim_trimeric = aux_create_pychem_sim(def_params, concs, "Trimer", "trimeric")


def test_fit_thermal_unfolding_three_state_given_CpTh_trimer_trimeric():

    expected = [Tm_VAL_1, DHm_VAL_1, Tm_VAL_2, DHm_VAL_2, CP1, INTERCEPT_N, INTERCEPT_U, INTERCEPT_I, SLOPE_N, SLOPE_U,
                EXPONENT_U]

    trimer_sim_trimeric.n_residues = 100
    trimer_sim_trimeric.guess_Cp()

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global(CpTh=CPTH)
    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global_global()

    trimer_sim_trimeric.fit_thermal_unfolding_three_state_global_global_global(model_scale_factor=True)

    np.testing.assert_allclose(trimer_sim_trimeric.params_df.iloc[:11, 1], expected, rtol=0.2, atol=1.5)

