"""
Test file for thermal unfolding of oligomeres based difference in concentrations
"""

import numpy as np

import pytest

from pychemelt.thermal_oligomer import ThermalOligomer

from pychemelt.utils.math import linear_baseline, exponential_baseline

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

# Calculate signal range for proper y-axis scaling
temp_range  = np.linspace(TEMP_START, TEMP_STOP, N_TEMPS)
temp_range_K = temp_range + 273.15


sample = ThermalOligomer()

def test_set_model():

    sample.set_model("Monomer")
    assert sample.model == "Monomer"

    sample.set_model("Dimer")
    assert sample.model == "Dimer"

    sample.set_model("dimer")
    assert sample.model == "Dimer"

    pytest.raises(ValueError, sample.set_model, "test_false")

def test_guess_Cp():

    sample.n_residues = 150
    sample.guess_Cp()

    assert sample.Cp0 == pytest.approx(2.0933, rel=1e-2)

    sample.n_residues = 0
    pytest.raises(ValueError, sample.guess_Cp)

    sample.n_residues = 5

    sample.guess_Cp()
    assert sample.Cp0 == pytest.approx(0, rel=1e-2)
    
def test_fit_thermal_unfolding_global_monomer():
    model = "Monomer"
    
    signal_fx = map_two_state_model_to_signal_fx(model)

    sample = ThermalOligomer()
    
    sample.set_model(model)
    
    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))


    sample.signal_dic['Simulated signal'] = signal_list
    sample.temp_dic['Simulated signal'] = [temp_range for _ in range(len(concs))]

    sample.conditions = concs

    sample.global_min_temp = np.min(temp_range)
    sample.global_max_temp = np.max(temp_range)

    sample.set_concentrations()

    sample.set_signal('Simulated signal')

    sample.select_conditions(normalise_to_global_max=True)
    sample.expand_multiple_signal()

    sample.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    sample.n_residues = 150  # only for cp initial guess
    sample.guess_Cp()

    sample.fit_thermal_unfolding_global()


    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(sample.params_df.iloc[:3,1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_dimer():
    model = "Dimer"

    signal_fx = map_two_state_model_to_signal_fx(model)

    sample = ThermalOligomer()

    sample.set_model(model)

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    sample.signal_dic['Simulated signal'] = signal_list
    sample.temp_dic['Simulated signal'] = [temp_range for _ in range(len(concs))]

    sample.set_model(model)
    sample.conditions = concs

    sample.global_min_temp = np.min(temp_range)
    sample.global_max_temp = np.max(temp_range)

    sample.set_concentrations()

    sample.set_signal('Simulated signal')

    sample.select_conditions(normalise_to_global_max=True)
    sample.expand_multiple_signal()

    sample.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    sample.n_residues = 150 # only for cp initial guess
    sample.guess_Cp()

    sample.fit_thermal_unfolding_global()

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(sample.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_trimer():
    model = "Trimer"

    signal_fx = map_two_state_model_to_signal_fx(model)

    sample = ThermalOligomer()

    sample.set_model(model)

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    sample.signal_dic['Simulated signal'] = signal_list
    sample.temp_dic['Simulated signal'] = [temp_range for _ in range(len(concs))]

    sample.set_model(model)
    sample.conditions = concs

    sample.global_min_temp = np.min(temp_range)
    sample.global_max_temp = np.max(temp_range)

    sample.set_concentrations()

    sample.set_signal('Simulated signal')

    sample.select_conditions(normalise_to_global_max=True)
    sample.expand_multiple_signal()

    sample.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    sample.n_residues = 150  # only for cp initial guess
    sample.guess_Cp()

    sample.fit_thermal_unfolding_global()

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(sample.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)


def test_fit_thermal_unfolding_global_tetramer():
    model = "Tetramer"

    signal_fx = map_two_state_model_to_signal_fx(model)

    sample = ThermalOligomer()

    sample.set_model(model)

    signal_list = []
    temp_list = []

    for D in concs:
        y = signal_fx(temp_range_K, D, **def_params)

        # Add gaussian error to signal
        y += rng.normal(0, 0.005, len(y))

        signal_list.append(y)
        temp_list.append(temp_range)

    p0 = [Tm_VAL, DHm_VAL, CP0_VAL] + [0.1] * (2 * len(concs))
    low_bounds = [TEMP_START, TEMP_START, 1] + [1e-5] * (2 * len(concs))
    high_bounds = [TEMP_STOP, 200, 5] + [1e3] * (2 * len(concs))

    sample.signal_dic['Simulated signal'] = signal_list
    sample.temp_dic['Simulated signal'] = [temp_range for _ in range(len(concs))]

    sample.set_model(model)
    sample.conditions = concs

    sample.global_min_temp = np.min(temp_range)
    sample.global_max_temp = np.max(temp_range)

    sample.set_concentrations()

    sample.set_signal('Simulated signal')

    sample.select_conditions(normalise_to_global_max=True)
    sample.expand_multiple_signal()

    sample.estimate_baseline_parameters(
        native_baseline_type='linear',
        unfolded_baseline_type='exponential'
    )

    sample.n_residues = 150  # only for cp initial guess
    sample.guess_Cp()

    sample.fit_thermal_unfolding_global()

    expected = [Tm_VAL, DHm_VAL, CP0_VAL]

    np.testing.assert_allclose(sample.params_df.iloc[:3, 1], expected, rtol=0.1, atol=0)
