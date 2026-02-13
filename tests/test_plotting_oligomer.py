import pychemelt as pychem
import numpy as np
from pychemelt import ThermalOligomer

from pychemelt.utils.math import (
    constant_baseline,
)

import plotly.graph_objs as go
from pychemelt.utils.signals import map_two_state_model_to_signal_fx


# Centralized test constants
RNG_SEED = 2
TEMP_START = 20.0
TEMP_STOP = 90.0
N_TEMPS = 100


def_params = {

     # Thermodynamic parameters

    'dHm': 300, # Enthalpy of unfolding
    'Tm': 273.15 + 70, # Temperature of unfolding
    'Cp': 1.0, # Heat capacity of unfolding

    # Native baseline parameters
    'p1_N': 0,  # dependence on denaturant concentration
    'p2_N': 50,  # intercept
    'p3_N': 0,  # slope term
    'p4_N': 0,  # ignored

    # Unfolded baseline parameters

    'p1_U': 0,  # dependence on denaturant concentration
    'p2_U': 95,  # intercept
    'p3_U': 0,  # pre-exponential factor
    'p4_U': 0,  # exponential coefficient

    'baseline_N_fx':constant_baseline,
    'baseline_U_fx':constant_baseline
}


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
def test_plot_unfolding_concentrations():

    # Test mM concentrations
    concs = np.arange(10, 100, 10)*1e-3

    pychem_sim = aux_create_pychem_sim(def_params,concs, model="Monomer")

    fig = pychem.plot_unfolding(pychem_sim)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    # Test μM concentrations
    concs = np.arange(10, 100, 10)*1e-6

    pychem_sim = aux_create_pychem_sim(def_params,concs, model="Monomer")

    fig = pychem.plot_unfolding(pychem_sim)

    assert fig is not None
    assert isinstance(fig, go.Figure)