import numpy as np
import pytest
from time import time
from pychemelt import Monomer as Sample

def test_fit_demo():

    # Start time counter
    start = time()

    sample = Sample()
    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')
    sample.set_denaturant_concentrations()

    sample.set_signal(['350nm'])

    conditions = [True for _ in range(6)] + [False] + [True for _ in range(14)] + [False for _ in range(27)]

    sample.select_conditions(conditions,normalise_to_global_max=True)

    sample.set_temperature_range(5, 100)

    sample.expand_multiple_signal()

    sample.estimate_derivative()

    sample.n_residues = 130
    sample.max_points = 100
    sample.pre_fit = False
    sample.guess_Tm()
    sample.guess_initial_parameters('linear','linear',10,10)
    sample.estimate_baseline_parameters('linear','exponential')
    sample.fit_thermal_unfolding_global()
    sample.fit_thermal_unfolding_global_global()
    sample.fit_thermal_unfolding_global_global_global()

    end = time()

    # assert time is less than 100 seconds
    assert end - start < 100