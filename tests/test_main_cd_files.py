import numpy as np
import pytest

from pychemelt import Monomer as Sample

def test_load_data_jasco():

    sample = Sample()
    sample.read_multiple_files('./test_files/jasco_thermal_ramp_example.csv')

    assert len(sample.conditions) == 1

    sample.set_denaturant_concentrations(5)

    sample.set_signal(sample.signals[0])

    sample.select_conditions(True)

    assert sample.signal_lst_multiple is not None

    assert len(sample.signals) > 10

def test_load_data_jasco_2():

    sample = Sample()
    sample.read_multiple_files('./test_files/jasco_thermal_ramp_one_wavelength.txt')

    assert len(sample.conditions) == 1

    sample.set_denaturant_concentrations()

    sample.set_signal(sample.signals[0])

    sample.select_conditions()

    assert sample.signal_lst_multiple is not None

    assert len(sample.signals) == 1

def test_load_data_chirascan():

    sample = Sample()
    sample.read_multiple_files('./test_files/chirascan_thermal_ramp.csv')

    assert len(sample.conditions) == 1

    sample.set_denaturant_concentrations()

    sample.set_signal(sample.signals[0])

    sample.select_conditions()

    assert sample.signal_lst_multiple is not None

    assert len(sample.signals) > 10