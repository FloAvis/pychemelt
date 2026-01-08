"""
Tests to ensure that the main functionalities of the pychemelt Sample class work as expected.
The order of the tests is important, as some functions depend on the previous ones.
"""
import pandas as pd
import numpy as np
import pytest

from pychemelt import Monomer as Sample

sample = Sample()

def test_load_data():
    sample.read_multiple_files('./test_files/example_data.supr')

    assert len(sample.conditions) == 384
    assert len(sample.labels) == 384

    sample.set_denaturant_concentrations()

    sample.set_signal(sample.signals[100])

    sample.select_conditions([False for _ in range(10)]+[True for _ in range(384-10)])

    # We need to interpolate the signal because it is not evenly sampled in the temperature dimension
    sample.estimate_derivative()

    assert sample.signal_names[0] == sample.signals[100]

    assert len(sample.signal_lst_multiple) == 1
    assert len(sample.signal_lst_multiple[0]) == 384-10

