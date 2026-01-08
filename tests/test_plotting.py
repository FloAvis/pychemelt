import plotly.graph_objs as go

from pychemelt.utils.plotting import plot_unfolding
from pychemelt import Monomer as Sample

def test_plot_unfolding():

    sample = Sample()

    sample.read_multiple_files('./test_files/nDSFdemoFile.xlsx')
    sample.set_denaturant_concentrations()
    sample.set_signal(['350nm', '330nm'])

    sample.select_conditions([True for _ in range(8)] + [False for _ in range(48 - 8)])

    fig = plot_unfolding(sample)

    assert fig is not None
    assert isinstance(fig, go.Figure)

    sample.expand_multiple_signal()
    sample.estimate_baseline_parameters(
        native_baseline_type='quadratic',
        unfolded_baseline_type='quadratic'
    )
    sample.estimate_derivative()
    sample.guess_Tm()
    sample.n_residues = 130
    sample.guess_Cp()
    sample.set_signal_id()
    sample.fit_thermal_unfolding_local()
    sample.fit_thermal_unfolding_global()

    fig = plot_unfolding(sample)

    assert fig is not None
    assert isinstance(fig, go.Figure)

