"""
PyChemelt package for the analysis of chemical and thermal denaturation data
"""

from .main import Sample
from .monomer import Monomer
from .thermal_oligomer import ThermalOligomer

from .utils.math import (
    get_rss,
    shift_temperature,
    constant_baseline,
    linear_baseline,
    quadratic_baseline,
    exponential_baseline
)

from .utils.signals import (
    signal_two_state_tc_unfolding,
    signal_two_state_t_unfolding,
    map_two_state_model_to_signal_fx,
    map_three_state_model_to_signal_fx
)

from .utils.plotting import (
    plot_unfolding
)


from .utils.fitting import (
    fit_line_robust,
    fit_quadratic_robust,
    fit_exponential_robust
)