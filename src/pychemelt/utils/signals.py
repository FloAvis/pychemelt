"""
This module contains helper functions to obtain the signal, given certain parameters
Author: Osvaldo Burastero
"""

from .rates import (
    eq_constant_termochem,
    eq_constant_thermo
)

from .fractions import (
    fn_two_state_monomer,
    fu_two_state_dimer,
    fu_two_state_trimer,
    fu_two_state_tetramer,
)

from .math import shift_temperature_K



def signal_two_state_tc_unfolding(
        T,D,DHm,Tm,Cp0,m0,m1,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        extra_arg=None):

    """
    Ref: Louise Hamborg et al., 2020. Global analysis of protein stability by temperature and chemical
    denaturation

    Parameters
    ----------
    T : array-like
        Temperature in Kelvin units
    D : array-like
        Denaturant agent concentration
    DHm : float
        Variation of enthalpy between the two considered states at Tm
    Tm : float
        Temperature at which the equilibrium constant equals one, in Kelvin units
    Cp0 : float
        Variation of calorific capacity between the two states
    m0 : float
        m-value at the reference temperature (Tref)
    m1 : float
        Variation of m-value with temperature
    p1_N, p2_N, p3_N, p4_N : float
        parameters describing the native-state baseline
    p1_U, p2_U, p3_U, p4_U : float
        parameters describing the unfolded-state baseline
    baseline_N_fx : function
        for the native-state baseline
    baseline_U_fx : function
        for the unfolded-state baseline
    extra_arg : None, optional
        Not used but present for API compatibility with oligomeric models

    Returns
    -------
    numpy.ndarray
        Signal at the given temperatures and denaturant agent concentration, given the parameters
    """

    K   = eq_constant_termochem(T,D,DHm,Tm,Cp0,m0,m1)
    fn  = fn_two_state_monomer(K)
    fu  = 1 - fn
    dT   = shift_temperature_K(T)

    # Baseline signals (with quadratic dependence on temperature)
    S_native   = baseline_N_fx(dT,D,p1_N, p2_N, p3_N, p4_N)
    S_unfolded = baseline_U_fx(dT,D,p1_U, p2_U, p3_U, p4_U)

    return  fn*(S_native) + fu*(S_unfolded)


def signal_two_state_t_unfolding(
        T,Tm,dHm,
        p1_N, p2_N, p3_N,
        p1_U, p2_U, p3_U,
        baseline_N_fx,
        baseline_U_fx,
        Cp=0,
        extra_arg=None):

    """
    Two-state temperature unfolding (monomer).

    Parameters
    ----------
    T : array-like
        Temperature
    Tm : float
        Temperature at which the equilibrium constant equals one
    dHm : float
        Variation of enthalpy between the two considered states at Tm
    p1_N, p2_N, p3_N : float
        baseline parameters for the native-state baseline
    p1_U, p2_U, p3_U : float
        baseline parameters for the unfolded-state baseline
    baseline_N_fx : callable
        function to calculate the baseline for the native state
    baseline_U_fx : callable
        function to calculate the baseline for the unfolded state
    Cp : float, optional
        Variation of heat capacity between the two states (default: 0)
    extra_arg : None, optional
        Not used but present for compatibility

    Returns
    -------
    numpy.ndarray
        Signal at the given temperatures, given the parameters
    """

    K   = eq_constant_thermo(T,dHm,Tm,Cp)
    fn  = fn_two_state_monomer(K)
    fu  = 1 - fn

    dT  = shift_temperature_K(T)

    S_native   = baseline_N_fx(dT,0,0,p1_N,p2_N,p3_N) # No denaturant dependence, that's why d=0 and den_slope = 0
    S_unfolded = baseline_U_fx(dT,0,0,p1_U,p2_U,p3_U) # No denaturant dependence, that's why d=0 and den_slope = 0

    return fn*(S_native) + fu*(S_unfolded)

def two_state_thermal_unfold_curve(
        T,C,Tm,dHm,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        Cp=0):

    """
    Two-state temperature unfolding (monomer).

    Parameters
    ----------
    T : array-like
        Temperature
    Tm : float
        Temperature at which the equilibrium constant equals one
    dHm : float
        Variation of enthalpy between the two considered states at Tm
    p1_N, p2_N, p3_N : float
        baseline parameters for the native-state baseline
    p1_U, p2_U, p3_U : float
        baseline parameters for the unfolded-state baseline
    baseline_N_fx : callable
        function to calculate the baseline for the native state
    baseline_U_fx : callable
        function to calculate the baseline for the unfolded state
    Cp : float, optional
        Variation of heat capacity between the two states (default: 0)
    extra_arg : None, optional
        Not used but present for compatibility

    Returns
    -------
    numpy.ndarray
        Signal at the given temperatures, given the parameters
    """

    K   = eq_constant_thermo(T,dHm,Tm,Cp)
    fn  = fn_two_state_monomer(K)
    fu  = 1 - fn

    dT  = shift_temperature_K(T)

    S_native   = baseline_N_fx(dT,C,p1_N,p2_N,p3_N,p4_N)
    S_unfolded = baseline_U_fx(dT,C,p1_U,p2_U,p3_U,p4_U)

    return fn*(S_native) + fu*(S_unfolded)

# two_state_thermal_unfold_curve == signal_two_state_t_unfolding
'''
N ⇔ U
'''

def two_state_thermal_unfold_curve_dimer(
        T,C,Tm,dHm,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        Cp=0,
        extra_arg=None):
    
    """
    N2 ⇔ 2U   C is the total concentration (M) of the protein in dimer equivalent.
    """

    K  = eq_constant_thermo(T,dHm,Tm,Cp)
    fu = fu_two_state_dimer(K,C)
    fn = 1 - fu

    dT  = shift_temperature_K(T)

    S_native   = baseline_N_fx(dT,C,p1_N,p2_N,p3_N,p4_N)
    S_unfolded = baseline_U_fx(dT,C,p1_U,p2_U,p3_U,p4_U)

    return fn*(S_native) + fu*(S_unfolded)*2

def two_state_thermal_unfold_curve_trimer(
        T,C,Tm,dHm,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        Cp=0,
        extra_arg=None):

    """
    N3 ⇔ 3U   C is the total concentration (M) of the protein in trimer equivalent.
    """

    K  = eq_constant_thermo(T,dHm,Tm,Cp)
    fu = fu_two_state_trimer(K,C)
    fn = 1 - fu

    dT  = shift_temperature_K(T)

    S_native   = baseline_N_fx(dT,C,p1_N,p2_N,p3_N,p4_N)
    S_unfolded = baseline_U_fx(dT,C,p1_U,p2_U,p3_U,p4_U)


    return fn*(S_native) + fu*(S_unfolded)*3

def two_state_thermal_unfold_curve_tetramer(
        T,C,Tm,dHm,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        Cp=0,
        extra_arg=None):

    """
    N4 ⇔ 4U   C is the total concentration (M) of the protein in tetramer equivalent.
    """

    K  = eq_constant_thermo(T,dHm,Tm,Cp)
    fu = fu_two_state_tetramer(K,C)
    fn = 1 - fu

    dT  = shift_temperature_K(T)

    S_native   = baseline_N_fx(dT,C,p1_N,p2_N,p3_N,p4_N)
    S_unfolded = baseline_U_fx(dT,C,p1_U,p2_U,p3_U,p4_U)


    return fn*(S_native) + fu*(S_unfolded)*4

def map_two_state_model_to_signal_fx(model):

    signal_fx_map = {
    'Monomer':  two_state_thermal_unfold_curve,
    'Dimer':    two_state_thermal_unfold_curve_dimer,
    'Trimer':   two_state_thermal_unfold_curve_trimer,
    'Tetramer': two_state_thermal_unfold_curve_tetramer
    }

    return signal_fx_map.get(model)
