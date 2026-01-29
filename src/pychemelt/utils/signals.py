"""
This module contains helper functions to obtain the signal, given certain parameters
Author: Osvaldo Burastero
"""

from .rates import (
    eq_constant_termochem,
    eq_constant_thermo
)

from .fractions import *

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

# Oligomeric thermal unfolding signals

def two_state_thermal_unfold_curve(
        T,C,Tm,dHm,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        Cp=0):

    """
    Two-state temperature unfolding (monomer).
    N ⇔ U

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

#Oligomeric thermal unfolding signals with intermediate signals

def unfolding_curve_monomer_monomeric_intermediate(
        T, C, T1, DH1, T2, DH2,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        bI,
        Cp1=0, CpTh=0):
    """
    Three states reversible unfolding N <-> I <-> U
    """

    A = eq_constant_thermo(T, DH1, T1, Cp1)
    B = eq_constant_thermo(T, DH2, T2, CpTh - Cp1)

    den = (1 + A + A * B)

    xN, xI, xU = 1 / den, A / den, A * B / den

    dT = shift_temperature_K(T)

    S_native   = baseline_N_fx(dT,C,p1_N,p2_N,p3_N,p4_N)
    S_unfolded = baseline_U_fx(dT,C,p1_U,p2_U,p3_U,p4_U)

    return xN * S_native + xI * bI + xU * S_unfolded


def unfolding_curve_dimer_monomeric_intermediate(
        T, C, T1, DH1, T2, DH2,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        bI,
        Cp1=0, CpTh=0):
    """
    N2 ⇔ 2Ι ⇔ 2U Three-state unfolding with a monomeric intermediate
    C = concentration in dimer equivalent
    CpTotal = Cp1 + 2*Cp2
    """

    K1 = eq_constant_thermo(T, DH1, T1, Cp1)
    K2 = eq_constant_thermo(T, DH2, T2, (CpTh - Cp1) / 2)

    fi = fi_three_state_dimer_monomeric_intermediate(K1, K2, C)
    fu = fi * K2

    dT = shift_temperature_K(T)

    S_native = baseline_N_fx(dT, C, p1_N, p2_N, p3_N, p4_N)
    S_unfolded = baseline_U_fx(dT, C, p1_U, p2_U, p3_U, p4_U)

    return (1 - fu - fi) * S_native + fi * bI * 2 + fu * S_unfolded * 2


def unfolding_curve_trimer_monomeric_intermediate(
        T, C, T1, DH1, T2, DH2,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        bI,
        Cp1=0, CpTh=0):
    """
    N3 ⇔ 3Ι ⇔ 3U Three-state unfolding with a monomeric intermediate
    C = concentration of the trimer equivalent
    """

    K1 = eq_constant_thermo(T, DH1, T1, Cp1)
    K2 = eq_constant_thermo(T, DH2, T2, (CpTh - Cp1) / 3)  # We should actually find how Cp2 depends on CpTh

    fi = fi_three_state_trimer_monomeric_intermediate(K1, K2, C)
    fu = fi * K2

    dT = shift_temperature_K(T)

    S_native = baseline_N_fx(dT, C, p1_N, p2_N, p3_N, p4_N)
    S_unfolded = baseline_U_fx(dT, C, p1_U, p2_U, p3_U, p4_U)

    return (1 - fu - fi) * S_native + fi * bI * 3 + fu * S_unfolded * 3


def unfolding_curve_tetramer_monomeric_intermediate(
        T, C, T1, DH1, T2, DH2,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        bI,
        Cp1=0, CpTh=0):
    """
    N4 ⇔ 4Ι ⇔ 4U Three-state unfolding with a monomeric intermediate
    C = concentration of the trimer equivalent
    """

    K1 = eq_constant_thermo(T, DH1, T1, Cp1)
    K2 = eq_constant_thermo(T, DH2, T2, (CpTh - Cp1) / 4)

    fi = fi_three_state_tetramer_monomeric_intermediate(K1, K2, C)
    fu = fi * K2

    dT = shift_temperature_K(T)

    S_native = baseline_N_fx(dT, C, p1_N, p2_N, p3_N, p4_N)
    S_unfolded = baseline_U_fx(dT, C, p1_U, p2_U, p3_U, p4_U)

    return (1 - fu - fi) * S_native + fi * bI * 4 + fu * S_unfolded * 4


def unfolding_curve_trimer_trimeric_intermediate(
        T, C, T1, DH1, T2, DH2,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        bI,
        Cp1=0, CpTh=0):
    """
    N3 ⇔ Ι3 ⇔ 3U Three-state unfolding with a trimeric intermediate
    C = concentration of the trimer equivalent
    """

    K1 = eq_constant_thermo(T, DH1, T1, Cp1)
    K2 = eq_constant_thermo(T, DH2, T2, CpTh - Cp1)

    fu = fu_three_state_trimer_trimeric_intermediate(K1, K2, C)
    fi = fi_three_state_trimer_trimeric_intermediate(fu, K2, C)

    dT = shift_temperature_K(T)

    S_native = baseline_N_fx(dT, C, p1_N, p2_N, p3_N, p4_N)
    S_unfolded = baseline_U_fx(dT, C, p1_U, p2_U, p3_U, p4_U)

    return (1 - fu - fi) * S_native + fi * bI + fu * S_unfolded * 3


def unfolding_curve_dimer_dimeric_intermediate(
        T, C, T1, DH1, T2, DH2,
        p1_N, p2_N, p3_N, p4_N,
        p1_U, p2_U, p3_U, p4_U,
        baseline_N_fx,
        baseline_U_fx,
        bI,
        Cp1=0, CpTh=0):
    """
    N2 ⇔ Ι2 ⇔ 2U Three-state unfolding with a monomeric intermediate
    C       = molar concentration in dimer equivalent
    CpTotal = Cp1 + Cp2
    """

    K1 = eq_constant_thermo(T, DH1, T1, Cp1)
    K2 = eq_constant_thermo(T, DH2, T2, CpTh - Cp1)

    fu = fu_three_state_dimer_dimeric_intermediate(K1, K2, C)
    fi = fi_three_state_dimer_dimeric_intermediate(fu, K2, C)

    dT = shift_temperature_K(T)

    S_native = baseline_N_fx(dT, C, p1_N, p2_N, p3_N, p4_N)
    S_unfolded = baseline_U_fx(dT, C, p1_U, p2_U, p3_U, p4_U)

    return (1 - fu - fi) * S_native + fi * bI + fu * S_unfolded * 2


def map_three_state_model_to_signal_fx(model):
    signal_fx_map = {
        'Monomer_monomeric_intermediate': unfolding_curve_monomer_monomeric_intermediate,
        'Dimer_monomeric_intermediate': unfolding_curve_dimer_monomeric_intermediate,
        'Dimer_dimeric_intermediate': unfolding_curve_dimer_dimeric_intermediate,
        'Trimer_monomeric_intermediate': unfolding_curve_trimer_monomeric_intermediate,
        'Trimer_trimeric_intermediate': unfolding_curve_trimer_trimeric_intermediate,
        'Tetramer_monomeric_intermediate': unfolding_curve_tetramer_monomeric_intermediate
    }

    return signal_fx_map.get(model)
