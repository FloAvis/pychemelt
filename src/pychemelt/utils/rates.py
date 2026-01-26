"""
This module contains helper functions to obtain equilibrium constants
Author: Osvaldo Burastero

Useful references for unfolding models:
    - Rumfeldt, Jessica AO, et al. "Conformational stability and folding mechanisms of dimeric proteins." Progress in biophysics and molecular biology 98.1 (2008): 61-84.
    - Bedouelle, Hugues. "Principles and equations for measuring and interpreting protein stability: From monomer to tetramer." Biochimie 121 (2016): 29-37.
    - Mazurenko, Stanislav, et al. "Exploration of protein unfolding by modelling calorimetry data from reheating." Scientific reports 7.1 (2017): 16321.

All thermodynamic parameters are used in kcal mol units

Unfolding functions for monomers have an argument called 'extra_arg' that is not used.
This is because unfolding functions for oligomers require the protein concentration in that position

"""

import numpy as np

from .math import shift_temperature_K

from .constants import R_gas

def eq_constant_thermo(T,DH1,T1,Cp):

    """
    T1 is the temperature at which ΔG(T) = 0
    ΔH1, the variation of enthalpy between the two considered states at T1
    Cp the variation of calorific capacity between the two states

    Parameters
    ----------
    T : array-like
        Temperature (Kelvin)
    DH1 : float
        Variation of enthalpy between the two considered states at T1 (kcal/mol)
    T1 : float
        Temperature at which the equilibrium constant equals one (Kelvin)
    Cp : float
        Variation of heat capacity between the two states (kcal/mol/K)

    Returns
    -------
    numpy.ndarray
        Equilibrium constant at the given temperature
    """

    DG = DH1*(1 - T/T1) - Cp*(T1 - T + T*np.log(T/T1))
    K  = np.exp(-DG / (R_gas * T))

    return K


def eq_constant_termochem(T,D,DHm,Tm,Cp0,m0,m1):

    """
    Ref: Louise Hamborg et al., 2020. Global analysis of protein stability by temperature and chemical
    denaturation

    Parameters
    ----------
    T : array-like
        Temperature (Kelvin only!)
    D : float
        Denaturant concentration (M)
    DHm : float
        Enthalpy change at Tm (kcal/mol)
    Tm : float
        Melting temperature where ΔG = 0 (Kelvin only!)
    Cp0 : float
        Heat capacity change (kcal/mol/K)
    m0 : float
        m-value at the reference temperature
    m1 : float
        Temperature dependence of the m-value

    Returns
    -------
    numpy.ndarray
        Equilibrium constant at a certain temperature and denaturant agent concentration
    """

    DT  = shift_temperature_K(T)

    DG   = DHm*(1 - T/Tm) + Cp0*(T - Tm - T*np.log(T/Tm)) - D*(m0 + m1*DT)

    DG_RT = -DG / (R_gas * T)

    K     = np.exp(DG_RT)

    return K