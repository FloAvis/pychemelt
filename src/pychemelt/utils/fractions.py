"""
This module contains helper functions to obtain the amount of folded/intermediate/unfolded (etc.) protein
Author: Osvaldo Burastero
"""

import numpy as np

from .math import (
    solve_one_root_quadratic,
    solve_one_root_depressed_cubic,
)

from .rates import (
    eq_constant_thermo,
    eq_constant_termochem,
)

__all__ = [
    "fn_two_state_monomer",
    "fu_two_state_dimer",
    "fu_two_state_trimer",
    "fu_two_state_tetramere",
    "two_state_rev_unfolding_fractions",
    "two_state_dimer_unfolding_fractions",
    "two_state_trimer_unfolding_fractions",
    "two_state_tetramer_unfolding_fractions",
    "map_two_state_model_to_fractions_fx",
]

def fn_two_state_monomer(K):
    """
    Given the equilibrium constant K of N <-> U, return the fraction of folded protein.

    Parameters
    ----------
    K : float
        Equilibrium constant of the reaction N <-> U

    Returns
    -------
    float
        Fraction of folded protein
    """
    return (1/(1 + K))

def fu_two_state_dimer(K,C):
    '''
    Given the equilibrium constant K, of N2 <-> 2U, 
    and the concentration of dimer equivalent C, return the fraction of unfolded protein
    '''
    return solve_one_root_quadratic(4*C, K, -K)

def fu_two_state_trimer(K,C):
    '''
    Given the equilibrium constant K, of N3 <-> 3U, 
    and the concentration of trimer equivalent C, return the fraction of unfolded protein
    '''
    p = K/27/np.square(C)
    return solve_one_root_depressed_cubic(p,-p)

def fu_two_state_tetramer(K,C):
    '''
    Given the equilibrium constant K, of N4 <-> 4U, 
    and the concentration of tetramer equivalent C, return the fraction of folded protein
    '''

    A = 1
    D = K/256/np.power(C,3)
    E = -D

    b = D/A
    c = E/A

    P = -c
    Q = -np.square(b)/8

    R = -Q/2 + np.sqrt(np.square(Q)/4+P**3/27)

    U = np.cbrt(R)
    y = U-P/(3*U)
    W = np.sqrt(2*y)

    x4 = 0.5*(-W+np.sqrt(-(2*y-2*b/W)))  

    x4_sel = np.logical_and(np.greater(x4,0),np.less(x4,1.01))

    fu = x4_sel*np.nan_to_num(x4,nan=0.0)

    return fu


def two_state_rev_unfolding_fractions(T,DHm,Tm,extra_arg,Cp=0):

    K  = eq_constant_thermo(T,DHm,Tm,Cp) 
    fn = fn_two_state_monomer(K)

    return {'Native': fn, 'Unfolded': (1-fn)}

def two_state_dimer_unfolding_fractions(T,DH1,T1,C,Cp=0):

    """
    N2 ⇔ 2U   where C is the total concentration (M) of the protein in dimer equivalent.
    """

    K  = eq_constant_thermo(T,DH1,T1,Cp)
    fu = fu_two_state_dimer(K,C)

    return {'Native dimer':(1-fu), 'Unfolded monomer':fu}

def two_state_trimer_unfolding_fractions(T,DH1,T1,C,Cp=0):

    """
    N3 ⇔ 3U   
    C is the total concentration (M) of the protein in trimer equivalent.
    """

    K  = eq_constant_thermo(T,DH1,T1,Cp)
    fu = fu_two_state_trimer(K,C)

    return {'Native trimer':(1-fu), 'Unfolded monomer':fu}

def two_state_tetramer_unfolding_fractions(T,DH1,T1,C,Cp=0):

    """
    N4 ⇔ 4U   C is the total concentration (M) of the protein in tetramer equivalent.
    """

    K  = eq_constant_thermo(T,DH1,T1,Cp)
    fu = fu_two_state_tetramer(K,C)

    return {'Native tetramer':(1-fu), 'Unfolded monomer':fu}


def map_two_state_model_to_fractions_fx(model):

    fractions_fx_map = {
    'Monomer':  two_state_rev_unfolding_fractions,
    'Dimer':    two_state_dimer_unfolding_fractions,
    'Trimer':   two_state_trimer_unfolding_fractions,
    'Tetramer': two_state_tetramer_unfolding_fractions
    }

    return fractions_fx_map.get(model)
