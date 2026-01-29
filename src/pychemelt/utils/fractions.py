"""
This module contains helper functions to obtain the amount of folded/intermediate/unfolded (etc.) protein
Author: Osvaldo Burastero
"""

import numpy as np

from .math import (
    solve_one_root_quadratic,
    solve_one_root_depressed_cubic,
)

__all__ = [
    "fn_two_state_monomer",
    "fu_two_state_dimer",
    "fu_two_state_trimer",
    "fu_two_state_tetramer",
    "fi_three_state_tetramer_monomeric_intermediate",
    "fi_three_state_dimer_monomeric_intermediate",
    "fu_three_state_dimer_dimeric_intermediate",
    "fi_three_state_dimer_dimeric_intermediate",
    "fi_three_state_trimer_monomeric_intermediate",
    "fu_three_state_trimer_trimeric_intermediate",
    "fi_three_state_trimer_trimeric_intermediate",
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

def fi_three_state_tetramer_monomeric_intermediate(K1,K2,Ct):
    '''
    Given the equilibrium constant K1, of N4 <-> 4I, K2, of I <-> U,
    and the concentration of tetramer equivalent Ct, return the fraction of intermediate
    '''
    Pt = Ct*4

    A = 4*(Pt**3)/K1
    D = 1+K2
    E = -1

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

    fi = x4_sel*np.nan_to_num(x4,nan=0.0)

    return fi

def fi_three_state_dimer_monomeric_intermediate(K1,K2,C):
    '''
    Given the equilibrium constant K1, of N2 <-> 2I, K2, of 2I <-> 2U
    and the concentration of dimer equivalent C, return the fraction of intermediate
    '''
    return solve_one_root_quadratic(4*C,K1*(1+K2),-K1)

def fu_three_state_dimer_dimeric_intermediate(K1,K2,C):
    '''
    Given the equilibrium constant K1, of N2 <-> I2, K2, of I2 <-> 2U
    and the concentration of dimer equivalent C, return the fraction of unfolded protein
    '''
    return solve_one_root_quadratic(4*C*(1+K1), K1*K2, -K1*K2)

def fi_three_state_dimer_dimeric_intermediate(fu,K2,C):
    '''
    Given the fraction of unfolded protein fu, the equilibrium constant K2, of I2 <-> 2U,
    and the concentration of dimer equivalent C, return the fraction of intermediate
    '''
    return 4*np.square(fu)*C/K2

def fi_three_state_trimer_monomeric_intermediate(K1,K2,C):
    '''
    Given the equilibrium constant K1, of N3 <-> 3I, K2, of 3I <-> 3U
    and the concentration of trimer equivalent C, return the fraction of unfolded protein
    '''
    p = K1*(1+K2)/27/np.square(C)
    q = -K1/27/np.square(C)

    return solve_one_root_depressed_cubic(p,q)

def fu_three_state_trimer_trimeric_intermediate(K1,K2,C):
    '''
    Given the equilibrium constant K1, of N3 <-> I3, K2, of I3 <-> 3U
    and the concentration of trimer equivalent C, return the fraction of unfolded protein
    '''
    p = K1*K2 / (27*np.square(C)*(1+K1))
    q = -p

    return solve_one_root_depressed_cubic(p,q)

def fi_three_state_trimer_trimeric_intermediate(fu,K2,C):
    '''
    Given the fraction of unfolded protein fu, the equilibrium constant K2, of I3 <-> 3U,
    and the concentration of trimer equivalent C, return the fraction of intermediate
    '''
    return 27*np.square(C)*(fu**3) / K2