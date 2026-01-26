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