"""
This module contains helper functions to obtain the amount of folded/intermediate/unfolded (etc.) protein
Author: Osvaldo Burastero
"""

__all__ = [
    "fn_two_state_monomer"
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


