"""
This module contains helper functions for mathematical operations
Author: Osvaldo Burastero
"""

import numpy as np
import pandas as pd

from collections        import Counter
from scipy              import stats

from ..utils.constants  import Tref_cst
from scipy.signal       import savgol_filter

temperature_to_kelvin  = lambda T: T + 273.15 if np.max(T) < 270 else T
temperature_to_celsius = lambda T: T - 273.15 if np.max(T) > 270 else T
shift_temperature      = lambda T: temperature_to_kelvin(T) - Tref_cst


def is_evenly_spaced(x, tol = 1e-4):
    """
    Check if x is evenly spaced within a given tolerance.

    Parameters
    ----------
    x : array-like
        x data
    tol : float, optional
        Tolerance for considering spacing equal (default: 1e-4)

    Returns
    -------
    bool
        True if x is evenly spaced, False otherwise
    """

    diffs = np.diff(x)

    return np.all(np.abs(diffs - diffs[0]) < tol)


def first_derivative_savgol(x, y, window_length=5, polyorder=4):

    """
    Estimate the first derivative using Savitzky-Golay filtering.

    Parameters
    ----------
    x : array-like
        x data (must be evenly spaced)
    y : array-like
        y data
    window_length : int, optional
        Length of the filter window, in temperature units (default: 5)
    polyorder : int, optional
        Order of the polynomial used to fit the samples (default: 4)

    Returns
    -------
    numpy.ndarray
        First derivative of y with respect to x

    Notes
    -----
    This function will raise a ValueError if `x` is not evenly spaced.
    """

    # Check if x is evenly spaced
    if not is_evenly_spaced(x):
        raise ValueError("x must be evenly spaced for Savitzky-Golay filter.")

    # Calculate spacing (assuming uniform x)
    dx = np.mean(np.diff(x))
    odd_n_data_points_window_len = np.ceil(window_length / dx) // 2 * 2 + 1

    if polyorder >= odd_n_data_points_window_len:
        polyorder = int(odd_n_data_points_window_len - 1)

    # Apply Savitzky-Golay filter for first derivative
    dydx = savgol_filter(y, window_length=odd_n_data_points_window_len, polyorder=polyorder, deriv=1,mode="nearest")

    return dydx


def relative_errors(params,cov):
    """
    Calculate the relative errors of the fitted parameters.

    Parameters
    ----------
    params : numpy.ndarray
        Fitted parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters

    Returns
    -------
    numpy.ndarray
        Relative errors of the fitted parameters (in percent)
    """

    error = np.sqrt(np.diag(cov))
    rel_error = np.abs(error / params) * 100

    return rel_error

def find_line_outliers(m,b,x,y,sigma=2.5):
    """
    Find outliers in a linear fit using the sigma rule.

    Parameters
    ----------
    m : float
        Slope of the line
    b : float
        Intercept of the line
    x : array-like
        x data
    y : array-like
        y data
    sigma : float, optional
        Number of standard deviations to use for outlier detection (default: 2.5)

    Returns
    -------
    numpy.ndarray
        Indices of the outliers
    """

    # Calculate the residuals
    residuals = y - (m * x + b)

    # Calculate the standard deviation of the residuals
    std_residuals = np.std(residuals)

    # Calculate the mean of the residuals
    mean_residuals = np.mean(residuals)

    # Identify outliers
    outliers = np.where(np.abs(residuals - mean_residuals) > sigma * std_residuals)[0]

    return outliers

def get_rss(y, y_fit):

    """
    Compute the residual sum of squares.

    Parameters
    ----------
    y : array-like
        Observed values
    y_fit : array-like
        Fitted values

    Returns
    -------
    float
        Residual sum of squares
    """

    residuals = y - y_fit
    rss       = np.sum(residuals ** 2)

    return rss