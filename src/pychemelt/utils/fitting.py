"""
This module contains helper functions to fit unfolding data
Author: Osvaldo Burastero
"""

import numpy as np
from scipy.optimize     import curve_fit
from scipy.optimize     import least_squares

from .math import temperature_to_kelvin

from .math import get_rss

__all__ = [
    "fit_line_robust",
    "fit_quadratic_robust",
    "fit_exponential_robust",
    "fit_thermal_unfolding",
    "fit_tc_unfolding_single_slopes",
    "fit_tc_unfolding_shared_slopes_many_signals",
    "fit_tc_unfolding_many_signals"
]

def baseline_fx_name_to_req_params(baseline_fx_name):

    # If baseline_fx_name is not a string, extract the name from the function object
    if not isinstance(baseline_fx_name, str):
        baseline_fx_name = baseline_fx_name.__name__

    if 'constant' in baseline_fx_name:

        return [False, False]

    elif 'linear' in baseline_fx_name:

        return [True, False]

    elif 'quadratic' in baseline_fx_name:

        return [True, True]

    #elif baseline_fx_name == 'exponential':
    else:

        return [True, True]


def fit_line_robust(x,y):

    """
    Fit a line to the data using robust fitting

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    m : float
        Slope of the fitted line
    b : float
        Intercept of the fitted line
    """

    def linear_model(x,params):
        m,b = params
        return m * x + b

    p0 = np.polyfit(x, y, 1)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: linear_model(x, params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1
    )

    m, b = res_robust.x

    return m, b

def fit_quadratic_robust(x,y):

    """
    Fit a quadratic equation to the data using robust fitting

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    a : float
        Quadratic coefficient of the fitted polynomial
    b : float
        Linear coefficient of the fitted polynomial
    c : float
        Constant coefficient of the fitted polynomial
    """

    def model(x,params):
        a,b,c = params
        return a*np.square(x) + b*x + c

    p0 = np.polyfit(x, y, 2)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: model(x, params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1
    )

    a,b,c = res_robust.x

    return a,b,c

def fit_exponential_robust(x,y):

    """
    Fit an exponential function to the data using robust fitting.

    Notes
    -----
    Temperatures should be shifted to the reference (Tref) before calling this function.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    a : float
        Baseline
    c : float
        Pre-exponential factor
    alpha : float
        Exponential factor
    """

    def model(x,a,c,alpha):

        return a + c * np.exp(-alpha * x)

    # Initial parameter estimation by grid search

    rss = np.inf

    alpha_seq = np.logspace(-8, -1, 24)

    p0 = np.array( [np.min(y), np.min(y)/2])
    best_alpha = alpha_seq[0]

    low_bounds = [0, 0]

    high_bounds = [1e6, 1e6]

    for alpha in alpha_seq:

        def fit_fx(x,a,c):

            return a + c * np.exp(-alpha * x)

        params, cov = curve_fit(
            fit_fx,
            x,
            y,
            p0=p0,
            bounds=(low_bounds, high_bounds))

        pred =  fit_fx(x, *params)

        rss_curr = get_rss(y, pred)

        if rss_curr < rss:

            p0 = params
            rss = rss_curr
            best_alpha = alpha

    p0 = p0.tolist() + [best_alpha]

    low_bounds.append(0)
    high_bounds.append(1e6)

    # Perform robust fitting
    res_robust = least_squares(
        lambda params: model(x, *params) - y,
        p0,
        loss='soft_l1',
        f_scale=0.1,
        bounds=(low_bounds, high_bounds),
    )

    a,c,alpha = res_robust.x

    return a,c,alpha



def fit_thermal_unfolding(
    list_of_temperatures, 
    list_of_signals,
    initial_parameters,
    low_bounds, 
    high_bounds,
    signal_fx,
    baseline_native_fx,
    baseline_unfolded_fx,
    Cp,
    list_of_oligomer_conc=None):

    """
    Fit the thermal unfolding profile of many curves at the same time.

    This performs global fitting of shared thermodynamic parameters with per-curve baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Function to calculate the signal based on the parameters

    baseline_native_fx : callable
        function to calculate the native state baseline

    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline

    Cp : float
        Heat capacity change (passed to `signal_fx`)
    list_of_oligomer_conc : list, optional
        List of oligomer concentrations for each dataset (if applicable)

    Returns
    -------
    global_fit_params : numpy.ndarray
        Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters
    predicted_lst : list of numpy.ndarray
        Predicted signals for each dataset based on the fitted parameters
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    def thermal_unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes or pre-exp terms, folded
            - Single slopes  or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        n_datasets = len(list_of_temperatures)
        Tm, dh     = args[:2]  # Temperature of melting, Enthalpy of unfolding

        intercepts_folded   = args[2:(2 + n_datasets)]
        intercepts_unfolded = args[(2 + n_datasets):(2 + n_datasets * 2)]

        id_param_init = (2 + n_datasets * 2)
        n_params      = n_datasets

        if baseline_native_params[0]:

            p2_Ns = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p2_Ns = np.zeros(n_params)

        if baseline_unfolded_params[0]:

            p2_Us = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p2_Us = np.zeros(n_params)

        if baseline_native_params[1]:

            p3_Ns = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p3_Ns = np.zeros(n_params)

        if baseline_unfolded_params[1]:

            p3_Us = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p3_Us = np.zeros(n_params)

        signal = []

        for i, T in enumerate(list_of_temperatures):

            p1_N = intercepts_folded[i]
            p1_U = intercepts_unfolded[i]

            p2_N = p2_Ns[i]
            p2_U = p2_Us[i]

            p3_N = p3_Ns[i]
            p3_U = p3_Us[i]

            y = signal_fx(
                T, Tm, dh,
                p1_N, p2_N, p3_N,
                p1_U, p2_U, p3_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                Cp
            )
            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        thermal_unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
        )

    predicted = thermal_unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    return global_fit_params, cov, predicted_lst


'''
def fit_thermal_unfolding_oligomere(
    list_of_temperatures, 
    list_of_signals,
    oligomere_concentration,
    initial_parameters,
    low_bounds, 
    high_bounds,
    signal_fx,
    baseline_native_fx,
    baseline_unfolded_fx,
    Cp,
    list_of_oligomer_conc=None):

    """
    Fit the thermal unfolding profile of many curves at the same time.

    This performs global fitting of shared thermodynamic parameters with per-curve baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Function to calculate the signal based on the parameters

    baseline_native_fx : callable
        function to calculate the native state baseline

    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline

    Cp : float
        Heat capacity change (passed to `signal_fx`)
    list_of_oligomer_conc : list, optional
        List of oligomer concentrations for each dataset (if applicable)

    Returns
    -------
    global_fit_params : numpy.ndarray
        Fitted global parameters
    cov : numpy.ndarray
        Covariance matrix of the fitted parameters
    predicted_lst : list of numpy.ndarray
        Predicted signals for each dataset based on the fitted parameters
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    def thermal_unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes or pre-exp terms, folded
            - Single slopes  or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded


        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        n_datasets = len(list_of_temperatures)
        Tm, dh     = args[:2]  # Temperature of melting, Enthalpy of unfolding

        Tm = temperature_to_kelvin(Tm)

        intercepts_folded   = args[2:(2 + n_datasets)]
        intercepts_unfolded = args[(2 + n_datasets):(2 + n_datasets * 2)]

        id_param_init = (2 + n_datasets * 2)
        n_params      = n_datasets

        if baseline_native_params[0]:

            p2_Ns = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p2_Ns = np.zeros(n_params)

        if baseline_unfolded_params[0]:

            p2_Us = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p2_Us = np.zeros(n_params)

        if baseline_native_params[1]:

            p3_Ns = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p3_Ns = np.zeros(n_params)

        if baseline_unfolded_params[1]:

            p3_Us = args[id_param_init:(id_param_init+n_params)]
            id_param_init += n_params

        else:

            p3_Us = np.zeros(n_params)

        signal = []

        for i, T in enumerate(list_of_temperatures):
            
            T = temperature_to_kelvin(T)

            p1_N = intercepts_folded[i]
            p1_U = intercepts_unfolded[i]

            p2_N = p2_Ns[i]
            p2_U = p2_Us[i]

            p3_N = p3_Ns[i]
            p3_U = p3_Us[i]

            oligo_conc = oligomere_concentration

            y = signal_fx(
                T, oligo_conc, Tm, dh,
                0, p1_N, p2_N, p3_N,
                0, p1_U, p2_U, p3_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                Cp
            )
            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        thermal_unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
        )

    predicted = thermal_unfolding(1,*global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init+n])
        init += n

    return global_fit_params, cov, predicted_lst
'''


def fit_tc_unfolding_single_slopes(
        list_of_temperatures,
        list_of_signals,
        denaturant_concentrations,
        initial_parameters,
        low_bounds,
        high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        list_of_oligomer_conc=None,
        fit_m1=False,
        cp_value=None,
        tm_value=None,
        dh_value=None):
    """
    Fit thermochemical unfolding curves sharing thermodynamic parameters while allowing different baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    denaturant_concentrations : list
        Denaturant concentrations (one per dataset)
    initial_parameters : array-like
        Initial guess for parameters
    low_bounds : array-like
        Lower bounds for parameters
    high_bounds : array-like
        Upper bounds for parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    list_of_oligomer_conc : list, optional
        Oligomer concentrations per dataset
    fit_m1 : bool, optional
        Whether to fit temperature dependence of m-value
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    """

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_datasets = len(list_of_temperatures)

    def unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Global Cp0
            - Global m0
            - Global m1 (if fit_m1 is True)
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes or pre-exp terms, folded
            - Single slopes or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        id_param_init = 0

        if tm_value is None:

            Tm = args[0]  # Temperature of melting
            id_param_init += 1

        else:

            Tm = tm_value

        if dh_value is None:

            DHm = args[id_param_init]  # Enthalpy of unfolding
            id_param_init += 1

        else:

            DHm = dh_value

        if cp_value is None:

            Cp0 = args[id_param_init]  # Cp0
            id_param_init += 1

        else:

            Cp0 = cp_value

        m0 = args[id_param_init]  # m0
        id_param_init += 1

        if fit_m1:

            m1 = args[id_param_init]
            id_param_init += 1

        else:

            m1 = 0

        param_1_folded = args[id_param_init:(id_param_init + n_datasets)]
        param_1_unfolded = args[(id_param_init + n_datasets):(id_param_init + n_datasets * 2)]

        id_param_init = (id_param_init + n_datasets * 2)
        n_params = n_datasets

        if baseline_native_params[0]:
            param_2_folded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        if baseline_unfolded_params[0]:
            param_2_unfolded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        if baseline_native_params[1]:
            param_3_folded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        if baseline_unfolded_params[1]:
            param_3_unfolded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        signal = []

        for i, T in enumerate(list_of_temperatures):

            p_1_N = param_1_folded[i]
            p_1_U = param_1_unfolded[i]

            p_2_N = 0 if not baseline_native_params[0] else param_2_folded[i]
            p_2_U = 0 if not baseline_unfolded_params[0] else param_2_unfolded[i]

            p_3_N = 0 if not baseline_native_params[1] else param_3_folded[i]
            p_3_U = 0 if not baseline_unfolded_params[1] else param_3_unfolded[i]

            c = 0 if list_of_oligomer_conc is None else list_of_oligomer_conc[i]

            d = denaturant_concentrations[i]

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                0, p_1_N, p_2_N, p_3_N,
                0, p_1_U, p_2_U, p_3_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                c
            )

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
    )

    predicted = unfolding(1, *global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    return global_fit_params, cov, predicted_lst

def fit_oligomere_unfolding_single_slopes(
        list_of_temperatures,
        list_of_signals,
        oligomere_concentrations,
        initial_parameters,
        low_bounds,
        high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        cp_value=None,
        tm_value=None,
        dh_value=None):
    """
    Fit thermochemical unfolding curves sharing thermodynamic parameters while allowing different baselines.

    Parameters
    ----------
    list_of_temperatures : list of array-like
        Temperature arrays for each dataset
    list_of_signals : list of array-like
        Signal arrays for each dataset
    oligomere_concentrations : list
        Oligomere concentrations (one per dataset)
    initial_parameters : array-like
        Initial guess for parameters
    low_bounds : array-like
        Lower bounds for parameters
    high_bounds : array-like
        Upper bounds for parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    list_of_oligomer_conc : list, optional
        Oligomer concentrations per dataset
    fit_m1 : bool, optional
        Whether to fit temperature dependence of m-value
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    """

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_datasets = len(list_of_temperatures)

    def unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Global Cp0
            - Single intercepts, folded
            - Single intercepts, unfolded
            - Single slopes or pre-exp terms, folded
            - Single slopes or pre-exp terms, unfolded
            - Single quadratic or exponential coefficients, folded
            - Single quadratic or exponential coefficients, unfolded

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        id_param_init = 0

        if tm_value is None:

            Tm = args[0]  # Temperature of melting
            id_param_init += 1

        else:

            Tm = tm_value

        Tm = temperature_to_kelvin(Tm)

        if dh_value is None:

            DHm = args[id_param_init]  # Enthalpy of unfolding
            id_param_init += 1

        else:

            DHm = dh_value

        if cp_value is None:

            Cp0 = args[id_param_init]  # Cp0
            id_param_init += 1

        else:

            Cp0 = cp_value


        param_1_folded = args[id_param_init:(id_param_init + n_datasets)]
        param_1_unfolded = args[(id_param_init + n_datasets):(id_param_init + n_datasets * 2)]

        id_param_init = (id_param_init + n_datasets * 2)
        n_params = n_datasets

        if baseline_native_params[0]:
            param_2_folded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        if baseline_unfolded_params[0]:
            param_2_unfolded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        if baseline_native_params[1]:
            param_3_folded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        if baseline_unfolded_params[1]:
            param_3_unfolded = args[id_param_init:(id_param_init + n_params)]
            id_param_init += n_params

        signal = []

        for i, T in enumerate(list_of_temperatures):
            
            T = temperature_to_kelvin(T)

            p_1_N = param_1_folded[i]
            p_1_U = param_1_unfolded[i]

            p_2_N = 0 if not baseline_native_params[0] else param_2_folded[i]
            p_2_U = 0 if not baseline_unfolded_params[0] else param_2_unfolded[i]

            p_3_N = 0 if not baseline_native_params[1] else param_3_folded[i]
            p_3_U = 0 if not baseline_unfolded_params[1] else param_3_unfolded[i]

            C = oligomere_concentrations[i]


            y = signal_fx(
                T,C, Tm, DHm,
                0, p_1_N, p_2_N, p_3_N,
                0, p_1_U, p_2_U, p_3_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                Cp0,
            )

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
    )

    predicted = unfolding(1, *global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    return global_fit_params, cov, predicted_lst

def fit_tc_unfolding_shared_slopes_many_signals(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        denaturant_concentrations,
        initial_parameters,
        low_bounds,
        high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        list_of_oligomer_conc=None,
        fit_m1=False,
        cp_value=None,
        tm_value=None,
        dh_value=None):
    """
    Fit thermochemical unfolding curves for multiple signal types sharing thermodynamic parameters and slopes.

    Parameters
    ----------
    list_of_temperatures : list of array-like
    list_of_signals : list of array-like
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    denaturant_concentrations : list
        Denaturant concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
     baseline_native_fx : callable
        function to calculate the baseline for the native state
    baseline_unfolded_fx : callable
        function to calculate the baseline for the unfolded state
    list_of_oligomer_conc : list, optional
        Oligomer concentrations per dataset
    fit_m1 : bool, optional
        Whether to fit temperature dependence of m-value
    cp_value, tm_value, dh_value : float or None, optional
        Optional fixed thermodynamic parameters

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    n_datasets = len(list_of_temperatures)

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx)

    def unfolding(dummyVariable, *args):

        """
        Calculate the thermal unfolding profile of many curves at the same time

        Requires:

            - The 'listOfTemperatures' containing each of them a single dataset

        The other arguments have to be in the following order:

            - Global melting temperature
            - Global enthalpy of unfolding
            - Global Cp0
            - Global m0
            - Global m1 (if fit_m1 is True)
            - Single intercepts, folded - one per curve
            - Single intercepts, unfolded - one per curve
            - Shared slopes or pre-exp term, folded - one per signal type
            - Shared slopes or pre-exp term, unfolded - one per signal type
            - Shared quadratic or exponential coefficients, folded - one per signal type
            - Shared quadratic or exponential coefficients, unfolded - one per signal type

        Returns:

            The melting curves based on the parameters Temperature of melting, enthalpy of unfolding,
                slopes and intercept of the folded and unfolded states

        """

        id_param_init = 0

        if tm_value is None:

            Tm = args[0]  # Temperature of melting
            id_param_init += 1

        else:

            Tm = tm_value

        if dh_value is None:

            DHm = args[id_param_init]  # Enthalpy of unfolding
            id_param_init += 1

        else:

            DHm = dh_value

        if cp_value is None:

            Cp0 = args[id_param_init]  # Cp0
            id_param_init += 1

        else:

            Cp0 = cp_value

        m0 = args[id_param_init]  # m0
        id_param_init += 1

        if fit_m1:
            m1 = args[id_param_init]
            id_param_init += 1
        else:
            m1 = 0

        intercepts_folded = args[id_param_init:(id_param_init + n_datasets)]
        intercepts_unfolded = args[(id_param_init + n_datasets):(id_param_init + n_datasets * 2)]

        id_param_init = (id_param_init + n_datasets * 2)

        if baseline_native_params[0]:
            p2_n_s = args[id_param_init:(id_param_init + n_signals)]
            id_param_init += n_signals
        else:
            p2_n_s = [0] * n_signals

        if baseline_unfolded_params[0]:
            p2_u_s = args[id_param_init:(id_param_init + n_signals)]
            id_param_init += n_signals
        else:
            p2_u_s = [0] * n_signals

        if baseline_native_params[1]:
            p3_n_s = args[id_param_init:(id_param_init + n_signals)]
            id_param_init += n_signals
        else:
            p3_n_s = [0] * n_signals

        if baseline_unfolded_params[1]:
            p3_u_s = args[id_param_init:(id_param_init + n_signals)]
            id_param_init += n_signals
        else:
            p3_u_s = [0] * n_signals

        signal = []

        for i, T in enumerate(list_of_temperatures):

            p1_n = intercepts_folded[i]
            p1_u = intercepts_unfolded[i]

            c = 0 if list_of_oligomer_conc is None else list_of_oligomer_conc[i]

            d = denaturant_concentrations[i]

            p2_n = p2_n_s[signal_ids[i]]
            p2_u = p2_u_s[signal_ids[i]]

            p3_n = p3_n_s[signal_ids[i]]
            p3_u = p3_u_s[signal_ids[i]]

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                0,p1_n, p2_n, p3_n,
                0,p1_u, p2_u, p3_u,
                baseline_native_fx,
                baseline_unfolded_fx,
                c
            )

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters, bounds=(low_bounds, high_bounds)
    )

    predicted = unfolding(1, *global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    return global_fit_params, cov, predicted_lst


def fit_tc_unfolding_many_signals(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        denaturant_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        baseline_native_fx,
        baseline_unfolded_fx,
        oligomer_concentrations=None,
        fit_m1=False,
        model_scale_factor=False,
        scale_factor_exclude_ids=[],
        cp_value=None,
        fit_native_den_slope=True,
        fit_unfolded_den_slope=True):
    """
    Fit thermochemical unfolding curves for many signals (optimized variant).

    Parameters
    ----------
    list_of_temperatures : list of array-like
    list_of_signals : list of array-like
    signal_ids : list of int
        Signal-type id for each dataset (0..n_signals-1)
    denaturant_concentrations : list
        Denaturant concentrations for each dataset (flattened across signals)
    initial_parameters : array-like
        Initial guess for the parameters
    low_bounds : array-like
        Lower bounds for the parameters
    high_bounds : array-like
        Upper bounds for the parameters
    signal_fx : callable
        Signal model function
    baseline_native_fx : callable
        function to calculate the native state baseline
    baseline_unfolded_fx : callable
        function to calculate the unfolded state baseline
    oligomer_concentrations : list, optional
        Oligomer concentrations per dataset (used by oligomeric models)
    fit_m1 : bool, optional
        Whether to include and fit temperature dependence of the m-value (m1)
    model_scale_factor : bool, optional
        If True, include a per-denaturant concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1
    cp_value : float or None, optional
        If provided, Cp is fixed to this value and not fitted

    Returns
    -------
    global_fit_params : numpy.ndarray
    cov : numpy.ndarray
    predicted_lst : list of numpy.ndarray
    """

    all_signal = np.concatenate(list_of_signals, axis=0)

    n_signals = np.max(signal_ids) + 1

    nr_den = int(len(denaturant_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    baseline_native_params = [fit_native_den_slope] + baseline_fx_name_to_req_params(baseline_native_fx)
    baseline_unfolded_params = [fit_unfolded_den_slope] + baseline_fx_name_to_req_params(baseline_unfolded_fx)

    def unfolding(dummyVariable, *args):

        """
        The parameters order is as follows:

            Tm, Dh, Cp0 and m-value

            Intercept folded
            Intercept unfolded

            Temperature slope or term pre-exponential factor folded
            Temperature slope term or pre-exponential factor unfolded

            Denaturant slope term folded
            Denaturant slope term unfolded

            Quadratic coefficient or exponential coefficient folded
            Quadratic coefficient or exponential coefficient unfolded

        """

        if cp_value is not None:

            Cp0 = cp_value
            Tm, DHm, m0 = args[:3]  # Enthalpy of unfolding, Temperature of melting, m0, m1
            id_param_init = 3

        else:

            Tm, DHm, Cp0, m0 = args[:4]  # Enthalpy of unfolding, Temperature of melting, Cp0, m0, m1
            id_param_init = 4

        if fit_m1:
            m1 = args[id_param_init]
            id_param_init += 1
        else:
            m1 = 0

        # Intercept parameters
        p2_Ns = args[id_param_init:id_param_init + n_signals]
        p2_Us = args[id_param_init + n_signals:id_param_init + 2 * n_signals]

        id_param_init = id_param_init + 2 * n_signals

        # Temperature slope or pre-exponential parameters
        if baseline_native_params[1]:

            p3_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            p3_Ns = [0] * n_signals

        if baseline_unfolded_params[1]:

            p3_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            p3_Us = [0] * n_signals

        # Denaturant slope parameters
        if baseline_native_params[0]:

            p1_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:

            p1_Ns = [0] * n_signals

        if baseline_unfolded_params[0]:

            p1_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:

            p1_Us = [0] * n_signals

        # Temperature-dependent quadratic or exponential coefficients
        if baseline_native_params[2]:

            p4_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            p4_Ns = [0] * n_signals

        if baseline_unfolded_params[2]:

            p4_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals

        else:
            p4_Us = [0] * n_signals

        if model_scale_factor:
            # One per denaturant concentration
            factors = args[id_param_init:id_param_init + (nr_den - len(scale_factor_exclude_ids))]

            for id_ex in scale_factor_exclude_ids:
                factors = np.insert(factors, id_ex, 1)

            # Repeat the list so have the same length as list_of_temperatures, equal to denaturant concentration * number of signals
            factors = np.tile(factors, n_signals)

            id_param_init += nr_den

        signal = []

        for i, T in enumerate(list_of_temperatures):

            p1_N = p1_Ns[signal_ids[i]]
            p1_U = p1_Us[signal_ids[i]]
            p2_N = p2_Ns[signal_ids[i]]
            p2_U = p2_Us[signal_ids[i]]
            p3_N = p3_Ns[signal_ids[i]]
            p3_U = p3_Us[signal_ids[i]]
            p4_N = p4_Ns[signal_ids[i]]
            p4_U = p4_Us[signal_ids[i]]

            d = denaturant_concentrations[i]

            c = 0 if oligomer_concentrations is None else oligomer_concentrations[i]

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                p1_N, p2_N, p3_N, p4_N,
                p1_U, p2_U, p3_U, p4_U,
                baseline_native_fx,
                baseline_unfolded_fx,
                c
            )

            scale_factor = 1 if not model_scale_factor else factors[i]

            y = y * scale_factor

            signal.append(y)

        return np.concatenate(signal, axis=0)

    global_fit_params, cov = curve_fit(
        unfolding, 1, all_signal,
        p0=initial_parameters,
        bounds=(low_bounds, high_bounds))

    predicted = unfolding(1, *global_fit_params)

    # Convert predict to list of lists
    predicted_lst = []

    init = 0
    for T in list_of_temperatures:
        n = len(T)
        predicted_lst.append(predicted[init:init + n])
        init += n

    return global_fit_params, cov, predicted_lst



def evaluate_need_to_refit(
        global_fit_params,
        high_bounds,
        low_bounds,
        p0,
        fit_m1=False,
        check_cp=True,
        check_dh=True,
        check_tm=True,
        fixed_cp=False,
        threshold=0.05):

    """
    Check and expand parameter bounds when fitted parameters are too close to boundaries.

    Parameters
    ----------
    global_fit_params : array-like
        Fitted parameters
    high_bounds : array-like
        Upper bounds
    low_bounds : array-like
        Lower bounds
    p0 : array-like
        Initial guess for parameters
    fit_m1 : bool, optional
    check_cp, check_dh, check_tm : bool, optional
    fixed_cp : bool, optional
    threshold : float, optional
        Threshold to compare if the fitted parameters are too close to the boundaries

    Returns
    -------
    re_fit : bool
        True if a refit is recommended after bounds expansion
    p0 : array-like
        Updated initial parameters
    low_bounds : array-like
        Updated lower bounds
    high_bounds : array-like
        Updated upper bounds
    """

    # We need to create copies of the arrays, otherwise they will be overwritten
    global_fit_params = global_fit_params.copy()
    p0 = p0.copy()
    high_bounds = high_bounds.copy()
    low_bounds = low_bounds.copy()

    re_fit = False

    # Check the Tm boundary - upper
    tm_diff = high_bounds[0] - global_fit_params[0]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        high_bounds[0] = global_fit_params[0] + 12
        p0[0] = global_fit_params[0] + 5
        re_fit = True

    # Check the Tm boundary - lower
    tm_diff = global_fit_params[0] - low_bounds[0]

    # Expand the boundary if the Tm is too close to the boundary
    if tm_diff < 6 and check_tm:
        low_bounds[0] = global_fit_params[0] - 12
        p0[0] = global_fit_params[0] - 5
        re_fit = True

    # Check the Dh boundary
    dh_diff = high_bounds[1] - global_fit_params[1]
    # Expand the boundary if the Dh is too close to the boundary
    if dh_diff < 20 and check_dh:
        high_bounds[1] = global_fit_params[1] + 80
        p0[1] = global_fit_params[1] + 50
        re_fit = True

    id_next = 2
    if not fixed_cp:

        # Check the Cp boundary
        cp_diff = high_bounds[2] - global_fit_params[2]
        # Expand the boundary if the Cp is too close to the boundary
        if cp_diff < 0.25 and check_cp:
            high_bounds[2] = global_fit_params[2] + 1
            p0[2] = global_fit_params[2] + 0.5
            re_fit = True
        
        id_next += 1

    # Check the m-value boundary
    m_diff = high_bounds[id_next] - global_fit_params[id_next]
    # Expand the boundary if the m-value is too close to the boundary
    if m_diff < 0.5:
        high_bounds[id_next] = global_fit_params[id_next] + 2
        p0[id_next] = global_fit_params[id_next] + 0.5
        re_fit = True

    # Evaluate if m1 is fitted
    id_start = id_next + 1

    if fit_m1:

        m1_diff = high_bounds[id_start] - global_fit_params[id_start]
        # Expand the boundary if the m-value is too close to the boundary
        if m1_diff < 0.1:
            high_bounds[id_start] = global_fit_params[id_start] + 1
            re_fit = True

        m1_diff = global_fit_params[id_start] - low_bounds[id_start]
        # Expand the boundary if the m-value is too close to the boundary
        if m1_diff < 0.1:
            low_bounds[id_start] = global_fit_params[id_start] - 1
            re_fit = True

        id_start += 1

    difference_to_upper = np.array([np.abs((a-b)/a) if a != np.inf and a != 0  else np.inf for a, b in zip(high_bounds[id_start:], global_fit_params[id_start:])])
    difference_to_lower = np.array([np.abs((a-b)/a) if b != -np.inf and a != 0 else np.inf for a, b in zip(global_fit_params[id_start:], low_bounds[id_start:])])

    # Evaluate all the other parameters
    for i in (range(len(global_fit_params)-id_start)):

        diff_to_high_i = difference_to_upper[i]
        diff_to_low_i = difference_to_lower[i]

        if diff_to_high_i < threshold:

            value = high_bounds[i+id_start]

            high_bounds[i+id_start] = value * 50 if value > 0 else value / 50
            re_fit = True

        if diff_to_low_i < threshold:

            value = low_bounds[i+id_start]
            low_bounds[i+id_start] = value * 50 if value < 0 else value / 50
            re_fit = True

    return re_fit, p0, low_bounds, high_bounds

def evaluate_fitting_and_refit(
        global_fit_params,
        cov,
        predicted,
        high_bounds,
        low_bounds,
        p0,
        fit_m_dep,
        limited_cp,
        limited_dh,
        limited_tm,
        fixed_cp,
        kwargs,
        fit_fx,
        n = 3):

    """
    Evaluate if the fitted parameters are too close to the fitting boundaries.
    If they are, re-fit with new expanded boundaries

    Parameters
    ----------
    global_fit_params: array-like
        fitted parameters
    cov: array-like
        covariance matrix of the fitted parameters
    predicted: list
        list of lists with the fitted values
    high_bounds: array-like
        upper bounds of the fitting parameters
    low_bounds: array-like
        lower bounds of the fitting parameters
    p0: array-like
        initial guess for the fitting parameters
    fit_m_dep: boolean
        if the m-dependence on temperature is fitted
    limited_cp: boolean
        if the cp bounds are user-defined
    limited_dh: boolean
        if the DH bounds are user-defined
    limited_tm: boolean
        if the Tm values are user-defined
    fixed_cp: boolean
        if the cp value is fixed
    kwargs: dict
        dictionary with the arguments for the fitting function
    fit_fx: callable
        function to perform the fitting
    n: int, optional
        number of times to re-fit

    Returns
    -------
    global_fit_params: array-like
        fitted parameters
    cov: array-like
        covariance matrix of the fitted parameters
    predicted: list
        list of lists with the fitted values
    p0: array-like
        initial guess for the fitting parameters
    low_bounds: array-like
        lower bounds of the fitting parameters
    high_bounds: array-like
        higher bounds of the fitting parameters
    """

    for _ in range(n):

        re_fit, p0_new, low_bounds_new, high_bounds_new = evaluate_need_to_refit(
            global_fit_params,
            high_bounds,
            low_bounds,
            p0,
            fit_m1=fit_m_dep,
            check_cp=not limited_cp,
            check_dh=not limited_dh,
            check_tm=not limited_tm,
            fixed_cp=fixed_cp
        )

        if re_fit:

            p0, low_bounds, high_bounds = p0_new, low_bounds_new, high_bounds_new

            kwargs['initial_parameters'] = p0
            kwargs['low_bounds'] = low_bounds
            kwargs['high_bounds'] = high_bounds

            global_fit_params, cov, predicted = fit_fx(**kwargs)

        else:

            break

    return global_fit_params, cov, predicted, p0, low_bounds, high_bounds