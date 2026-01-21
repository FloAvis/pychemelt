"""
This module contains helper functions to process data
Author: Osvaldo Burastero
"""
import re
import numpy as np
import itertools

from collections import Counter

from .math import shift_temperature, relative_errors

from .fitting import (
    fit_line_robust,
    fit_quadratic_robust,
    fit_exponential_robust,
    fit_thermal_unfolding,
    fit_thermal_unfolding_oligomere,
    baseline_fx_name_to_req_params
)

from .fractions import (
    two_state_rev_unfolding_fractions,
    two_state_dimer_unfolding_fractions,
    two_state_trimer_unfolding_fractions,
    two_state_tetramer_unfolding_fractions,
    map_two_state_model_to_fractions_fx,
)

from .signals import (
    signal_two_state_t_unfolding,
    two_state_thermal_unfold_curve_dimer,
    two_state_thermal_unfold_curve_trimer,
    two_state_thermal_unfold_curve_tetramer,
    map_two_state_model_to_signal_fx,
)

from .palette import VIRIDIS

__all__ = [
    'set_param_bounds',
    'expand_temperature_list',
    'clean_conditions_labels',
    'subset_signal_by_temperature',
    'guess_Tm_from_derivative',
    'estimate_signal_baseline_params',
    'fit_local_thermal_unfolding_to_signal_lst',
    'fit_oligomere_local_thermal_unfolding_to_signal_lst',
    're_arrange_predictions',
    're_arrange_params',
    'subset_data',
    'get_colors_from_numeric_values',
    'combine_sequences',
    'adjust_value_to_interval'
]

def set_param_bounds(p0,param_names):

    low_bounds = []
    high_bounds = []

    for p in p0:

        if -1 < p < 1:

            low_bounds.append(-5e2)
            high_bounds.append(5e2)

        elif p > 1:

            low_bounds.append(p/1e3)
            high_bounds.append(p*1e3)

        else:

            low_bounds.append(p*1e3)
            high_bounds.append(p/1e3)

    # Set low bounds to zero for specific parameters
    # For example for all parameters containing 'exp'

    for i,p in enumerate(param_names):

        c1 = 'intercept' in p and 'native' in p
        c2 = 'exponential_coefficient' in p
        c3 = 'pre_exponential_factor' in p
        c4 = low_bounds[i] < 0

        if (c1 or c2 or c3) and c4:

            low_bounds[i] = 0

    return low_bounds, high_bounds

def expand_temperature_list(temp_lst,signal_lst):

    """
    Expand the temperature list to match the length of the signal list.

    Parameters
    ----------
    temp_lst : list
        List of temperatures
    signal_lst : list
        List of signals

    Returns
    -------
    list
        Expanded temperature list
    """

    if len(temp_lst) < len(signal_lst):
        temp_lst = [temp_lst[0] for _ in signal_lst]

    return temp_lst


def delete_words_appearing_more_than_five_times(strings):
    """
    Deletes words that appear more than 5 times from a list of strings.

    Parameters
    ----------
    strings : list of str
        List of strings.

    Returns
    -------
    list of str
        List of strings with frequent words removed.
    """
    all_words = " ".join(strings).split()
    word_counts = Counter(all_words)
    words_to_remove = {word for word, count in word_counts.items() if count > 5}
    cleaned_strings = [
        " ".join(word for word in string.split() if word not in words_to_remove)
        for string in strings
    ]
    return cleaned_strings


def remove_letter_number_combinations(text):
    """
    Removes any combination of a single letter followed by one or two digits (e.g., A1, B10, D5) from the input string.

    Parameters
    ----------
    text : str
        The input string from which patterns should be removed.

    Returns
    -------
    str
        The cleaned string with all matching patterns removed.
    """
    # Pattern: one letter (case-insensitive) followed by 1 or 2 digits, as a whole word
    pattern = r'\b[A-Za-z]\d{1,2}\b'
    cleaned_text = re.sub(pattern, '', text)
    # Optionally remove extra spaces left behind
    return re.sub(r'\s{2,}', ' ', cleaned_text).strip()


def remove_numbers_after_letter(text):
    """
    Removes all numbers coming after a letter until an underscore or space appears.

    Parameters
    ----------
    text : str
        The input string.

    Returns
    -------
    str
        The cleaned string.
    """

    pattern = r'(?<=[A-Za-z])\d+(?=[_\s])'

    return re.sub(pattern, '', text)


def remove_non_numeric_char(input_string):
    """
    Remove all non-numeric characters except dots from a string.

    Parameters
    ----------
    input_string : str
        Input string

    Returns
    -------
    str
        String with non-numeric characters (except dots) removed
    """

    return re.sub(r'[^\d.]', '', input_string)

def adjust_value_to_interval(value,lower_bound,upper_bound,shift):

    """
    Verify that a value is within the specified bounds.
    If the value is outside the bounds, adjust it to the nearest bound.
    Parameters
    ----------
    value : float
        The value to be adjusted.
    lower_bound : float
        The lower bound of the interval.
    upper_bound : float
        The upper bound of the interval.
    shift : float
        How much to shift the value if it is outside the bounds.
    """

    if value < lower_bound:
        return lower_bound + shift
    elif value > upper_bound:
        return upper_bound - shift
    else:
        return value


def clean_conditions_labels(conditions):
    """
    Clean the conditions labels by removing unwanted characters and patterns.

    Parameters
    ----------
    conditions : list
        List of condition strings.

    Returns
    -------
    list
        List of cleaned condition strings.
    """
    conditions = [text.replace("_", " ") for text in conditions]
    conditions = delete_words_appearing_more_than_five_times(conditions)
    conditions = [remove_letter_number_combinations(text) for text in conditions]
    conditions = [remove_numbers_after_letter(text)       for text in conditions]
    conditions = [remove_non_numeric_char(text)           for text in conditions]

    # Try to convert to float or return 0
    for i, text in enumerate(conditions):
        try:
            conditions[i] = float(text)
        except ValueError:
            conditions[i] = 0.0

    return conditions


def subset_signal_by_temperature(signal_lst, temp_lst, min_temp, max_temp):
    """
    Subset the signal and temperature lists based on the specified temperature range.

    Parameters
    ----------
    signal_lst : list
        List of signal arrays.
    temp_lst : list
        List of temperature arrays.
    min_temp : float
        Minimum temperature for subsetting.
    max_temp : float
        Maximum temperature for subsetting.

    Returns
    -------
    tuple
        Tuple containing the subsetted signal and temperature lists.
    """

    # Limit the signal to the temperature range
    subset_signal = [s[np.logical_and(t >= min_temp, t <= max_temp)] for s,t in zip(signal_lst,temp_lst)]
    subset_temp   = [t[np.logical_and(t >= min_temp, t <= max_temp)] for t in temp_lst]

    return subset_signal, subset_temp

def guess_Tm_from_derivative(temp_lst, deriv_lst, x1, x2):

    t_melting_init = []

    for sd,t in zip(deriv_lst,temp_lst):

        min_t = np.min(t)
        max_t = np.max(t)

        # max_t - min_t can't be lower than x2
        if (max_t - min_t) < x2:
            raise ValueError('The temperature range is too small to estimate the Tm. ' \
            'Please increase the range or decrease x2.')

        der_temp_init = sd[np.logical_and(t < min_t + x2, t > min_t + x1)]
        der_temp_end  = sd[np.logical_and(t < max_t - x1, t > max_t - x2)]

        med_init = np.median(der_temp_init, axis=0)
        med_end  = np.median(der_temp_end,  axis=0)

        mid_value = (med_init + med_end) / 2
        mid_value = mid_value * np.where(mid_value > 0, 1, -1)

        der_temp  = sd[np.logical_and(t > min_t + x1, t < max_t - x1)]
        temp_temp = t[np.logical_and(t > min_t + x1, t < max_t - x1)]

        der_temp = np.add(der_temp, mid_value)

        max_der = np.abs(np.max(der_temp, axis=0))
        min_der = np.abs(np.min(der_temp, axis=0))

        idx = np.argmax(der_temp) if max_der > min_der else np.argmin(der_temp)

        t_melting_init.append(temp_temp[idx])

    return t_melting_init

def estimate_signal_baseline_params(
    signal_lst,
    temp_lst,
    native_baseline_type,
    unfolded_baseline_type,
    window_range_native=12,
    window_range_unfolded=12):
        
    """
    Estimate the baseline parameters for the sample

    Parameters
    ---------
    window_range_native : float
        Range of the temperature window to estimate the native state baseline
    window_range_unfolded : float
        Range of the temperature window to estimate the unfolded state baseline
    native_baseline_type : str
        options: 'constant', 'linear', 'quadratic', 'exponential'
    unfolded_baseline_type : str
        options: 'constant', 'linear', 'quadratic', 'exponential'

    Returns

    """

    p1Ns  = []
    p1Us  = []
    p2Ns  = []
    p2Us  = []
    p3Ns  = []
    p3Us  = []

    for s,t in zip(signal_lst,temp_lst):

        signal_native = s[t < np.min(t) + window_range_native]
        temp_native   = t[t < np.min(t) + window_range_native]

        # Shift temperature to be centered at Tref !!! defined in constants.py
        temp_native = shift_temperature(temp_native)

        signal_denat  = s[t > np.max(t) - window_range_unfolded]
        temp_denat    = t[t > np.max(t) - window_range_unfolded]

        # Shift temperature to be centered at Tref !!! defined in constants.py
        temp_denat = shift_temperature(temp_denat)

        if native_baseline_type == 'constant':

            p1N = np.median(signal_native)
            p1Ns.append(p1N)

        if unfolded_baseline_type == 'constant':

            p1U = np.median(signal_denat)
            p1Us.append(p1U)

        if native_baseline_type == 'linear':

            p2N, p1N = fit_line_robust(temp_native,signal_native)

            p2Ns.append(p2N)
            p1Ns.append(p1N)

        if unfolded_baseline_type == 'linear':

            p2U, p1U = fit_line_robust(temp_denat,signal_denat)

            p2Us.append(p2U)
            p1Us.append(p1U)

        if native_baseline_type == 'quadratic':

            p3N, p2N, p1N = fit_quadratic_robust(temp_native,signal_native)

            p3Ns.append(p3N)
            p2Ns.append(p2N)
            p1Ns.append(p1N)

        if unfolded_baseline_type == 'quadratic':

            p3U, p2U, p1U = fit_quadratic_robust(temp_denat,signal_denat)

            p3Us.append(p3U)
            p2Us.append(p2U)
            p1Us.append(p1U)

        if native_baseline_type == 'exponential':

            p1N, p2N, p3N = fit_exponential_robust(temp_native,signal_native)

            p3Ns.append(p3N)
            p2Ns.append(p2N)
            p1Ns.append(p1N)

        if unfolded_baseline_type == 'exponential':

            p1U, p2U, p3U = fit_exponential_robust(temp_denat,signal_denat)

            p3Us.append(p3U)
            p2Us.append(p2U)
            p1Us.append(p1U)

    return p1Ns, p1Us, p2Ns, p2Us, p3Ns, p3Us


def fit_local_thermal_unfolding_to_signal_lst(
    signal_lst,
    temp_lst,
    t_melting_init,
    p1_Ns,
    p1_Us,
    p2_Ns,
    p2_Us,
    p3_Ns,
    p3_Us,
    baseline_native_fx,
    baseline_unfolded_fx):
    
    predicted_lst = []
    Tms           = []
    dHs           = []

    # Obtain the name of the function baseline_native_fx and baseline_unfolded_fx
    baseline_native_fx_name = baseline_native_fx.__name__
    baseline_unfolded_fx_name = baseline_unfolded_fx.__name__

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx_name)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx_name)

    i = 0
    for s,t in zip(signal_lst,temp_lst):

        p0 = np.array([t_melting_init[i], 85, p1_Ns[i], p1_Us[i]])

        if baseline_native_params[0]:
            p0 = np.concatenate([p0, [p2_Ns[i]]])
        if baseline_unfolded_params[0]:
            p0 = np.concatenate([p0, [p2_Us[i]]])

        if baseline_native_params[1]:
            p0 = np.concatenate([p0, [p3_Ns[i]]])
        if baseline_unfolded_params[1]:
            p0 = np.concatenate([p0, [p3_Us[i]]])

        low_bounds  = p0.copy()
        high_bounds = p0.copy()

        low_bounds[2:]  = [x / 200 - 50 if x > 0 else 200 * x - 50 for x in low_bounds[2:]]
        high_bounds[2:] = [200 * x + 50 if x > 0 else x / 200 + 50 for x in high_bounds[2:]]

        low_bounds[0]  = np.min(t)
        high_bounds[0] = np.max(t) + 15

        low_bounds[1]  = 10
        high_bounds[1] = 500

        try:

            params, cov, predicted = fit_thermal_unfolding(
                list_of_temperatures=[t],
                list_of_signals=[s],
                initial_parameters=p0,
                low_bounds=low_bounds,
                high_bounds=high_bounds,
                signal_fx=signal_two_state_t_unfolding,
                baseline_native_fx=baseline_native_fx,
                baseline_unfolded_fx=baseline_unfolded_fx,
                Cp=0)

            rel_errors = relative_errors(params, cov)

            if rel_errors[0] < 50 and rel_errors[1] < 50:
                Tms.append(params[0])
                dHs.append(params[1])

            predicted_lst.append(predicted[0])

        except:

            pass

        i += 1

    return Tms, dHs, predicted_lst

def fit_oligomere_local_thermal_unfolding_to_signal_lst(
    signal_lst,
    temp_lst,
    oligomere_concentration,
    t_melting_init,
    p1_Ns,
    p1_Us,
    p2_Ns,
    p2_Us,
    p3_Ns,
    p3_Us,
    baseline_native_fx,
    baseline_unfolded_fx,
    model):
    
    predicted_lst = []
    Tms           = []
    dHs           = []

    signal_fx    = map_two_state_model_to_signal_fx(model)
    fractions_fx = map_two_state_model_to_fractions_fx(model)

    # Obtain the name of the function baseline_native_fx and baseline_unfolded_fx
    baseline_native_fx_name = baseline_native_fx.__name__
    baseline_unfolded_fx_name = baseline_unfolded_fx.__name__

    baseline_native_params = baseline_fx_name_to_req_params(baseline_native_fx_name)
    baseline_unfolded_params = baseline_fx_name_to_req_params(baseline_unfolded_fx_name)

    i = 0
    for s,t in zip(signal_lst,temp_lst):

        p0 = np.array([t_melting_init[i], 85, oligomere_concentration, p1_Ns[i], p1_Us[i]])

        if baseline_native_params[0]:
            p0 = np.concatenate([p0, [p2_Ns[i]]])
        if baseline_unfolded_params[0]:
            p0 = np.concatenate([p0, [p2_Us[i]]])

        if baseline_native_params[1]:
            p0 = np.concatenate([p0, [p3_Ns[i]]])
        if baseline_unfolded_params[1]:
            p0 = np.concatenate([p0, [p3_Us[i]]])

        low_bounds  = p0.copy()
        high_bounds = p0.copy()

        minX = np.min(temp_lst)
        maxX = np.max(temp_lst)

        midX = p0[0]

        t1_min = minX + 5 if model == 'Monomer' else minX + 15
        t1_max = maxX - 5 if model == 'Monomer' else maxX + 25

        if model in ['Trimer', 'Tetramer']:

            t1_min, t1_max, midX = t1_min + 20, t1_max + 30, midX + 20

            if model == 'Tetramer':
                t1_min, t1_max, midX = t1_min + 10, t1_max + 30, midX + 20

        low_bounds = [t1_min, 10] + [x / 50 if x > 0 else 50 * x for x in p0[2:]]
        high_bounds = [t1_max, 600] + [50 * x if x > 0 else x / 50 for x in p0[2:]]

        try:

            params, cov, predicted = fit_thermal_unfolding_oligomere(
                list_of_temperatures=[t],
                list_of_signals=[s],
                oligomere_concentration=oligomere_concentration,
                initial_parameters=p0,
                low_bounds=low_bounds,
                high_bounds=high_bounds,
                signal_fx=signal_fx,
                baseline_native_fx=baseline_native_fx,
                baseline_unfolded_fx=baseline_unfolded_fx,
                Cp=0)

            rel_errors = relative_errors(params, cov)

            if rel_errors[0] < 50 and rel_errors[1] < 50:
                Tms.append(params[0])
                dHs.append(params[1])

            predicted_lst.append(predicted[0])

        except:

            pass

        i += 1

    return Tms, dHs, predicted_lst


def re_arrange_predictions(predicted_lst, n_signals, n_denaturants):

    """
    Re-arrange the flattened predictions to match the original signal list with sublists
    Args:
        predicted_lst (list): Flattened list of predicted signals of length n_signals * n_denaturants
        n_signals (int): Number of signals
        n_denaturants (int): Number of denaturants
    Returns:
        list: Re-arranged list of predicted signals to be of length n_signals with sublists of length n_denaturants
    """

    data = []

    for i in range(n_signals):

        data_i = predicted_lst[i*n_denaturants:(i+1)*n_denaturants]
        data.append(data_i)

    return data

def re_arrange_params(params,n_signals):

    """
    Re arrange the flattened parameters to be a list with sublists, as many sublists  as n_signals
    Args:
        params (list): Flattened list of parameters
        n_signals (int): Number of signals
    Returns:
        list: Re-arranged list of parameters to be of length n_signals with sublists
    """

    n_params = int(len(params) / n_signals)

    params_arranged = []

    for i in range(n_signals):

        params_i = params[i*n_params:(i+1)*n_params]
        params_i_arr = np.array(params_i) # We need an array because later we will use them for fitting the signal dependence on denaturant concentration
        params_arranged.append(params_i_arr)

    return params_arranged

def subset_data(data,max_points):

    """
    Args:
        data (np.ndarray): Input data array
        max_points (int): Maximum number of points to keep
    Returns:
        np.ndarray: Subsetted data array
    """

    # Remove one every two points until the number of points is less than max_points
    do_remove = len(data) >= max_points

    while do_remove:
        data = data[::2]
        do_remove = len(data) >= max_points

    return data


def get_colors_from_numeric_values(values, min_val, max_val, use_log_scale=False):
    """
    Map numeric values to colors in the VIRIDIS palette based on a specified range.

    Parameters
    ----------
    values : list or np.ndarray
        Numeric values to map to colors.
    min_val : float
        Minimum value of the range.
    max_val : float
        Maximum value of the range.
    use_log_scale : bool, optional
        Whether to use logarithmic scaling for the values, default is True.

    Returns
    -------
    list
        List of hex color codes corresponding to the input values.
    """
    values = np.array(values)
    if use_log_scale:
        min_val = np.log10(min_val)
        max_val = np.log10(max_val)
        values = np.log10(values)
    seq = np.linspace(min_val, max_val, len(VIRIDIS))
    idx = [np.argmin(np.abs(v - seq)) for v in values]

    return [VIRIDIS[i] for i in idx]


def combine_sequences(seq1, seq2):
    """
    Combine two sequences to generate all possible combinations of their elements.

    Parameters
    ----------
    seq1 : list
        First sequence of elements.
    seq2 : list
        Second sequence of elements.

    Returns
    -------
    list
        A list of tuples, where each tuple contains one element from seq1 and one from seq2.
    """
    return list(itertools.product(seq1, seq2))