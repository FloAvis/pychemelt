"""
Main class to handle thermal denaturation data of mono- and oligomers up to tetramers
The current model assumes the proteins' unfolding is reversible
"""

import pandas as pd
import numpy as np

from itertools import chain
from copy import deepcopy

from .main import Sample

from .utils.signals import (
    map_two_state_model_to_signal_fx,
    map_three_state_model_to_signal_fx,
)

from .utils.math import (
    temperature_to_kelvin,
    temperature_to_celsius,
    relative_errors,
    constant_baseline,
    linear_baseline,
    quadratic_baseline,
    exponential_baseline,
)

from .utils.processing import (
    guess_Tm_from_derivative,
    set_param_bounds,
    adjust_value_to_interval,
    re_arrange_params,
    re_arrange_predictions,
    subset_data,
    estimate_signal_baseline_params,
    oligomer_number,
    transform_to_list
)

from .utils.fitting import (
    fit_oligomer_unfolding_single_slopes,
    fit_oligomer_unfolding_shared_slopes_many_signals,
    fit_oligomer_unfolding_many_signals,
    fit_oligomer_unfolding_three_states_single_slopes,
    fit_oligomer_unfolding_three_states_shared_slopes_many_signals,
    fit_oligomer_unfolding_three_states_many_signals,
    evaluate_fitting_and_refit,
    baseline_fx_name_to_req_params
)



class ThermalOligomer(Sample):
    """
    Class to hold the data of a DSF experiment of thermal unfolding with different concentrations of an oligomer.
    """

    def __init__(self, name='Test'):

        super().__init__(name)

        self.nr_olig = 0  # Number of oligomer concentrations in data
        self.model = None # Oligomer model type
        self.oligomeric = True # Flag for oligomer or denaturant

    def set_model(self, model_name, intermediate_name=None):

        """
        Set thermodynamic model of oligomer used for the analysis.
        Currently supported are 2 state models of monomeres, dimers, trimeres and tetrameres

        Parameters
        ----------
        model_name : str
            name of the used model. Can be: "Monomer", "Dimer", "Trimer", "Tetramer".
            Case insensitive

        Raises
        ------
        ValueError
            If the provided model name is not in the supported list.

        Notes
        -----
        This method creates/updates the following attributes on the instance:
        - self.model: oligomeric model used for analysis
        """

        allowed_models = ["monomer", "dimer", "trimer", "tetramer"]

        allowed_intermediate_models = ["monomeric", "dimeric", "trimeric"]

        allowed_combinations = ['monomer_monomeric_intermediate',
                                'dimer_monomeric_intermediate',
                                'dimer_dimeric_intermediate',
                                'trimer_monomeric_intermediate',
                                'trimer_trimeric_intermediate',
                                'tetramer_monomeric_intermediate']


        model = model_name.lower()


        if model not in allowed_models:
            raise ValueError(
            f"Invalid model '{model_name}'. "
            f"Allowed models are: {', '.join(m.capitalize() for m in allowed_models)}."
        )

        if intermediate_name is not None:

            intermediate = intermediate_name.lower()

            if intermediate not in allowed_intermediate_models:
                raise ValueError(
                f"Invalid intermediate '{intermediate_name}'. "
                f"Allowed models are: {', '.join(m.capitalize() for m in allowed_intermediate_models)}."
                )

            model = model + '_' + intermediate + '_intermediate'

            if model not in allowed_combinations:
                raise ValueError(
                f"Invalid intermediate model '{model.capitalize()}'. "
                f"Allowed models are: {', '.join(m.capitalize() for m in allowed_combinations)}."
                )



        # Save model with first letter uppercase
        self.model = model.capitalize()

        return None

    def set_concentrations(self, concentrations=None):
        """
        Set the oligomeric concentrations for the sample

        Parameters
        ----------
        concentrations : list, optional
            List of oligomer concentrations. If None, use the sample conditions

        Notes
        -----
        Creates/updates attribute `oligomer_concentrations_pre` (numpy.ndarray)
        """

        if concentrations is None:
            concentrations = self.conditions

        concentrations = transform_to_list(concentrations)

        self.oligomer_concentrations_pre = np.array(concentrations)

        return None

    def select_conditions(
            self,
            boolean_lst=None):

        """
        For each signal, select the conditions to be used for the analysis

        Parameters
        ----------
        boolean_lst : list of bool, optional
            List of booleans selecting which conditions to keep. If None, keep all.

        Notes
        -----
        Creates/updates several attributes used by downstream fitting:
        - signal_lst_multiple, temp_lst_multiple : lists of lists with selected data
        - oligomer_concentrations : list of selected oligomer concentrations
        - oligomer_concentrations_expanded : flattened numpy array matching expanded signals
        - boolean_lst, nr_olig : control flags/values
        """

        # If boolean_lst is a boolean, convert it to a list of one boolean
        boolean_lst = transform_to_list(boolean_lst)

        if boolean_lst is None:
            self.signal_lst_multiple = self.signal_lst_pre_multiple
            self.temp_lst_multiple = self.temp_lst_pre_multiple
            self.oligomer_concentrations = self.oligomer_concentrations_pre
        else:

            self.signal_lst_multiple = [None for _ in range(len(self.signal_lst_pre_multiple))]
            self.temp_lst_multiple = [None for _ in range(len(self.temp_lst_pre_multiple))]

            for i in range(len(self.signal_lst_pre_multiple)):
                self.signal_lst_multiple[i] = [x for j, x in enumerate(self.signal_lst_pre_multiple[i]) if
                                               boolean_lst[j]]
                self.temp_lst_multiple[i] = [x for j, x in enumerate(self.temp_lst_pre_multiple[i]) if boolean_lst[j]]

            self.oligomer_concentrations = [x for i, x in enumerate(self.oligomer_concentrations_pre) if
                                              boolean_lst[i]]

        self.nr_olig = len(self.oligomer_concentrations)

        # For compatibility with the plotting functions, we need to have the "nr_den" variable which is functionally the
        # same as "nr_olig":
        self.nr_den = self.nr_olig

        # Expand the number of oligomer concentrations to match the number of signals
        oligomer_concentrations = [self.oligomer_concentrations for _ in range(self.nr_signals)]

        self.oligomer_concentrations_expanded = np.concatenate(oligomer_concentrations, axis=0)

        self.boolean_lst = boolean_lst

        self.oligomer_concentrations = np.array(self.oligomer_concentrations)

        # Needed for compatibility
        self.denaturant_concentrations = self.oligomer_concentrations

        return None

    
    def guess_Cp(self):

        """
        Guess the Cp of the assembled oligomer by the number of residues.

        Raises
        ------
        ValueError
            If `self.n_residues` is not set.

        Notes
        -----
        The number of residues represent the total number of residues in the oligomer

        This method creates/updates attributes used later in fitting:
        - Cp0 assigned to self.Cp0
        """

        # If the number of residues is still zero, raise an error
        if self.n_residues == 0:
            raise ValueError('The number of residues is still zero. Please set n_residues before calling guess_Cp')

        Cp0 = self.n_residues * 0.0148 - 0.1267

        # Cp0 needs to be positive
        Cp0 = max(Cp0, 0)

        self.Cp0 = Cp0

        return None

    def estimate_baseline_parameters(
            self,
            native_baseline_type,
            unfolded_baseline_type,
            window_range_native=12,
            window_range_unfolded=12):

        """
        Estimate the baseline parameters for multiple signals of the oligomer. The native baseline represents the
        curve for the assemble doligomer while the unfolded baseline represents the curve for the unfolded and
        disassembled oligomer.

        Parameters
        ----------
        native_baseline_type : str
            one of 'constant', 'linear', 'quadratic', 'exponential'
        unfolded_baseline_type : str
            one of 'constant', 'linear', 'quadratic', 'exponential'
        window_range_native : int, optional
            Range of the window (in degrees) to estimate the baselines and slopes of the native state
        window_range_unfolded : int, optional
            Range of the window (in degrees) to estimate the baselines and slopes of the unfolded state

        Notes
        -----
        This method sets or updates these attributes:
        - bNs_per_signal, bUs_per_signal, kNs_per_signal, kUs_per_signal, qNs_per_signal, qUs_per_signal
        - poly_order_native, poly_order_unfolded
        """

        self.first_param_Ns_per_signal = []
        self.first_param_Us_per_signal = []
        self.second_param_Ns_per_signal = []
        self.second_param_Us_per_signal = []
        self.third_param_Ns_per_signal = []
        self.third_param_Us_per_signal = []

        oligomer_concentrations = np.repeat(self.oligomer_concentrations,
                                            np.array(self.signal_lst_multiple).shape[-1])
        oligomer_concentrations = np.split(oligomer_concentrations, len(self.oligomer_concentrations))

        for i in range(len(self.signal_lst_multiple)):

            self.adjusted_signal_lst_multiple = list(np.array(self.signal_lst_multiple[i])/ np.array(oligomer_concentrations))


            p1Ns, p1Us, p2Ns, p2Us, p3Ns, p3Us = estimate_signal_baseline_params(
                self.adjusted_signal_lst_multiple,
                self.temp_lst_multiple[i],
                native_baseline_type,
                unfolded_baseline_type,
                window_range_native,
                window_range_unfolded,
                oligomer_number(self.model)
            )

            self.first_param_Ns_per_signal.append(p1Ns)
            self.first_param_Us_per_signal.append(p1Us)
            self.second_param_Ns_per_signal.append(p2Ns)
            self.second_param_Us_per_signal.append(p2Us)
            self.third_param_Ns_per_signal.append(p3Ns)
            self.third_param_Us_per_signal.append(p3Us)

        baseline_fx_dic = {
            'constant': constant_baseline,
            'linear': linear_baseline,
            'quadratic': quadratic_baseline,
            'exponential': exponential_baseline
        }

        self.window_range_native = window_range_native
        self.window_range_unfolded = window_range_unfolded

        self.baseline_N_fx = baseline_fx_dic[native_baseline_type]
        self.baseline_U_fx = baseline_fx_dic[unfolded_baseline_type]

        self.native_baseline_type = native_baseline_type
        self.unfolded_baseline_type = unfolded_baseline_type

        return None

    def create_dg_df(self):

        """
        Create a dataframe of the dg values versus temperature
        """

        # Create a dataframe of the parameters
        Tm, DHm, Cp0 = self.global_fit_params[:3]

        T_c = np.arange(0, 150, 0.5)
        T = temperature_to_kelvin(T_c)
        Tm = temperature_to_kelvin(Tm)

        DG = DHm * (1 - T / Tm) + Cp0 * (T - Tm - T * np.log(T / Tm))

        dg_df = pd.DataFrame({
            'DG (kcal/mol)': DG,
            'Temperature (°C)': T_c
        })

        self.dg_df = dg_df

        return None

    def fit_thermal_unfolding_global(
            self,
            cp_limits=None,
            dh_limits=None,
            tm_limits=None,
            cp_value=None):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit all the curves at once, with global thermodynamic parameters but local slopes and local baselines)
        Multiple signals can be fitted at the same time, such as 350nm and 330nm

        Parameters
        ----------
        cp_limits : list, optional
            List of two values, the lower and upper bounds for the Cp value. If None, bounds set automatically
        dh_limits : list, optional
            List of two values, the lower and upper bounds for the dH value. If None, bounds set automatically
        tm_limits : list, optional
            List of two values, the lower and upper bounds for the Tm value. If None, bounds set automatically
        cp_value : float, optional
            If provided, the Cp value is fixed to this value, the bounds are ignored

        Notes
        -----
        This is a heavy routine that creates/updates many fitting-related attributes, including:
        - bNs_expanded, bUs_expanded, kNs_expanded, kUs_expanded, qNs_expanded, qUs_expanded
        - p0, low_bounds, high_bounds, global_fit_params, rel_errors
        - predicted_lst_multiple, params_names, params_df, dg_df
        - flags: global_fit_done, limited_tm, limited_dh, limited_cp, fixed_cp
        """

        # Requires Cp0
        if self.Cp0 <= 0:
            raise ValueError('Cp0 must be positive. Please run guess_Cp before fitting globally.')

        # Get Guess of Tm:
        
        tm_lst = []

        x1 = 6
        x2 = 11

        if not hasattr(self, "deriv_lst_multiple"):
            self.estimate_derivative()

        for i in range(len(self.signal_lst_multiple)):
            tm_lst.append(guess_Tm_from_derivative(
                self.temp_deriv_lst_multiple[i],
                self.deriv_lst_multiple[i],
                x1,
                x2
            ))

        Tm = np.average(tm_lst)

        dh = 100

        if self.model == 'Dimer':
            dh = 120
        elif self.model == 'Trimer':
            dh = 150
        elif self.model == 'Tetramer':
            dh = 180

        p0 = [Tm, dh, self.Cp0]


        params_names = [
            'Tm (°C)',
            'ΔH (kcal/mol)',
            'Cp (kcal/mol/°C)']

        self.first_param_Ns_expanded = np.concatenate(self.first_param_Ns_per_signal, axis=0)
        self.first_param_Us_expanded = np.concatenate(self.first_param_Us_per_signal, axis=0)
        self.second_param_Ns_expanded = np.concatenate(self.second_param_Ns_per_signal, axis=0)
        self.second_param_Us_expanded = np.concatenate(self.second_param_Us_per_signal, axis=0)
        self.third_param_Ns_expanded = np.concatenate(self.third_param_Ns_per_signal, axis=0)
        self.third_param_Us_expanded = np.concatenate(self.third_param_Us_per_signal, axis=0)

        p0 = np.concatenate([p0, self.first_param_Ns_expanded, self.first_param_Us_expanded])

        # We need to append as many bN and bU as the number of oligomer concentrations
        # times the number of signal types
        for signal in self.signal_names:

            params_names += (['intercept_native - ' + str(self.oligomer_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_olig)])

        for signal in self.signal_names:

            params_names += (['intercept_unfolded - ' + str(self.oligomer_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p0 = np.concatenate([p0, self.second_param_Ns_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p0 = np.concatenate([p0, self.second_param_Us_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p0 = np.concatenate([p0, self.third_param_Ns_expanded])
            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p0 = np.concatenate([p0, self.third_param_Us_expanded])

            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        low_bounds = (p0.copy())
        high_bounds = (p0.copy())

        low_bounds[3:], high_bounds[3:] = set_param_bounds(p0[3:],params_names[3:])

        # Adjusting boundaries based on the size of the oligomer
        self.limited_tm = tm_limits is not None

        if self.limited_tm:

            tm_lower, tm_upper = tm_limits

        else:

            tm_lower = p0[0] - 12 if self.model == 'Monomer' else p0[0] - 20
            tm_upper = p0[0] + 20 if self.model == 'Monomer' else p0[0] + 30

            if self.model in ['Trimer', 'Tetramer']:

                tm_lower, tm_upper, p0[0] = tm_lower + 10, tm_upper + 10, p0[0] + 10

                if self.model == 'Tetramer':
                    tm_lower, tm_upper, p0[0] = tm_lower + 10, tm_upper + 20, p0[0] + 20


        low_bounds[0] = tm_lower
        high_bounds[0] = tm_upper

        # Verify that the initial guess is within the user-defined limits
        p0[0] = adjust_value_to_interval(p0[0], tm_lower, tm_upper,1)

        self.limited_dh = dh_limits is not None

        if self.limited_dh:

            dh_lower, dh_upper = dh_limits

            p0[1] = adjust_value_to_interval(p0[1], dh_lower, dh_upper, 1)

        else:
            dh_lower = 10
            dh_upper = 500


        low_bounds[1] = dh_lower
        high_bounds[1] = dh_upper

        self.cp_value = cp_value
        self.fixed_cp = cp_value is not None

        self.limited_cp = cp_limits is not None and not self.fixed_cp

        if self.limited_cp:

            cp_lower, cp_upper = cp_limits

        else:

            cp_lower, cp_upper = 0.1, 5

        if self.fixed_cp:

            # Remove the Cp from p0, low_bounds and high_bounds
            # Remove Cp0 from the parameter names
            p0 = np.delete(p0, 2)
            low_bounds = np.delete(low_bounds, 2)
            high_bounds = np.delete(high_bounds, 2)
            params_names.pop(2)

        else:

            low_bounds[2] = cp_lower
            high_bounds[2] = cp_upper

            # Verify that the Cp initial guess is within the user-defined limits
            p0[2] = adjust_value_to_interval(p0[2], cp_lower, cp_upper, 0.5)

        # Populate the expanded signal and temperature lists
        self.expand_multiple_signal()

        signal_fx = map_two_state_model_to_signal_fx(self.model)

        kwargs = {
            'oligomer_concentrations' : self.oligomer_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds' : low_bounds,
            'high_bounds' : high_bounds,
            'cp_value' : cp_value,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx,
            'signal_fx' : signal_fx
        }

        fit_fx = fit_oligomer_unfolding_single_slopes

        # Do a quick prefit with a reduced data set
        if self.pre_fit:

            kwargs['list_of_temperatures'] = self.temp_lst_expanded_subset
            kwargs['list_of_signals'] = self.signal_lst_expanded_subset

            global_fit_params, cov, predicted = fit_fx(**kwargs)

            p0 = global_fit_params

        # Now use the whole dataset
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        kwargs['list_of_signals'] = self.signal_lst_expanded

        global_fit_params, cov, predicted = fit_fx(**kwargs)

        global_fit_params, cov, predicted, p0, low_bounds, high_bounds = evaluate_fitting_and_refit(
            global_fit_params,
            cov,
            predicted,
            high_bounds,
            low_bounds,
            p0,
            False,
            self.limited_cp,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
            fit_m_value=False,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(predicted, self.nr_signals, self.nr_olig)

        self.global_fit_done = True

        self.params_names = params_names

        self.create_params_df()
        self.create_dg_df()

        return None

    def fit_thermal_unfolding_three_state_global(
            self,
            t1_init=0,
            t2_init=0,
            dh_limits=None,
            tm_limits=None,
            CpTh=None,):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data on a three state model
        We fit all the curves at once, with global thermodynamic parameters but local slopes and local baselines)
        Multiple signals can be fitted at the same time, such as 350nm and 330nm

        Parameters
        ----------
        t1_init, t2_init : float, optional
            initial user given values of the melting temperatures of the three states, t1_init: Native to intermediate, t2_init: intermediate to unfolded
        dh_limits : list of lists, optional
            List of two lists with two values each, the lower and upper bounds for the dH values.
            If None, bounds set automatically
        tm_limits : list of lists, optional
            List of two lists with two values each, the lower and upper bounds for the Tm values.
            If None, bounds set automatically
        CpTh : float, optional
            Given estimate of the Total Cp of the system. If given, the Cp value of the transition from native to intermediate will be fitted as Cp1. If not given,
            the system assumes a total Cp of 0

        Notes
        -----
        This is a heavy routine that creates/updates many fitting-related attributes, including:
        - bNs_expanded, bUs_expanded, kNs_expanded, kUs_expanded, qNs_expanded, qUs_expanded
        - p0, low_bounds, high_bounds, global_fit_params, rel_errors
        - predicted_lst_multiple, params_names, params_df, dg_df
        - flags: global_fit_done, limited_tm, limited_dh, limited_cp, fixed_cp
        """

        # Initial parameters have to be in order:
        # Global melting temperature 1, Global enthalpy of unfolding 1,
        # Global melting temperature 2, Global enthalpy of unfolding 2,
        # Optionally Cp1
        # Single intercepts folded, Single slopes folded,
        # Single intercepts unfolded, Single slopes unfolded


        if not hasattr(self, "deriv_lst_multiple"):
            self.estimate_derivative()

        if CpTh is not None:
            if CpTh <= 0.1:
                raise ValueError('CpTh must be large enough for fitting. If you do not wish to fit the Cp values, omit this parameter.')

            # Parameters for T1, T2, will be added later via gridsearch
            p0 = [0, 250, 0, 250, self.Cp0]

        else:
            p0 = [0, 250, 0, 250, 0]


        params_names = [
            'Tm1 (°C)',
            'ΔH1 (kcal/mol)',
            'Tm2 (°C)',
            'ΔH2 (kcal/mol)',
            'Cp1 (kcal/mol/°C)']

        self.first_param_Ns_expanded = np.concatenate(self.first_param_Ns_per_signal, axis=0)
        self.first_param_Us_expanded = np.concatenate(self.first_param_Us_per_signal, axis=0)
        self.second_param_Ns_expanded = np.concatenate(self.second_param_Ns_per_signal, axis=0)
        self.second_param_Us_expanded = np.concatenate(self.second_param_Us_per_signal, axis=0)
        self.third_param_Ns_expanded = np.concatenate(self.third_param_Ns_per_signal, axis=0)
        self.third_param_Us_expanded = np.concatenate(self.third_param_Us_per_signal, axis=0)

        #Estimating intermediate intercept

        self.intercept_intermediate = (self.first_param_Ns_expanded + self.first_param_Us_expanded) / 2

        p0 = np.concatenate([p0, self.first_param_Ns_expanded, self.first_param_Us_expanded, self.intercept_intermediate])

        # We need to append as many bN and bU as the number of oligomer concentrations
        # times the number of signal types
        for signal in self.signal_names:
            params_names += (['intercept_native - ' + str(self.oligomer_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_olig)])

        for signal in self.signal_names:
            params_names += (['intercept_unfolded - ' + str(self.oligomer_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_olig)])

        for signal in self.signal_names:
            params_names += (['intercept_intermediate - ' + str(self.oligomer_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.native_baseline_type in ['linear', 'quadratic', 'exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p0 = np.concatenate([p0, self.second_param_Ns_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.unfolded_baseline_type in ['linear', 'quadratic', 'exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p0 = np.concatenate([p0, self.second_param_Us_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p0 = np.concatenate([p0, self.third_param_Ns_expanded])
            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p0 = np.concatenate([p0, self.third_param_Us_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomer_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_olig)])

        low_bounds = (p0.copy())
        high_bounds = (p0.copy())

        low_bounds[5:], high_bounds[5:] = set_param_bounds(p0[5:], params_names[5:])

        self.limited_tm = tm_limits is not None

        if self.limited_tm:

            tm1_lower, tm1_upper, tm2_lower, tm2_upper = np.array(tm_limits).flatten().tolist()

        else:

            tm1_lower = 15
            tm1_upper = self.user_max_temp + 20

            tm2_lower = 15
            tm2_upper = self.user_max_temp + 20

        low_bounds[0] = tm1_lower
        high_bounds[0] = tm1_upper

        low_bounds[2] = tm2_lower
        high_bounds[2] = tm2_upper

        # Verify that the initial guesses are within the user-defined limits
        p0[0] = adjust_value_to_interval(p0[0], tm1_lower, tm1_upper, 1)
        p0[2] = adjust_value_to_interval(p0[2], tm2_lower, tm2_upper, 1)

        self.limited_dh = dh_limits is not None

        if self.limited_dh:

            dh1_lower, dh1_upper, dh2_lower, dh2_upper, = np.array(dh_limits).flatten().tolist()

            p0[1] = adjust_value_to_interval(p0[1], dh1_lower, dh1_upper, 1)
            p0[3] = adjust_value_to_interval(p0[3], dh2_lower, dh2_upper, 1)

        else:
            dh1_lower = 10
            dh1_upper = 500

            dh2_lower = 10
            dh2_upper = 500

        low_bounds[1] = dh1_lower
        high_bounds[1] = dh1_upper

        low_bounds[3] = dh2_lower
        high_bounds[3] = dh2_upper

        self.cp_value = CpTh
        self.fixed_cp = CpTh is not None


        if not self.fixed_cp:

            # Remove the Cp from p0, low_bounds and high_bounds
            # Remove Cp0 from the parameter names
            p0 = np.delete(p0, 4)
            low_bounds = np.delete(low_bounds, 4)
            high_bounds = np.delete(high_bounds, 4)
            params_names.pop(4)

        else:

            cp_lower, cp_upper = 0.1, CpTh

            low_bounds[4] = cp_lower
            high_bounds[4] = cp_upper

            # Verify that the Cp initial guess is within the user-defined limits
            p0[4] = adjust_value_to_interval(p0[4], cp_lower, cp_upper, 0.5)


        # Populate the expanded signal and temperature lists
        self.expand_multiple_signal()

        signal_fx = map_three_state_model_to_signal_fx(self.model)

        if t1_init != 0:
            p0[0], low_bounds[0], high_bounds[0] = t1_init, np.max([t1_init - 20, 0]), t1_init + 20

        if t2_init != 0:
            p0[2], low_bounds[2], high_bounds[2] = t2_init, np.max([t2_init - 20, 0]), t2_init + 20

        kwargs = {
            'oligomer_concentrations': self.oligomer_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'baseline_native_fx': self.baseline_N_fx,
            'baseline_unfolded_fx': self.baseline_U_fx,
            'signal_fx': signal_fx,
            'CpTh_value': CpTh,
        }

        fit_fx = fit_oligomer_unfolding_three_states_single_slopes

        step = 6
        num_rows = len(self.oligomer_concentrations)


        if num_rows > 3:
            step += 2
        if num_rows > 4:
            step += 2

        if t1_init == 0 or t2_init == 0:

            if self.limited_tm:
                test_T1s = np.arange(np.max([tm1_lower, 20]), tm1_upper, step)
                test_T2s = np.arange(np.max([tm2_lower, 20]) + step, tm2_upper, step)

            else:
                test_T1s = np.arange(np.max([self.global_min_temp + 10, 20]), self.global_max_temp - 25, step)
                test_T2s = np.arange(np.max([self.global_min_temp + 10, 20]) + step, self.global_max_temp + 5, step)

            if t1_init != 0:
                test_T1s = np.array([t1_init])
            elif t2_init != 0:
                test_T2s = np.array([t2_init])

            combinations = [(t1, t2) for t1 in test_T1s for t2 in test_T2s]
            combinations = [(t1, t2) for t1, t2 in combinations if t1 < t2]

            df = pd.DataFrame(combinations, columns=['t1', 't2'])

            df_tm = pd.DataFrame(np.zeros_like(combinations), columns=['t1', 't2'], dtype=float)
            df_dh = pd.DataFrame(np.zeros_like(combinations), columns=['dh1', 'dh2'], dtype=float)

            rss_all = []

            #Using a subset for fitting
            kwargs['list_of_temperatures'] = self.temp_lst_expanded_subset
            kwargs['list_of_signals'] = self.signal_lst_expanded_subset


            for index, row in df.iterrows():
                kwargs['t1'] = row['t1']
                kwargs['t2'] = row['t2']

                try:
                    fit_params, cov, pred = fit_oligomer_unfolding_three_states_single_slopes(**kwargs)
                except Exception as e:
                    print(f"Warning: {e}")
                    continue

                #using the fitted parameters as a base for fitting
                df_tm.iloc[index, 0] = fit_params[0]
                df_tm.iloc[index, 1] = fit_params[2]

                df_dh.iloc[index, 0] = fit_params[1]
                df_dh.iloc[index, 1] = fit_params[3]


                rss = np.nansum((np.array(pred) - np.array(self.signal_lst_expanded_subset)) ** 2)
                rss_all.append(rss)

            idx = np.argmin(rss_all)

            t1_init, t2_init = df_tm['t1'][idx], df_tm['t2'][idx]
            p0[0], p0[2] = t1_init, t2_init

            dh1_init, dh2_init = df_dh['dh1'][idx], df_dh['dh2'][idx]
            p0[1], p0[3] = dh1_init, dh2_init

            low_bounds[0], low_bounds[2] = t1_init - 30, t2_init - 30
            high_bounds[0], high_bounds[2] = t1_init + 30, t2_init + 30

            kwargs['initial_parameters'] = p0
            kwargs['low_bounds'] = low_bounds
            kwargs['high_bounds'] = high_bounds
            kwargs['t1'] = None
            kwargs['t2'] = None


        # Now use the whole dataset
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        kwargs['list_of_signals'] = self.signal_lst_expanded

        global_fit_params, cov, predicted = fit_fx(**kwargs)

        global_fit_params, cov, predicted, p0, low_bounds, high_bounds = evaluate_fitting_and_refit(
            global_fit_params,
            cov,
            predicted,
            high_bounds,
            low_bounds,
            p0,
            False,
            False,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
            fit_m_value=False,
            three_state_model=True,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(predicted, self.nr_signals, self.nr_olig)

        self.global_fit_done = True

        self.params_names = params_names

        self.create_params_df_three_state()
        self.create_dg_df()

        return None

    def fit_thermal_unfolding_global_global(self):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit all the curves at once, with global thermodynamic parameters and global slopes (but local baselines)
        Multiple refers to the fact that we fit many signals at the same time, such as 350nm and 330nm
        Must be run after fit_thermal_unfolding_global_multiple

        Notes
        -----
        Updates global fitting attributes and sets `global_global_fit_done` when complete.
        """

        # Requires global fit done
        if not self.global_fit_done:
            self.fit_thermal_unfolding_global()

        if self.signal_ids is None:
            self.set_signal_id()

        param_init = 2 + (self.cp_value is None)

        p0 = self.global_fit_params[:param_init]
        low_bounds = self.low_bounds[:param_init]
        high_bounds = self.high_bounds[:param_init]

        n_datasets = self.nr_olig * self.nr_signals

        p1Ns = self.global_fit_params[param_init:param_init + n_datasets]
        p1Us = self.global_fit_params[param_init + n_datasets:param_init + 2 * n_datasets]

        low_bounds_p1Ns = self.low_bounds[param_init:param_init + n_datasets]
        low_bounds_p1Us = self.low_bounds[param_init + n_datasets:param_init + 2 * n_datasets]

        high_bounds_p1Ns = self.high_bounds[param_init:param_init + n_datasets]
        high_bounds_p1Us = self.high_bounds[param_init + n_datasets:param_init + 2 * n_datasets]

        id_start = param_init + 2 * n_datasets

        params_names = self.params_names[:id_start]

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p2Ns = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p2Ns = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p2Ns = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p2Us = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p2Us = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p2Us = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p3Ns = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p3Ns = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p3Ns = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p3Us = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p3Us = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p3Us = self.high_bounds[id_start:id_start + n_datasets]

        p0 = np.concatenate([p0, p1Ns, p1Us])
        low_bounds = np.concatenate([low_bounds, low_bounds_p1Ns, low_bounds_p1Us])
        high_bounds = np.concatenate([high_bounds, high_bounds_p1Ns, high_bounds_p1Us])

        # Baselines are still independent for each signal and oligomer concentration
        # Slopes and quadratic terms are shared - per signal only

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            p2Ns = re_arrange_params(p2Ns, self.nr_signals)
            low_bounds_p2Ns = re_arrange_params(low_bounds_p2Ns, self.nr_signals)
            high_bounds_p2Ns = re_arrange_params(high_bounds_p2Ns, self.nr_signals)

            for kNs_i, low_bounds_kNs_i, high_bounds_kNs_i in zip(p2Ns, low_bounds_p2Ns, high_bounds_p2Ns):
                p0 = np.append(p0, np.median(kNs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_kNs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_kNs_i))

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            p2Us = re_arrange_params(p2Us, self.nr_signals)
            low_bounds_p2Us = re_arrange_params(low_bounds_p2Us, self.nr_signals)
            high_bounds_p2Us = re_arrange_params(high_bounds_p2Us, self.nr_signals)

            for kUs_i, low_bounds_kUs_i, high_bounds_kUs_i in zip(p2Us, low_bounds_p2Us, high_bounds_p2Us):
                p0 = np.append(p0, np.median(kUs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_kUs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_kUs_i))

        if self.native_baseline_type in ['quadratic', 'exponential']:

            p3Ns = re_arrange_params(p3Ns, self.nr_signals)
            low_bounds_p3Ns = re_arrange_params(low_bounds_p3Ns, self.nr_signals)
            high_bounds_p3Ns = re_arrange_params(high_bounds_p3Ns, self.nr_signals)

            for qNs_i, low_bounds_qNs_i, high_bounds_qNs_i in zip(p3Ns, low_bounds_p3Ns, high_bounds_p3Ns):
                p0 = np.append(p0, np.median(qNs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_qNs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_qNs_i))

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            p3Us = re_arrange_params(p3Us, self.nr_signals)
            low_bounds_p3Us = re_arrange_params(low_bounds_p3Us, self.nr_signals)
            high_bounds_p3Us = re_arrange_params(high_bounds_p3Us, self.nr_signals)

            for qUs_i, low_bounds_qUs_i, high_bounds_qUs_i in zip(p3Us, low_bounds_p3Us, high_bounds_p3Us):
                p0 = np.append(p0, np.median(qUs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_qUs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_qUs_i))

        signal_fx = map_two_state_model_to_signal_fx(self.model)

        kwargs = {

            'oligomer_concentrations': self.oligomer_concentrations_expanded,
            'list_of_temperatures': self.temp_lst_expanded_subset,
            'list_of_signals': self.signal_lst_expanded_subset,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'cp_value': self.cp_value,
            'signal_ids':self.signal_ids,
            'baseline_native_fx': self.baseline_N_fx,
            'baseline_unfolded_fx': self.baseline_U_fx,
            'signal_fx' : signal_fx,
        }

        fit_fx = fit_oligomer_unfolding_shared_slopes_many_signals

        if self.pre_fit:
            # Do a pre-fit with a reduced data set
            global_fit_params, cov, predicted = fit_fx(**kwargs)

            p0 = global_fit_params
        # End of pre-fit

        # Use whole dataset
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        kwargs['list_of_signals'] = self.signal_lst_expanded

        global_fit_params, cov, predicted = fit_fx(**kwargs)

        global_fit_params, cov, predicted, p0, low_bounds, high_bounds = evaluate_fitting_and_refit(
            global_fit_params,
            cov,
            predicted,
            high_bounds,
            low_bounds,
            p0,
            False,
            self.limited_cp,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
            fit_m_value=False,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(
            predicted, self.nr_signals, self.nr_olig)

        self.params_names = params_names

        self.create_params_df()
        self.create_dg_df()

        self.global_global_fit_done = True

        return None

    def fit_thermal_unfolding_three_state_global_global(self):

        """
        Fit the thermal unfolding with three states of the sample using the signal and temperature data
        We fit all the curves at once, with global thermodynamic parameters and global slopes (but local baselines)
        Multiple refers to the fact that we fit many signals at the same time, such as 350nm and 330nm
        Must be run after fit_thermal_unfolding_global

        Notes
        -----
        Updates global fitting attributes and sets `global_global_fit_done` when complete.
        """

        # Requires global fit done
        if not self.global_fit_done:
            self.fit_thermal_unfolding_three_state_global()

        if self.signal_ids is None:
            self.set_signal_id()

        param_init = 4

        if self.fixed_cp:
            param_init += 1

        p0 = self.global_fit_params[:param_init]
        low_bounds = self.low_bounds[:param_init]
        high_bounds = self.high_bounds[:param_init]

        n_datasets = self.nr_olig * self.nr_signals

        p1Ns = self.global_fit_params[param_init:param_init + n_datasets]
        p1Us = self.global_fit_params[param_init + n_datasets:param_init + 2 * n_datasets]
        intermediate_baseline = self.global_fit_params[param_init + 2* n_datasets:param_init + 3 * n_datasets]

        low_bounds_p1Ns = self.low_bounds[param_init:param_init + n_datasets]
        low_bounds_p1Us = self.low_bounds[param_init + n_datasets:param_init + 2 * n_datasets]
        low_bounds_intermediate_baseline = self.low_bounds[param_init + 2 * n_datasets:param_init + 3 * n_datasets]

        high_bounds_p1Ns = self.high_bounds[param_init:param_init + n_datasets]
        high_bounds_p1Us = self.high_bounds[param_init + n_datasets:param_init + 2 * n_datasets]
        high_bounds_intermediate_baseline = self.high_bounds[param_init + 2 * n_datasets:param_init + 3 * n_datasets]

        id_start = param_init + 3 * n_datasets

        params_names = self.params_names[:id_start]

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p2Ns = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p2Ns = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p2Ns = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p2Us = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p2Us = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p2Us = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p3Ns = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p3Ns = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p3Ns = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p3Us = self.global_fit_params[id_start:id_start + n_datasets]
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]
            low_bounds_p3Us = self.low_bounds[id_start:id_start + n_datasets]
            high_bounds_p3Us = self.high_bounds[id_start:id_start + n_datasets]
            id_start += n_datasets

        p0 = np.concatenate([p0, p1Ns, p1Us, intermediate_baseline])
        low_bounds = np.concatenate([low_bounds, low_bounds_p1Ns, low_bounds_p1Us, low_bounds_intermediate_baseline])
        high_bounds = np.concatenate([high_bounds, high_bounds_p1Ns, high_bounds_p1Us, high_bounds_intermediate_baseline])

        # Baselines are still independent for each signal and oligomer concentration
        # Slopes and quadratic terms are shared - per signal only

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            p2Ns = re_arrange_params(p2Ns, self.nr_signals)
            low_bounds_p2Ns = re_arrange_params(low_bounds_p2Ns, self.nr_signals)
            high_bounds_p2Ns = re_arrange_params(high_bounds_p2Ns, self.nr_signals)

            for kNs_i, low_bounds_kNs_i, high_bounds_kNs_i in zip(p2Ns, low_bounds_p2Ns, high_bounds_p2Ns):
                p0 = np.append(p0, np.median(kNs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_kNs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_kNs_i))

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            p2Us = re_arrange_params(p2Us, self.nr_signals)
            low_bounds_p2Us = re_arrange_params(low_bounds_p2Us, self.nr_signals)
            high_bounds_p2Us = re_arrange_params(high_bounds_p2Us, self.nr_signals)

            for kUs_i, low_bounds_kUs_i, high_bounds_kUs_i in zip(p2Us, low_bounds_p2Us, high_bounds_p2Us):
                p0 = np.append(p0, np.median(kUs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_kUs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_kUs_i))

        if self.native_baseline_type in ['quadratic', 'exponential']:

            p3Ns = re_arrange_params(p3Ns, self.nr_signals)
            low_bounds_p3Ns = re_arrange_params(low_bounds_p3Ns, self.nr_signals)
            high_bounds_p3Ns = re_arrange_params(high_bounds_p3Ns, self.nr_signals)

            for qNs_i, low_bounds_qNs_i, high_bounds_qNs_i in zip(p3Ns, low_bounds_p3Ns, high_bounds_p3Ns):
                p0 = np.append(p0, np.median(qNs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_qNs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_qNs_i))

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            p3Us = re_arrange_params(p3Us, self.nr_signals)
            low_bounds_p3Us = re_arrange_params(low_bounds_p3Us, self.nr_signals)
            high_bounds_p3Us = re_arrange_params(high_bounds_p3Us, self.nr_signals)

            for qUs_i, low_bounds_qUs_i, high_bounds_qUs_i in zip(p3Us, low_bounds_p3Us, high_bounds_p3Us):
                p0 = np.append(p0, np.median(qUs_i))
                low_bounds = np.append(low_bounds, np.min(low_bounds_qUs_i))
                high_bounds = np.append(high_bounds, np.max(high_bounds_qUs_i))

        signal_fx = map_three_state_model_to_signal_fx(self.model)

        kwargs = {

            'oligomer_concentrations': self.oligomer_concentrations_expanded,
            'list_of_temperatures': self.temp_lst_expanded_subset,
            'list_of_signals': self.signal_lst_expanded_subset,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'signal_ids':self.signal_ids,
            'baseline_native_fx': self.baseline_N_fx,
            'baseline_unfolded_fx': self.baseline_U_fx,
            'signal_fx' : signal_fx,
            'CpTh_value' : self.cp_value,
        }

        fit_fx = fit_oligomer_unfolding_three_states_shared_slopes_many_signals

        if self.pre_fit:
            # Do a pre-fit with a reduced data set
            global_fit_params, cov, predicted = fit_fx(**kwargs)

            p0 = global_fit_params
        # End of pre-fit

        # Use whole dataset
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        kwargs['list_of_signals'] = self.signal_lst_expanded

        global_fit_params, cov, predicted = fit_fx(**kwargs)

        global_fit_params, cov, predicted, p0, low_bounds, high_bounds = evaluate_fitting_and_refit(
            global_fit_params,
            cov,
            predicted,
            high_bounds,
            low_bounds,
            p0,
            False,
            False,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
            fit_m_value=False,
            three_state_model=True,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(
            predicted, self.nr_signals, self.nr_olig)

        self.params_names = params_names

        self.create_params_df_three_state()
        self.create_dg_df()

        self.global_global_fit_done = True

        return None

    def fit_thermal_unfolding_global_global_global(
            self,
            model_scale_factor=True):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit all the curves at once, with global thermodynamic parameters, global slopes and global baselines
        Must be run after fit_thermal_unfolding_global_global

        Parameters
        ----------
        model_scale_factor : bool, optional
            If True, model a scale factor for each oligomer concentration

        Notes
        -----
        Updates many global fitting attributes and sets `global_global_global_fit_done` when complete. If
        `model_scale_factor` is True the method also creates scaled signal attributes:
        - signal_lst_multiple_scaled, predicted_lst_multiple_scaled
        """

        # Requires global global fit done
        if not self.global_global_fit_done:
            self.fit_thermal_unfolding_global_global()

        param_init = 2 + (self.cp_value is None)

        params_names = self.params_names[:param_init]

        p0 = self.global_fit_params[:param_init]
        low_bounds = self.low_bounds[:param_init]
        high_bounds = self.high_bounds[:param_init]

        n_datasets = self.nr_olig * self.nr_signals

        p1Ns = self.global_fit_params[param_init:param_init + n_datasets]
        p1Us = self.global_fit_params[param_init + n_datasets:param_init + 2 * n_datasets]

        p1Ns_per_signal = re_arrange_params(p1Ns, self.nr_signals)
        p1Us_per_signal = re_arrange_params(p1Us, self.nr_signals)

        p1Ns = np.mean(p1Ns_per_signal, axis=1)
        p1Us = np.mean(p1Us_per_signal, axis=1)

        p1Ns_low_bounds = [p1N / 100 if p1N > 0 else 100 * p1N for p1N in p1Ns]
        p1Us_low_bounds = [p1U / 100 if p1U > 0 else 100 * p1U for p1U in p1Us]

        p1Ns_high_bounds = [p1N * 100 if p1N > 0 else p1N / 100 for p1N in p1Ns]
        p1Us_high_bounds = [p1U * 100 if p1U > 0 else p1U / 100 for p1U in p1Us]

        idx = param_init + 2 * n_datasets

        params_names += ['intercept_native - ' + signal_name for signal_name in self.signal_names]
        params_names += ['intercept_unfolded - ' + signal_name for signal_name in self.signal_names]

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            kNs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_kNs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_kNs = self.high_bounds[idx:idx + self.nr_signals]

            idx += self.nr_signals
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            kUs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_kUs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_kUs = self.high_bounds[idx:idx + self.nr_signals]

            idx += self.nr_signals
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            qNs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_qNs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_qNs = self.high_bounds[idx:idx + self.nr_signals]
            idx += self.nr_signals

            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            qUs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_qUs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_qUs = self.high_bounds[idx:idx + self.nr_signals]
            idx += self.nr_signals

            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        p0 = np.concatenate([p0, p1Ns, p1Us])
        low_bounds = np.concatenate([low_bounds, p1Ns_low_bounds, p1Us_low_bounds])
        high_bounds = np.concatenate([high_bounds, p1Ns_high_bounds, p1Us_high_bounds])

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            p0 = np.concatenate([p0, kNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kNs])

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            p0 = np.concatenate([p0, kUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kUs])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            p0 = np.concatenate([p0, qNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qNs])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            p0 = np.concatenate([p0, qUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qUs])

        # If required, include a scale factor for each oligomer concentration
        if model_scale_factor:
            # The last oligomer concentration is fixed to 1, the rest are fitted
            scale_factors = [1 for _ in range(self.nr_olig - 1)]
            scale_factors_low = [0.5882 for _ in range(self.nr_olig - 1)]
            scale_factors_high = [1.7 for _ in range(self.nr_olig - 1)]

            p0 = np.concatenate([p0, scale_factors])
            low_bounds = np.concatenate([low_bounds, scale_factors_low])
            high_bounds = np.concatenate([high_bounds, scale_factors_high])

            params_names += ['Scale factor - ' + str(d) + ' (M). ID: ' + str(i) for
                             i, d in enumerate(self.oligomer_concentrations)]

            params_names.pop()  # Remove the last one, as it is fixed to 1

        scale_factor_exclude_ids = [self.nr_olig - 1] if model_scale_factor else []

        signal_fx = map_two_state_model_to_signal_fx(self.model)

        # Do a prefit with a reduced dataset
        kwargs = {

            'list_of_temperatures' : self.temp_lst_expanded_subset,
            'list_of_signals' : self.signal_lst_expanded_subset,
            'signal_ids' : self.signal_ids,
            'oligomer_concentrations': self.oligomer_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'model_scale_factor':model_scale_factor,
            'cp_value' : self.cp_value,
            'scale_factor_exclude_ids':scale_factor_exclude_ids,
            'signal_fx' : signal_fx,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx
        }

        fit_fx = fit_oligomer_unfolding_many_signals

        if self.pre_fit:

            global_fit_params, cov, predicted = fit_fx(**kwargs)

            # Assign the fitted parameters to the initial guess for the full dataset
            p0 = global_fit_params

            # End of prefit with reduced dataset

        # Use the whole dataset
        kwargs['list_of_signals'] = self.signal_lst_expanded
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        global_fit_params, cov, predicted = fit_fx(**kwargs)

        # Remove scale factors that are not significant
        if model_scale_factor:

            # 2 parameters corresponding to Tm, dH
            # plus Cp if fitted
            idx_start = 2 + (self.cp_value is None)

            native_factor   = 1+np.sum(baseline_fx_name_to_req_params(self.baseline_N_fx))
            unfolded_factor = 1+np.sum(baseline_fx_name_to_req_params(self.baseline_U_fx))

            # Add index according to the native baseline polynomial order
            idx_start += native_factor * self.nr_signals
            # Add index according to the unfolded baseline polynomial order
            idx_start += unfolded_factor * self.nr_signals

            for _ in range(5):

                # Sort in ascending order the IDs to exclude
                scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

                n_fixed_factors = len(scale_factor_exclude_ids)
                n_fit_factors = self.nr_olig - n_fixed_factors

                if n_fit_factors == 0:
                    break

                sf_params = global_fit_params[idx_start:(idx_start + n_fit_factors)]

                idxs_to_remove = []
                re_fit = False

                # Add dummy variable where we need to skip the index
                for id in scale_factor_exclude_ids:
                    sf_params = np.insert(sf_params, id, np.nan)

                for i, sf in enumerate(sf_params):

                    if i in scale_factor_exclude_ids:
                        continue

                    if 0.995 <= sf <= 1.015:
                        # Exclude the scale factor from the fit
                        scale_factor_exclude_ids.append(i)
                        re_fit = True

                        j1 = np.sum(np.array(scale_factor_exclude_ids) < i)
                        j2 = len(idxs_to_remove)

                        idxs_to_remove.append(idx_start + i - j1 + j2)

                if not re_fit:
                    break

                else:

                    for idx in reversed(idxs_to_remove):

                        global_fit_params = np.delete(global_fit_params, idx)
                        low_bounds = np.delete(low_bounds, idx)
                        high_bounds = np.delete(high_bounds, idx)

                        del params_names[idx]

                    kwargs['initial_parameters'] = global_fit_params
                    kwargs['low_bounds'] = low_bounds
                    kwargs['high_bounds'] = high_bounds
                    kwargs['scale_factor_exclude_ids'] = scale_factor_exclude_ids

                    global_fit_params, cov, predicted = fit_fx(**kwargs)

        rel_errors = relative_errors(global_fit_params, cov)

        self.params_names = params_names
        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(
            predicted, self.nr_signals, self.nr_olig)

        self.create_params_df()
        self.create_dg_df()

        self.global_global_global_fit_done = True

        # Obtained the scaled signal too
        if model_scale_factor:

            # signal scaled hos one sublist per selected signal type
            signal_scaled = deepcopy(self.signal_lst_multiple)
            predicted_scaled = deepcopy(self.predicted_lst_multiple)

            for value, param in zip(self.global_fit_params, self.params_names):

                if 'Scale factor' in param:

                    id = int(param.split('(M). ID: ')[-1])

                    for i in range(len(signal_scaled)):
                        signal_scaled[i][id] /= value
                        predicted_scaled[i][id] /= value

            self.signal_lst_multiple_scaled = signal_scaled
            self.predicted_lst_multiple_scaled = predicted_scaled

        return None

    def fit_thermal_unfolding_three_state_global_global_global(
            self,
            model_scale_factor=True):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data for models assuming three states
        We fit all the curves at once, with global thermodynamic parameters, global slopes and global baselines
        Must be run after fit_thermal_unfolding_global_global

        Parameters
        ----------
        model_scale_factor : bool, optional
            If True, model a scale factor for each oligomer concentration

        Notes
        -----
        Updates many global fitting attributes and sets `global_global_global_fit_done` when complete. If
        `model_scale_factor` is True the method also creates scaled signal attributes:
        - signal_lst_multiple_scaled, predicted_lst_multiple_scaled
        """

        # Requires global global fit done
        if not self.global_global_fit_done:
            self.fit_thermal_unfolding_three_state_global_global()

        param_init = 4

        if self.fixed_cp:
            param_init += 1


        params_names = self.params_names[:param_init]

        p0 = self.global_fit_params[:param_init]
        low_bounds = self.low_bounds[:param_init]
        high_bounds = self.high_bounds[:param_init]

        n_datasets = self.nr_olig * self.nr_signals

        p1Ns = self.global_fit_params[param_init:param_init + n_datasets]
        p1Us = self.global_fit_params[param_init + n_datasets:param_init + 2 * n_datasets]
        intermediate_baselines = self.global_fit_params[param_init + 2 * n_datasets:param_init + 3 * n_datasets]

        p1Ns_per_signal = re_arrange_params(p1Ns, self.nr_signals)
        p1Us_per_signal = re_arrange_params(p1Us, self.nr_signals)
        intermediate_baselines_per_signal = re_arrange_params(intermediate_baselines, self.nr_signals)

        p1Ns = np.mean(p1Ns_per_signal, axis=1)
        p1Us = np.mean(p1Us_per_signal, axis=1)
        intermediate_baselines = np.mean(intermediate_baselines_per_signal, axis=1)

        p1Ns_low_bounds = [p1N / 100 if p1N > 0 else 100 * p1N for p1N in p1Ns]
        p1Us_low_bounds = [p1U / 100 if p1U > 0 else 100 * p1U for p1U in p1Us]
        intermediate_baselines_low_bounds = [intermediate_baseline / 100 if intermediate_baseline > 0 else 100 * intermediate_baseline for intermediate_baseline in intermediate_baselines]

        p1Ns_high_bounds = [p1N * 100 if p1N > 0 else p1N / 100 for p1N in p1Ns]
        p1Us_high_bounds = [p1U * 100 if p1U > 0 else p1U / 100 for p1U in p1Us]
        intermediate_baselines_high_bounds = [
            intermediate_baseline * 100 if intermediate_baseline > 0 else intermediate_baseline / 100 for
            intermediate_baseline in intermediate_baselines]

        idx = param_init + 3 * n_datasets

        params_names += ['intercept_native - ' + signal_name for signal_name in self.signal_names]
        params_names += ['intercept_unfolded - ' + signal_name for signal_name in self.signal_names]
        params_names += ['intercept_intermediate - ' + signal_name for signal_name in self.signal_names]

        if self.native_baseline_type in ['linear', 'quadratic', 'exponential']:
            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            kNs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_kNs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_kNs = self.high_bounds[idx:idx + self.nr_signals]

            idx += self.nr_signals
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.unfolded_baseline_type in ['linear', 'quadratic', 'exponential']:
            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            kUs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_kUs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_kUs = self.high_bounds[idx:idx + self.nr_signals]

            idx += self.nr_signals
            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.native_baseline_type in ['quadratic', 'exponential']:
            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            qNs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_qNs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_qNs = self.high_bounds[idx:idx + self.nr_signals]
            idx += self.nr_signals

            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:
            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            qUs = self.global_fit_params[idx:idx + self.nr_signals]
            low_bounds_qUs = self.low_bounds[idx:idx + self.nr_signals]
            high_bounds_qUs = self.high_bounds[idx:idx + self.nr_signals]
            idx += self.nr_signals

            params_names += [param_name + ' - ' + signal_name for signal_name in self.signal_names]

        p0 = np.concatenate([p0, p1Ns, p1Us, intermediate_baselines])
        low_bounds = np.concatenate([low_bounds, p1Ns_low_bounds, p1Us_low_bounds, intermediate_baselines_low_bounds])
        high_bounds = np.concatenate([high_bounds, p1Ns_high_bounds, p1Us_high_bounds, intermediate_baselines_high_bounds])

        if self.native_baseline_type in ['linear', 'quadratic', 'exponential']:
            p0 = np.concatenate([p0, kNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kNs])

        if self.unfolded_baseline_type in ['linear', 'quadratic', 'exponential']:
            p0 = np.concatenate([p0, kUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kUs])

        if self.native_baseline_type in ['quadratic', 'exponential']:
            p0 = np.concatenate([p0, qNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qNs])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:
            p0 = np.concatenate([p0, qUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qUs])

        # If required, include a scale factor for each oligomer concentration
        if model_scale_factor:
            # The last oligomer concentration is fixed to 1, the rest are fitted
            scale_factors = [1 for _ in range(self.nr_olig - 1)]
            scale_factors_low = [0.5882 for _ in range(self.nr_olig - 1)]
            scale_factors_high = [1.7 for _ in range(self.nr_olig - 1)]

            p0 = np.concatenate([p0, scale_factors])
            low_bounds = np.concatenate([low_bounds, scale_factors_low])
            high_bounds = np.concatenate([high_bounds, scale_factors_high])

            params_names += ['Scale factor - ' + str(d) + ' (M). ID: ' + str(i) for
                             i, d in enumerate(self.oligomer_concentrations)]

            params_names.pop()  # Remove the last one, as it is fixed to 1

        scale_factor_exclude_ids = [self.nr_olig - 1] if model_scale_factor else []

        signal_fx = map_three_state_model_to_signal_fx(self.model)

        # Do a prefit with a reduced dataset
        kwargs = {
            'list_of_temperatures' : self.temp_lst_expanded_subset,
            'list_of_signals' : self.signal_lst_expanded_subset,
            'signal_ids' : self.signal_ids,
            'oligomer_concentrations': self.oligomer_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'model_scale_factor':model_scale_factor,
            'scale_factor_exclude_ids':scale_factor_exclude_ids,
            'signal_fx' : signal_fx,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx,
            'CpTh_value' : self.cp_value,
        }

        fit_fx = fit_oligomer_unfolding_three_states_many_signals

        if self.pre_fit:

            global_fit_params, cov, predicted = fit_fx(**kwargs)

            # Assign the fitted parameters to the initial guess for the full dataset
            p0 = global_fit_params

            # End of prefit with reduced dataset

        # Use the whole dataset
        kwargs['list_of_signals'] = self.signal_lst_expanded
        kwargs['list_of_temperatures'] = self.temp_lst_expanded

        global_fit_params, cov, predicted = fit_fx(**kwargs)

        # Remove scale factors that are not significant
        if model_scale_factor:

            # 4 parameters corresponding to Tm, dH
            idx_start = 4

            # Additional parameter if CpTh is given
            if self.fixed_cp:
                idx_start += 1

            native_factor   = 1+np.sum(baseline_fx_name_to_req_params(self.baseline_N_fx))
            unfolded_factor = 1+np.sum(baseline_fx_name_to_req_params(self.baseline_U_fx))

            # Add index according to the native baseline polynomial order
            idx_start += native_factor * self.nr_signals
            # Add index according to the unfolded baseline polynomial order
            idx_start += unfolded_factor * self.nr_signals

            for _ in range(5):

                # Sort in ascending order the IDs to exclude
                scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

                n_fixed_factors = len(scale_factor_exclude_ids)
                n_fit_factors = self.nr_olig - n_fixed_factors

                if n_fit_factors == 0:
                    break

                sf_params = global_fit_params[idx_start:(idx_start + n_fit_factors)]

                idxs_to_remove = []
                re_fit = False

                # Add dummy variable where we need to skip the index
                for id in scale_factor_exclude_ids:
                    sf_params = np.insert(sf_params, id, np.nan)

                for i, sf in enumerate(sf_params):

                    if i in scale_factor_exclude_ids:
                        continue

                    if 0.995 <= sf <= 1.015:
                        # Exclude the scale factor from the fit
                        scale_factor_exclude_ids.append(i)
                        re_fit = True

                        j1 = np.sum(np.array(scale_factor_exclude_ids) < i)
                        j2 = len(idxs_to_remove)

                        idxs_to_remove.append(idx_start + i - j1 + j2)

                if not re_fit:
                    break

                else:

                    for idx in reversed(idxs_to_remove):

                        global_fit_params = np.delete(global_fit_params, idx)
                        low_bounds = np.delete(low_bounds, idx)
                        high_bounds = np.delete(high_bounds, idx)

                        del params_names[idx]

                    kwargs['initial_parameters'] = global_fit_params
                    kwargs['low_bounds'] = low_bounds
                    kwargs['high_bounds'] = high_bounds
                    kwargs['scale_factor_exclude_ids'] = scale_factor_exclude_ids

                    global_fit_params, cov, predicted = fit_fx(**kwargs)

        rel_errors = relative_errors(global_fit_params, cov)

        self.params_names = params_names
        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(
            predicted, self.nr_signals, self.nr_olig)

        self.create_params_df_three_state()
        self.create_dg_df()

        self.global_global_global_fit_done = True

        # Obtained the scaled signal too
        if model_scale_factor:

            # signal scaled hos one sublist per selected signal type
            signal_scaled = deepcopy(self.signal_lst_multiple)
            predicted_scaled = deepcopy(self.predicted_lst_multiple)

            for value, param in zip(self.global_fit_params, self.params_names):

                if 'Scale factor' in param:

                    id = int(param.split('(M). ID: ')[-1])

                    for i in range(len(signal_scaled)):
                        signal_scaled[i][id] /= value
                        predicted_scaled[i][id] /= value

            self.signal_lst_multiple_scaled = signal_scaled
            self.predicted_lst_multiple_scaled = predicted_scaled

        return None

    def signal_to_df(self, signal_type='raw', scaled=False):
        """
        Create a dataframe with three columns: Temperature, Signal, and oligomer.
        Optimized for speed by avoiding per-curve DataFrame creation.

        Parameters
        ----------
        signal_type : {'raw', 'fitted', 'derivative'}, optional
            Which signal to include in the dataframe. 'raw' uses experimental data, 'fitted' uses model predictions,
            'derivative' uses the estimated derivative signal.
        scaled : bool, optional
            If True and signal_type == 'fitted' or 'raw', use the scaled versions if available.

        Returns
        -------
        pd.DataFrame
            A DataFrame with columns: ['Temperature', 'Signal', 'Oligomer', 'ID'].
        """

        # Flatten all arrays and repeat oligomer values accordingly

        if signal_type == 'derivative':

            deriv_lst = self.deriv_lst_multiple[0]
            temp_lst = self.temp_deriv_lst_multiple[0]

            signal_all = np.concatenate(deriv_lst)
            temp_all = np.concatenate(temp_lst)

        else:

            # temperature is shared for the experimental and fitted signals
            temp_lst = self.temp_lst_multiple[0]

            if self.max_points is not None:
                temp_lst = [subset_data(x, self.max_points) for x in temp_lst]

            temp_all = np.concatenate(temp_lst)

            # fitted data signal does not need subset!
            if signal_type == 'fitted':

                if not scaled:

                    predicted_lst = self.predicted_lst_multiple[0]

                else:

                    predicted_lst = self.predicted_lst_multiple_scaled[0]

                signal_all = np.concatenate(predicted_lst)
                temp_all = np.concatenate(temp_lst)

            # Signal_type set to 'raw'
            else:

                if not scaled:

                    signal_lst = self.signal_lst_multiple[0]

                else:

                    signal_lst = self.signal_lst_multiple_scaled[0]

                if self.max_points is not None:
                    signal_lst = [subset_data(x, self.max_points) for x in signal_lst]

                signal_all = np.concatenate(signal_lst)

        oligomer_all = np.concatenate([
            np.full_like(temp_lst[i], self.oligomer_concentrations[i], dtype=np.float64)
            for i in range(len(temp_lst))
        ])

        # Add an ID column, so we can identify the curves, even with the same oligomer concentration
        id_all = np.concatenate([
            np.full_like(temp_lst[i], i, dtype=np.int32)
            for i in range(len(temp_lst))
        ])

        signal_df = pd.DataFrame({
            'Temperature': temp_all,
            'Signal': signal_all,
            'Oligomer': oligomer_all,
            'ID': id_all
        })

        return signal_df

    def create_params_df_three_state(self):
        """
        Create a dataframe of the parameters
        """

        # convert the first param to Celsius
        self.global_fit_params[0] = temperature_to_celsius(self.global_fit_params[0])
        self.low_bounds[0] = temperature_to_celsius(self.low_bounds[0])
        self.high_bounds[0] = temperature_to_celsius(self.high_bounds[0])

        self.global_fit_params[2] = temperature_to_celsius(self.global_fit_params[2])
        self.low_bounds[2] = temperature_to_celsius(self.low_bounds[2])
        self.high_bounds[2] = temperature_to_celsius(self.high_bounds[2])

        # Create a dataframe of the parameters
        self.params_df = pd.DataFrame({
            'Parameter': self.params_names,
            'Value': self.global_fit_params,
            'Relative error (%)': self.rel_errors,
            'Fitting low Bound': self.low_bounds,
            'Fitting high Bound': self.high_bounds
        })

        return None