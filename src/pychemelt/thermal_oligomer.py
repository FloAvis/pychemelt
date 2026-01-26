"""
Main class to handle thermal denaturation data of mono- and oligomeres
The current model assumes the proteins' unfolding is reversible
TODO: 
- check out necessary data loading and if it fits, need to know data shape for adjustment
- adjust ThermalOligomere to its new purpose
- implement new fittings inspired by chirakit structure
"""

import pandas as pd
import numpy as np

from itertools import chain
from copy import deepcopy

from .main import Sample

from .utils.signals import (
    map_two_state_model_to_signal_fx,
)

from .utils.math import (
    temperature_to_kelvin,
    relative_errors,
)

from .utils.processing import (
    guess_Tm_from_derivative,
    set_param_bounds,
    adjust_value_to_interval,
    re_arrange_params,
    re_arrange_predictions,
    subset_data
)

from .utils.fitting import (
    fit_line_robust,
    fit_oligomere_unfolding_single_slopes,
    fit_oligomere_unfolding_shared_slopes_many_signals,
    fit_tc_unfolding_many_signals,
    evaluate_fitting_and_refit,
    baseline_fx_name_to_req_params
)



class ThermalOligomer(Sample):
    """
    Class to hold the data of a DSF experiment of themal unfolding with different concentrations of oligomeres
    """

    def __init__(self, name='Test'):

        super().__init__(name)

        self.fit_m_dep = False  # Fit the temperature dependence of the m-value
        self.thermodynamic_params_guess = None
        self.nr_den = 0  # Number of oligomere concentrations
        self.model = None

    def set_model(self, model_name):

        """
        Set thermodynamic model of oligomere used for the analysis.
        Currently supported are 2 state models of monomeres, dimers, trimeres and tetrameres

        Parameters
        ----------
        model_name : str
            name of the used model. Can be: "Monomer", "Dimer", "Trimer", "Tetramer".
            Case insensitive

        Notes
        -----
        This method creates/updates the following attributes on the instance:
        - self.model: oligomeric model used for analysis
        """

        allowed_models = ["monomer", "dimer", "trimer", "tetramer"]

        model = model_name.lower()

        if model not in allowed_models:
            raise ValueError(
            f"Invalid model '{model_name}'. "
            f"Allowed models are: {', '.join(m.capitalize() for m in allowed_models)}."
        )

        # Save model with first letter uppercase
        self.model = model.capitalize()

        return None

    def set_concentrations(self, concentrations=None):

        """
        Set the oligomere concentrations for the sample 

        Parameters
        ----------
        concentrations : list, optional
            List of oligomere concentrations. If None, use the sample conditions

        Notes
        -----
        Creates/updates attribute `oligomere_concentrations_pre` (numpy.ndarray)
        """

        if concentrations is None:
            concentrations = self.conditions

        self.oligomere_concentrations_pre = np.array(concentrations)

        return None

    def select_conditions(
            self,
            boolean_lst=None,
            normalise_to_global_max=True):

        """
        For each signal, select the conditions to be used for the analysis

        Parameters
        ----------
        boolean_lst : list of bool, optional
            List of booleans selecting which conditions to keep. If None, keep all.
        normalise_to_global_max : bool, optional
            If True, normalise the signal to the global maximum - per signal type

        Notes
        -----
        Creates/updates several attributes used by downstream fitting:
        - signal_lst_multiple, temp_lst_multiple : lists of lists with selected data
        - oligomere_concentrations : list of selected oligomere concentrations
        - oligomere_concentrations_expanded : flattened numpy array matching expanded signals
        - boolean_lst, normalise_to_global_max, nr_den : control flags/values
        """

        if boolean_lst is None:
            self.signal_lst_multiple = self.signal_lst_pre_multiple
            self.temp_lst_multiple = self.temp_lst_pre_multiple
            self.oligomere_concentrations = self.oligomere_concentrations_pre
        else:

            self.signal_lst_multiple = [None for _ in range(len(self.signal_lst_pre_multiple))]
            self.temp_lst_multiple = [None for _ in range(len(self.temp_lst_pre_multiple))]

            for i in range(len(self.signal_lst_pre_multiple)):
                self.signal_lst_multiple[i] = [x for j, x in enumerate(self.signal_lst_pre_multiple[i]) if
                                               boolean_lst[j]]
                self.temp_lst_multiple[i] = [x for j, x in enumerate(self.temp_lst_pre_multiple[i]) if boolean_lst[j]]

            self.oligomere_concentrations = [x for i, x in enumerate(self.oligomere_concentrations_pre) if
                                              boolean_lst[i]]

        if normalise_to_global_max:

            flat = list(chain.from_iterable(chain.from_iterable(self.signal_lst_multiple)))
            global_max = np.max(flat)  # Global maximum across all signals

            for i in range(len(self.signal_lst_multiple)):
                self.signal_lst_multiple[i] = [x / global_max * 100 for x in self.signal_lst_multiple[i]]

        self.nr_den = len(self.oligomere_concentrations)

        # Expand the number of oligomere concentrations to match the number of signals
        oligomere_concentrations = [self.oligomere_concentrations for _ in range(self.nr_signals)]

        self.oligomere_concentrations_expanded = np.concatenate(oligomere_concentrations, axis=0)

        self.boolean_lst = boolean_lst
        self.normalise_to_global_max = normalise_to_global_max

        self.oligomere_concentrations = np.array(self.oligomere_concentrations)

        self.denaturant_concentrations = self.oligomere_concentrations

        return None

    
    def guess_Cp(self):

        """
        Guess the Cp of the sample by fitting a line to the Tm and dH values

        Notes
        -----
        This method creates/updates attributes used later in fitting:
        - Tms, dHs, slope_dh_tm, intercept_dh_tm, Cp0, Cp0 assigned to self.Cp0
        """

        # If the number of residues is still zero, raise an error
        if self.n_residues == 0:
            raise ValueError('The number of residues is still zero. Please set n_residues before calling guess_Cp')

        # Requires self.single_fit_done

        Cp0 = self.n_residues * 0.0148 - 0.1267

        # Cp0 needs to be positive
        Cp0 = max(Cp0, 0)

        self.Cp0 = Cp0

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

        for i in range(len(self.signal_lst_multiple)):
            tm_lst.append(guess_Tm_from_derivative(
                self.temp_deriv_lst_multiple[i],
                self.deriv_lst_multiple[i],
                x1,
                x2
            ))

        Tm = np.average(tm_lst)



        p0 = [Tm, 40, self.Cp0]


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

        # We need to append as many bN and bU as the number of oligomere concentrations
        # times the number of signal types
        for signal in self.signal_names:

            params_names += (['intercept_native - ' + str(self.oligomere_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_den)])

        for signal in self.signal_names:

            params_names += (['intercept_unfolded - ' + str(self.oligomere_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_den)])

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p0 = np.concatenate([p0, self.second_param_Ns_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomere_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p0 = np.concatenate([p0, self.second_param_Us_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.oligomere_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p0 = np.concatenate([p0, self.third_param_Ns_expanded])
            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.oligomere_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p0 = np.concatenate([p0, self.third_param_Us_expanded])

            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.oligomere_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        low_bounds = (p0.copy())
        high_bounds = (p0.copy())

        low_bounds[3:], high_bounds[3:] = set_param_bounds(p0[3:],params_names[3:])

        self.limited_tm = tm_limits is not None

        if self.limited_tm:

            tm_lower, tm_upper = tm_limits

        else:

            tm_lower = p0[0] - 12
            tm_upper = np.max([self.user_max_temp + 20, p0[0] + 10])

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
            'oligomere_concentrations' : self.oligomere_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds' : low_bounds,
            'high_bounds' : high_bounds,
            'cp_value' : cp_value,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx,
            'signal_fx' : signal_fx
        }

        fit_fx = fit_oligomere_unfolding_single_slopes

        # Do a quick prefit with a reduced data set
        if self.pre_fit:

            kwargs['list_of_temperatures'] = self.temp_lst_expanded_subset
            kwargs['list_of_signals'] = self.signal_lst_expanded_subset

            global_fit_params, cov, predicted = fit_fx(**kwargs)

            p0 = global_fit_params

        # Now use the whole dataset
        kwargs['list_of_temperatures'] = self.temp_lst_expanded
        kwargs['list_of_signals'] = self.signal_lst_expanded

        # First fit without m-value dependence on temperature
        global_fit_params, cov, predicted = fit_fx(**kwargs)

        fit_m_dep = False

        global_fit_params, cov, predicted, p0, low_bounds, high_bounds = evaluate_fitting_and_refit(
            global_fit_params,
            cov,
            predicted,
            high_bounds,
            low_bounds,
            p0,
            fit_m_dep,
            self.limited_cp,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(predicted, self.nr_signals, self.nr_den)

        self.global_fit_done = True

        self.fit_m_dep = fit_m_dep

        self.params_names = params_names

        self.create_params_df()
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

        n_datasets = self.nr_den * self.nr_signals

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

        # Baselines are still independent for each signal and oligomere concentration
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

            'oligomer_concentrations': self.oligomere_concentrations_expanded,
            'list_of_temperatures': self.temp_lst_expanded_subset,
            'list_of_signals': self.signal_lst_expanded_subset,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'cp_value': self.cp_value,
            'fit_m1': False,
            'signal_ids':self.signal_ids,
            'baseline_native_fx': self.baseline_N_fx,
            'baseline_unfolded_fx': self.baseline_U_fx,
            'signal_fx' : signal_fx
        }

        fit_fx = fit_oligomere_unfolding_shared_slopes_many_signals

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
            self.fit_m_dep,
            self.limited_cp,
            self.limited_dh,
            self.limited_tm,
            self.fixed_cp,
            kwargs,
            fit_fx,
        )

        rel_errors = relative_errors(global_fit_params, cov)

        self.p0 = p0
        self.low_bounds = low_bounds
        self.high_bounds = high_bounds
        self.global_fit_params = global_fit_params
        self.rel_errors = rel_errors

        self.predicted_lst_multiple = re_arrange_predictions(
            predicted, self.nr_signals, self.nr_den)

        self.params_names = params_names

        self.create_params_df()
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
            If True, model a scale factor for each oligomere concentration

        Notes
        -----
        Updates many global fitting attributes and sets `global_global_global_fit_done` when complete. If
        `model_scale_factor` is True the method also creates scaled signal attributes:
        - signal_lst_multiple_scaled, predicted_lst_multiple_scaled
        """

        # Requires global global fit done
        if not self.global_global_fit_done:
            self.fit_thermal_unfolding_global_global()

        param_init = 3 + self.fit_m_dep + (self.cp_value is None)

        params_names = self.params_names[:param_init]

        p0 = self.global_fit_params[:param_init]
        low_bounds = self.low_bounds[:param_init]
        high_bounds = self.high_bounds[:param_init]

        n_datasets = self.nr_den * self.nr_signals

        p1Ns = self.global_fit_params[param_init:param_init + n_datasets]
        p1Us = self.global_fit_params[param_init + n_datasets:param_init + 2 * n_datasets]

        p1Ns_per_signal = re_arrange_params(p1Ns, self.nr_signals)
        p1Us_per_signal = re_arrange_params(p1Us, self.nr_signals)

        m1s, b1s, m1s_low, b1s_low, m1s_high, b1s_high = [], [], [], [], [], []
        m2s, b2s, m2s_low, b2s_low, m2s_high, b2s_high = [], [], [], [], [], []

        for p1Ns, p1Us in zip(p1Ns_per_signal, p1Us_per_signal):

            # Estimate the slope of bNs versus oligomere concentration
            m1, b1 = fit_line_robust(self.oligomere_concentrations, p1Ns)
            m1_low = m1 / 100 if m1 > 0 else 100 * m1
            m1_high = 100 * m1 if m1 > 0 else m1 / 100
            b1_low = b1 / 100 if b1 > 0 else 100 * b1
            b1_high = 100 * b1 if b1 > 0 else b1 / 100

            # Estimate the slope of bUs versus oligomere concentration
            m2, b2 = fit_line_robust(self.oligomere_concentrations, p1Us)
            m2_low = m2 / 100 if m2 > 0 else 100 * m2
            m2_high = 100 * m2 if m2 > 0 else m2 / 100
            b2_low = b2 / 100 if b2 > 0 else 100 * b2
            b2_high = 100 * b2 if b2 > 0 else b2 / 100

            m1s.append(m1)
            b1s.append(b1)
            m1s_low.append(m1_low)
            b1s_low.append(b1_low)
            m1s_high.append(m1_high)
            b1s_high.append(b1_high)

            m2s.append(m2)
            b2s.append(b2)
            m2s_low.append(m2_low)
            b2s_low.append(b2_low)
            m2s_high.append(m2_high)
            b2s_high.append(b2_high)

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

        params_names += ['oligomere_slope_term_native - ' + signal_name for signal_name in self.signal_names]
        params_names += ['oligomere_slope_term_unfolded - ' + signal_name for signal_name in self.signal_names]

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

        p0 = np.concatenate([p0, b1s, b2s])
        low_bounds = np.concatenate([low_bounds, b1s_low, b2s_low])
        high_bounds = np.concatenate([high_bounds, b1s_high, b2s_high])

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            p0 = np.concatenate([p0, kNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kNs])

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            p0 = np.concatenate([p0, kUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_kUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_kUs])

        p0 = np.concatenate([p0, m1s, m2s])
        low_bounds = np.concatenate([low_bounds, m1s_low, m2s_low])
        high_bounds = np.concatenate([high_bounds, m1s_high, m2s_high])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            p0 = np.concatenate([p0, qNs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qNs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qNs])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            p0 = np.concatenate([p0, qUs])
            low_bounds = np.concatenate([low_bounds, low_bounds_qUs])
            high_bounds = np.concatenate([high_bounds, high_bounds_qUs])

        # Increase the bounds for c_N and c_U
        # Find index in the param names
        for signal_name in self.signal_names:

            c_N_name = 'oligomere_slope_term_native - ' + signal_name
            c_U_name = 'oligomere_slope_term_unfolded - ' + signal_name

            c_N_idx = params_names.index(c_N_name)
            c_U_idx = params_names.index(c_U_name)

            low_bounds[c_N_idx] -= 5
            high_bounds[c_N_idx] += 5

            low_bounds[c_U_idx] -= 5
            high_bounds[c_U_idx] += 5

        # If required, include a scale factor for each oligomere concentration
        if model_scale_factor:
            # The last oligomere concentration is fixed to 1, the rest are fitted
            scale_factors = [1 for _ in range(self.nr_den - 1)]
            scale_factors_low = [0.5882 for _ in range(self.nr_den - 1)]
            scale_factors_high = [1.7 for _ in range(self.nr_den - 1)]

            p0 = np.concatenate([p0, scale_factors])
            low_bounds = np.concatenate([low_bounds, scale_factors_low])
            high_bounds = np.concatenate([high_bounds, scale_factors_high])

            params_names += ['Scale factor - ' + str(d) + ' (M). ID: ' + str(i) for
                             i, d in enumerate(self.oligomere_concentrations)]

            params_names.pop()  # Remove the last one, as it is fixed to 1

        scale_factor_exclude_ids = [self.nr_den - 1] if model_scale_factor else []

        # Do a prefit with a reduced dataset
        kwargs = {

            'list_of_temperatures' : self.temp_lst_expanded_subset,
            'list_of_signals' : self.signal_lst_expanded_subset,
            'signal_ids' : self.signal_ids,
            'oligomere_concentrations': self.oligomere_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'fit_m1': self.fit_m_dep,
            'model_scale_factor':model_scale_factor,
            'cp_value' : self.cp_value,
            'scale_factor_exclude_ids':scale_factor_exclude_ids,
            'signal_fx' : signal_two_state_tc_unfolding,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx,
            'fit_native_den_slope' : True,
            'fit_unfolded_den_slope' : True
        }

        fit_fx = fit_tc_unfolding_many_signals

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

            # 3 parameters corresponding to Tm, dH, m
            # plus Cp if fitted
            # plus m1 if fitted
            idx_start = 3 + self.fit_m_dep + (self.cp_value is None)

            native_factor   = 2+np.sum(baseline_fx_name_to_req_params(self.baseline_N_fx))
            unfolded_factor = 2+np.sum(baseline_fx_name_to_req_params(self.baseline_U_fx))

            # Add index according to the native baseline polynomial order
            idx_start += native_factor * self.nr_signals
            # Add index according to the unfolded baseline polynomial order
            idx_start += unfolded_factor * self.nr_signals

            # Take m1 into account, if fitting it
            idx_start += self.fit_m_dep

            for _ in range(5):

                # Sort in ascending order the IDs to exclude
                scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

                n_fixed_factors = len(scale_factor_exclude_ids)
                n_fit_factors = self.nr_den - n_fixed_factors

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
            predicted, self.nr_signals, self.nr_den)

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

    def signal_to_df(self, signal_type='raw', scaled=False):
        """
        Create a dataframe with three columns: Temperature, Signal, and Oligomere.
        Optimized for speed by avoiding per-curve DataFrame creation.

        Parameters
        ----------
        signal_type : {'raw', 'fitted', 'derivative'}, optional
            Which signal to include in the dataframe. 'raw' uses experimental data, 'fitted' uses model predictions,
            'derivative' uses the estimated derivative signal.
        scaled : bool, optional
            If True and signal_type == 'fitted' or 'raw', use the scaled versions if available.
        """

        # Flatten all arrays and repeat oligomere values accordingly

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

        denat_all = np.concatenate([
            np.full_like(temp_lst[i], self.oligomere_concentrations[i], dtype=np.float64)
            for i in range(len(temp_lst))
        ])

        # Add an ID column, so we can identify the curves, even with the same oligomere concentration
        id_all = np.concatenate([
            np.full_like(temp_lst[i], i, dtype=np.int32)
            for i in range(len(temp_lst))
        ])

        signal_df = pd.DataFrame({
            'Temperature': temp_all,
            'Signal': signal_all,
            'Oligomere': denat_all,
            'ID': id_all
        })

        return signal_df