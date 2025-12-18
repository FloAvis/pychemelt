"""
Main class to handle thermal and chemical denaturation data
The current model assumes the protein is a monomer and that the unfolding is reversible
"""

import pandas as pd
import numpy as np

from .utils.signals import *
from .utils.fitting import *
from .utils.files import *
from .utils.math import *

from itertools import chain

from copy import deepcopy


class Sample:
    """
    Class to hold the data of a single sample and fit it
    """

    def __init__(self, name='Test'):

        self.name = name
        self.signal_dic = {}
        self.deriv_dic = {}
        self.temp_dic = {}
        self.conditions = []
        self.labels = []

        self.signals = []

        self.first_param_Ns_per_signal = []
        self.first_param_Us_per_signal = []
        self.second_param_Ns_per_signal = []
        self.second_param_Us_per_signal = []
        self.third_param_Ns_per_signal = []
        self.third_param_Us_per_signal = []

        self.Cp0 = 0

        self.nr_den = 0  # Number of denaturant concentrations

        self.signal_ids = None

        self.t_melting_init_multiple = None # Initial guess for tm based on derivative

        self.fit_m_dep = False  # Fit the temperature dependence of the m-value

        self.single_fit_done = False  # Individual thermodynamic parameters, baselines and slopes

        self.global_fit_done = False  # Global thermodynamic parameters, local baselines and slopes

        self.global_global_fit_done = False  # Global thermodynamic parameters, global slopes and local baselines

        self.global_global_global_fit_done = False  # Global thermodynamic parameters, global slopes and global baselines

        self.global_fit_params = None

        self.thermodynamic_params_guess = None

        self.global_min_temp = 30
        self.global_max_temp = 80

        self.n_residues = 0

        self.pre_fit = True
        self.max_points = None

        self.user_min_temp = 5
        self.user_max_temp = 100

        self.predicted = None # Flattened list of fitted signals

    def read_file(self, file):

        """
        Read the file and load the data into the sample object

        Parameters
        ----------
        file : str
            Path to the file

        Returns
        -------
        bool
            True if the file was read and loaded into the sample object
        """

        file_type = detect_file_type(file)

        read_fx_map = {
            'prometheus': load_nanoDSF_xlsx,
            'panta': load_panta_xlsx,
            'uncle': load_uncle_multi_channel,
            'thermofluor': load_thermofluor_xlsx,
            'quantstudio': load_quantstudio_txt,
            'mx3005p': load_mx3005p_txt,
            'supr': load_supr_dsf,
            'csv': load_csv_file
        }

        read_fx = read_fx_map.get(file_type)

        signal_data_dic, temp_data_dic, conditions, signals_i = read_fx(file)

        # If self.signals is not empty, signals_i must be the same as self.signals
        if len(self.signals) > 0:
            if set(signals_i) != set(self.signals):
                # We can't combine files with different signals
                return False

        for si in signals_i:
            if si not in self.signals:
                self.signals.append(si)

        # For each key in signal_data_dic, find if the key already exists in self.signal_dic and append the data
        for k, v in signal_data_dic.items():
            # v is a list of arrays

            # make sure we have a list of arrays
            v = [np.array(x) for x in v]

            if k in self.signal_dic:
                self.signal_dic[k].extend(v)
            else:
                self.signal_dic[k] = v

        # For each key in temp_data_dic, find if the key already exists in self.temp_dic and append the data
        for k, v in temp_data_dic.items():

            min_temp_v = np.min(v)
            max_temp_v = np.max(v)

            self.global_min_temp = min(min_temp_v, self.global_min_temp)
            self.global_max_temp = max(max_temp_v, self.global_max_temp)

            # make sure we have a list of arrays
            v = [np.array(x) for x in v]

            if k in self.temp_dic:
                self.temp_dic[k].extend(v)
            else:
                self.temp_dic[k] = v

        # Keep original labels
        self.labels += conditions

        # Remove all non-numeric characters from the conditions
        conditions = clean_conditions_labels(conditions)

        self.conditions += conditions

        return True

    def read_multiple_files(self, files):
        """
        Read multiple files and load the data into the sample object

        Parameters
        ----------
        files : list or str
            List of paths to the files (or a single path)

        Returns
        -------
        bool
            True if the files were read and loaded into the sample object
        """

        # Convert to list if not isinstance(files, list):
        if not isinstance(files, list):
            files = [files]

        for file in files:
            read_status = self.read_file(file)
            if not read_status:
                return False

        return True

    def set_signal(self, signal_names):

        """
        Set multiple signals to be used for the analysis.
        This way, we can fit globally multiple signals at the same time, such as 350nm and 330nm

        Parameters
        ----------
        signal_names : list or str
            List of names of the signals to be used. E.g., ['350nm','330nm'] or a single name

        Notes
        -----
        This method creates/updates the following attributes on the instance:
        - signal_lst_pre_multiple, temp_lst_pre_multiple : lists of lists
        - signal_names : list of signal name strings
        - nr_signals : int, number of signal types
        """

        # Convert signal_names to list if it is a string
        if isinstance(signal_names, str):
            signal_names = [signal_names]

        signals = []
        temps = []

        for signal_name in signal_names:
            signal = self.signal_dic[signal_name]
            temp = self.temp_dic[signal_name]

            signals.append(signal)
            temps.append(temp)

        self.signal_lst_pre_multiple = signals
        self.temp_lst_pre_multiple = temps

        self.signal_names = signal_names
        self.nr_signals = len(signal_names)

        return None

    def set_denaturant_concentrations(self, concentrations=None):

        """
        Set the denaturant concentrations for the sample

        Parameters
        ----------
        concentrations : list, optional
            List of denaturant concentrations. If None, use the sample conditions

        Notes
        -----
        Creates/updates attribute `denaturant_concentrations_pre` (numpy.ndarray)
        """

        if concentrations is None:
            concentrations = self.conditions

        self.denaturant_concentrations_pre = np.array(concentrations)

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
        - denaturant_concentrations : list of selected denaturant concentrations
        - denaturant_concentrations_expanded : flattened numpy array matching expanded signals
        - boolean_lst, normalise_to_global_max, nr_den : control flags/values
        """

        if boolean_lst is None:
            self.signal_lst_multiple = self.signal_lst_pre_multiple
            self.temp_lst_multiple = self.temp_lst_pre_multiple
            self.denaturant_concentrations = self.denaturant_concentrations_pre
        else:

            self.signal_lst_multiple = [None for _ in range(len(self.signal_lst_pre_multiple))]
            self.temp_lst_multiple = [None for _ in range(len(self.temp_lst_pre_multiple))]

            for i in range(len(self.signal_lst_pre_multiple)):
                self.signal_lst_multiple[i] = [x for j, x in enumerate(self.signal_lst_pre_multiple[i]) if
                                               boolean_lst[j]]
                self.temp_lst_multiple[i] = [x for j, x in enumerate(self.temp_lst_pre_multiple[i]) if boolean_lst[j]]

            self.denaturant_concentrations = [x for i, x in enumerate(self.denaturant_concentrations_pre) if
                                              boolean_lst[i]]

        if normalise_to_global_max:

            flat = list(chain.from_iterable(chain.from_iterable(self.signal_lst_multiple)))
            global_max = np.max(flat)  # Global maximum across all signals

            for i in range(len(self.signal_lst_multiple)):
                self.signal_lst_multiple[i] = [x / global_max * 100 for x in self.signal_lst_multiple[i]]

        self.nr_den = len(self.denaturant_concentrations)

        # Expand the number of denaturant concentrations to match the number of signals
        denaturant_concentrations = [self.denaturant_concentrations for _ in range(self.nr_signals)]

        self.denaturant_concentrations_expanded = np.concatenate(denaturant_concentrations, axis=0)

        self.boolean_lst = boolean_lst
        self.normalise_to_global_max = normalise_to_global_max

        self.denaturant_concentrations = np.array(self.denaturant_concentrations)

        return None

    def set_temperature_range(
            self,
            min_temp=0,
            max_temp=100):
        """
        Set the temperature range for the sample

        Parameters
        ----------
        min_temp : float, optional
            Minimum temperature
        max_temp : float, optional
            Maximum temperature
        """

        # Give an error if max_temp is smaller than min_temp
        if max_temp < min_temp:
            raise ValueError('max_temp must be larger than min_temp')

        # Limit the signal to the temperature range
        for i in range(len(self.signal_lst_multiple)):
            self.signal_lst_multiple[i], self.temp_lst_multiple[i] = subset_signal_by_temperature(

                self.signal_lst_multiple[i],
                self.temp_lst_multiple[i],
                min_temp,
                max_temp

            )

        self.user_min_temp = min_temp
        self.user_max_temp = max_temp

        return None

    def set_signal_id(self):
        """
        Create a list with the same length as the total number of signals
        The elements of the list indicated the ID of the signal,
        e.g., all 350nm datasets are mapped to 0, all 330nm datasets to 1, etc.
        """

        signal_ids = []

        for i, s in enumerate(self.signal_lst_multiple):
            signal_ids.extend([i for _ in range(len(s))])

        self.signal_ids = signal_ids

        return None

    def estimate_derivative(self, window_length=8):

        """
        Estimate the derivative of the signal using Savitzky-Golay filter

        Parameters
        ----------
        window_length : int, optional
            Length of the filter window in degrees

        Notes
        -----
        Creates/updates attributes:
        - temp_deriv_lst_multiple, deriv_lst_multiple : lists storing estimated derivatives and corresponding temps
        """

        self.temp_deriv_lst_multiple = []
        self.deriv_lst_multiple = []

        for i in range(len(self.signal_lst_multiple)):

            temp_deriv_lst = []
            deriv_lst = []

            for s, t in zip(self.signal_lst_multiple[i], self.temp_lst_multiple[i]):

                check = is_evenly_spaced(t)

                if check:

                    derivative = first_derivative_savgol(t, s, window_length)

                else:

                    t_for_ip = np.arange(np.min(t), np.max(t), 0.1)

                    # We interpolate the data to make it evenly spaced, every 0.1 degrees
                    s = np.interp(t_for_ip, t, s)
                    t = t_for_ip

                    # We use the Savitzky-Golay filter to estimate the derivative
                    derivative = first_derivative_savgol(t, s, window_length)

                temp_deriv_lst.append(t)
                deriv_lst.append(derivative)

            self.temp_deriv_lst_multiple.append(temp_deriv_lst)
            self.deriv_lst_multiple.append(deriv_lst)

        return None

    def guess_Tm(self, x1=6, x2=11):

        """
        Guess the Tm of the sample using the derivative of the signal

        Parameters
        ----------
        x1 : float, optional
            Shift from the minimum and maximum temperature to estimate the median of the initial and final baselines
        x2 : float, optional
            Shift from the minimum and maximum temperature to estimate the median of the initial and final baselines

        Notes
        -----
        x2 must be greater than x1.

        This method creates/updates attributes:
        - t_melting_init_multiple : list of initial Tm guesses per signal
        - t_melting_df_multiple : list of pandas.DataFrame objects with Tm vs Denaturant
        """

        self.t_melting_init_multiple = []
        self.t_melting_df_multiple = []

        for i in range(len(self.signal_lst_multiple)):
            t_melting_init = guess_Tm_from_derivative(
                self.temp_deriv_lst_multiple[i],
                self.deriv_lst_multiple[i],
                x1,
                x2
            )
            # Create a dataframe of the Tm values versus the denaturant concentrations
            t_melting_df = pd.DataFrame({
                'Tm': t_melting_init,
                'Denaturant': self.denaturant_concentrations
            })

            self.t_melting_df_multiple.append(t_melting_df)

            self.t_melting_init_multiple.append(t_melting_init)

        return None

    def estimate_baseline_parameters(
            self,
            native_baseline_type,
            unfolded_baseline_type,
            window_range_native=12,
            window_range_unfolded=12):

        """
        Estimate the baseline parameters for multiple signals

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

        for i in range(len(self.signal_lst_multiple)):
            p1Ns, p1Us, p2Ns, p2Us, p3Ns, p3Us = estimate_signal_baseline_params(
                self.signal_lst_multiple[i],
                self.temp_lst_multiple[i],
                native_baseline_type,
                unfolded_baseline_type,
                window_range_native,
                window_range_unfolded
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

        self.baseline_N_fx = baseline_fx_dic[native_baseline_type]
        self.baseline_U_fx = baseline_fx_dic[unfolded_baseline_type]

        self.native_baseline_type = native_baseline_type
        self.unfolded_baseline_type = unfolded_baseline_type

        return None

    def reset_fittings_results(self):

        self.global_fit_done = False  # Global thermodynamic parameters, local baselines and slopes
        self.global_global_fit_done = False  # Global thermodynamic parameters, global slopes and local baselines
        self.global_global_global_fit_done = False  # Global thermodynamic parameters, global slopes and global baselines

        self.first_param_Ns_per_signal = []
        self.first_param_Us_per_signal = []
        self.second_param_Ns_per_signal = []
        self.second_param_Us_per_signal = []
        self.third_param_Ns_per_signal = []
        self.third_param_Us_per_signal = []

        self.params_df = None
        self.dg_df = None

        self.global_fit_params = None
        self.predicted_lst_multiple = None

        self.predicted_lst_multiple_scaled = None
        self.signal_lst_multiple_scaled = None

        self.p0 = None
        self.low_bounds = None
        self.high_bounds = None
        self.rel_errors = None

        return None

    def fit_thermal_unfolding_local(self):

        """
        Fit the thermal unfolding of the sample using the signal and temperature data
        We fit one curve at a time, with individual parameters
        """

        # Require self.t_melting_init_multiple
        if self.t_melting_init_multiple is None:

            self.estimate_derivative()
            self.guess_Tm()

        self.Tms_multiple = []
        self.dHs_multiple = []
        self.predicted_lst_multiple = []

        for i in range(len(self.signal_lst_multiple)):

            Tms, dHs, predicted_lst = fit_local_thermal_unfolding_to_signal_lst(
                self.signal_lst_multiple[i],
                self.temp_lst_multiple[i],
                self.t_melting_init_multiple[i],
                self.first_param_Ns_per_signal[i],
                self.first_param_Us_per_signal[i],
                self.second_param_Ns_per_signal[i],
                self.second_param_Us_per_signal[i],
                self.third_param_Ns_per_signal[i],
                self.third_param_Us_per_signal[i],
                baseline_native_fx=self.baseline_N_fx,
                baseline_unfolded_fx=self.baseline_U_fx
            )

            self.Tms_multiple.append(Tms)
            self.dHs_multiple.append(dHs)
            self.predicted_lst_multiple.append(predicted_lst)

        self.single_fit_done = True

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

        expected_Cp0 = self.n_residues * 0.0148 - 0.1267

        if not self.single_fit_done:
            self.fit_thermal_unfolding_local()

        try:

            Tms = []
            dHs = []

            for i in range(len(self.Tms_multiple)):
                Tms.extend(self.Tms_multiple[i])
                dHs.extend(self.dHs_multiple[i])

            self.Tms = Tms
            self.dHs = dHs

            tms = np.array(self.Tms)
            dhs = np.array(self.dHs)

            m, b = fit_line_robust(tms, dhs)

            outliers = find_line_outliers(m, b, tms, dhs)

            if len(outliers) > 0:
                # Remove outliers
                tms = np.delete(tms, outliers)
                dhs = np.delete(dhs, outliers)

                # Assign the new values
                self.Tms = tms
                self.dHs = dhs

                m, b = fit_line_robust(self.Tms, self.dHs)

            self.slope_dh_tm = m
            self.intercept_dh_tm = b

            Cp0 = m if m > 0 else -1

            # Verify that the initial Cp is between the expected range
            if Cp0 < expected_Cp0 / 1.5 or Cp0 > expected_Cp0 * 1.5:
                Cp0 = expected_Cp0

        except:

            Cp0 = expected_Cp0

        # Cp0 needs to be positive
        Cp0 = max(Cp0, 0)

        self.Cp0 = Cp0

        return None

    def expand_multiple_signal(self):

        """
        Create a single list with all the signals
        Create a single list with all the temperatures

        Notes
        -----
        Creates/updates attributes:
        - signal_lst_expanded, temp_lst_expanded
        - signal_lst_expanded_subset, temp_lst_expanded_subset
        """

        # Create a single list with all the signals
        self.signal_lst_expanded = []
        self.temp_lst_expanded = []

        for i in range(len(self.signal_lst_multiple)):
            self.signal_lst_expanded += self.signal_lst_multiple[i]
            self.temp_lst_expanded += self.temp_lst_multiple[i]

        # Create a reduced dataset for faster fitting
        self.signal_lst_expanded_subset = [subset_data(x, 160) for x in self.signal_lst_expanded]
        self.temp_lst_expanded_subset = [subset_data(x, 160) for x in self.temp_lst_expanded]

        if self.max_points is not None:
            self.signal_lst_expanded = [subset_data(x, self.max_points) for x in self.signal_lst_expanded]
            self.temp_lst_expanded = [subset_data(x, self.max_points) for x in self.temp_lst_expanded]

        return None

    def guess_initial_parameters(
            self,
            native_baseline_type,
            unfolded_baseline_type,
            window_range_native=12,
            window_range_unfolded=12
    ):

        # We will use the Ratio signal, if available, to estimate the initial parameters
        use_ratio = 'Ratio' in self.signals and self.signal_names[0] != 'Ratio'

        if use_ratio:

            current_signal = self.signal_names[0]

            # Extract temperature limits
            self.set_signal('Ratio')
            self.select_conditions(self.boolean_lst, normalise_to_global_max=True)
            self.set_temperature_range(self.user_min_temp, self.user_max_temp)

        # Fit the data using the linear - constant option
        self.estimate_baseline_parameters(
            native_baseline_type,
            unfolded_baseline_type,
            window_range_native,
            window_range_unfolded
        )

        self.fit_thermal_unfolding_local()
        self.guess_Cp()

        # Apply a first fitting round to obtain initial estimates for the thermodynamic parameters
        self.fit_thermal_unfolding_global()

        self.thermodynamic_params_guess = self.global_fit_params[:4]

        if use_ratio:
            # Go back to the original signal
            self.set_signal(current_signal)
            self.select_conditions(self.boolean_lst, normalise_to_global_max=self.normalise_to_global_max)
            self.set_temperature_range(self.user_min_temp, self.user_max_temp)

        return None

    def create_params_df(self):
        """
        Create a dataframe of the parameters
        """

        # convert the first param to Celsius
        self.global_fit_params[0] = temperature_to_celsius(self.global_fit_params[0])

        # Create a dataframe of the parameters
        self.params_df = pd.DataFrame({
            'Parameter': self.params_names,
            'Value': self.global_fit_params,
            'Relative error (%)': self.rel_errors,
            'Fitting low Bound': self.low_bounds,
            'Fitting high Bound': self.high_bounds
        })

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
            fit_m_dep=False,
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
        fit_m_dep : bool, optional
            If True, fit the temperature dependence of the m-value
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
        - flags: global_fit_done, fit_m_dep, limited_tm, limited_dh, limited_cp, fixed_cp
        """

        # Requires Cp0
        if self.Cp0 <= 0:
            raise ValueError('Cp0 must be positive. Please run guess_Cp before fitting globally.')

        max_tm_id = np.argmax(self.Tms)

        if self.thermodynamic_params_guess is None:

            p0 = [self.Tms[max_tm_id], np.max([self.dHs[max_tm_id], 40]), self.Cp0, 2.5]

        else:

            p0 = self.thermodynamic_params_guess

        params_names = [
            'Tm (°C)',
            'ΔH (kcal/mol)',
            'Cp (kcal/mol/°C)',
            'm-value (kcal/mol/M)']

        self.first_param_Ns_expanded = np.concatenate(self.first_param_Ns_per_signal, axis=0)
        self.first_param_Us_expanded = np.concatenate(self.first_param_Us_per_signal, axis=0)
        self.second_param_Ns_expanded = np.concatenate(self.second_param_Ns_per_signal, axis=0)
        self.second_param_Us_expanded = np.concatenate(self.second_param_Us_per_signal, axis=0)
        self.third_param_Ns_expanded = np.concatenate(self.third_param_Ns_per_signal, axis=0)
        self.third_param_Us_expanded = np.concatenate(self.third_param_Us_per_signal, axis=0)

        p0 = np.concatenate([p0, self.first_param_Ns_expanded, self.first_param_Us_expanded])

        # We need to append as many bN and bU as the number of denaturant concentrations
        # times the number of signal types
        for signal in self.signal_names:

            params_names += (['intercept_native - ' + str(self.denaturant_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_den)])

        for signal in self.signal_names:

            params_names += (['intercept_unfolded - ' + str(self.denaturant_concentrations[i]) +
                              ' - ' + str(signal) for i in range(self.nr_den)])

        if self.native_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_native' if self.native_baseline_type == 'exponential' else 'slope_term_native'

            p0 = np.concatenate([p0, self.second_param_Ns_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.unfolded_baseline_type in ['linear', 'quadratic','exponential']:

            param_name = 'pre_exponential_factor_unfolded' if self.unfolded_baseline_type == 'exponential' else 'slope_term_unfolded'

            p0 = np.concatenate([p0, self.second_param_Us_expanded])

            for signal in self.signal_names:
                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.native_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_native' if self.native_baseline_type == 'exponential' else 'quadratic_term_native'

            p0 = np.concatenate([p0, self.third_param_Ns_expanded])
            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        if self.unfolded_baseline_type in ['quadratic', 'exponential']:

            param_name = 'exponential_coefficient_unfolded' if self.unfolded_baseline_type == 'exponential' else 'quadratic_term_unfolded'

            p0 = np.concatenate([p0, self.third_param_Us_expanded])

            for signal in self.signal_names:

                params_names += ([param_name + ' - ' + str(self.denaturant_concentrations[i]) +
                                  ' - ' + str(signal) for i in range(self.nr_den)])

        low_bounds = (p0.copy())
        high_bounds = (p0.copy())

        low_bounds[4:], high_bounds[4:] = set_param_bounds(p0[4:],params_names[4:])

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

            if self.thermodynamic_params_guess is None:

                dh_lower = 10
                dh_upper = 500

            else:

                dh_lower = self.thermodynamic_params_guess[1] / 5
                dh_upper = self.thermodynamic_params_guess[1] * 5

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

        id_m = 2 + (not self.fixed_cp)

        low_bounds[id_m] = 0.5
        high_bounds[id_m] = 9

        # Populate the expanded signal and temperature lists
        self.expand_multiple_signal()

        kwargs = {
            'denaturant_concentrations' : self.denaturant_concentrations_expanded,
            'initial_parameters': p0,
            'low_bounds' : low_bounds,
            'high_bounds' : high_bounds,
            'cp_value' : cp_value,
            'baseline_native_fx' : self.baseline_N_fx,
            'baseline_unfolded_fx' : self.baseline_U_fx,
            'signal_fx' : signal_two_state_tc_unfolding
        }

        fit_fx = fit_tc_unfolding_single_slopes

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

        # Insert the initial estimate for the m-value dependence of temperature, in the position 4
        if fit_m_dep:

            kwargs['fit_m1'] = fit_m_dep

            p0 = global_fit_params
            p0 = np.insert(p0, id_m+1, 0)
            low_bounds = np.insert(low_bounds, id_m+1, -0.5)
            high_bounds = np.insert(high_bounds, id_m+1, 0.5)

            kwargs['initial_parameters'] = p0
            kwargs['low_bounds'] = low_bounds
            kwargs['high_bounds'] = high_bounds

            params_names.insert(id_m+1, 'm - T dependence')

            global_fit_params, cov, predicted = fit_fx(**kwargs)

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

        param_init = 3 + self.fit_m_dep + (self.cp_value is None)

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

        # Baselines are still independent for each signal and denaturant concentration
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

        kwargs = {

            'denaturant_concentrations': self.denaturant_concentrations_expanded,
            'list_of_temperatures': self.temp_lst_expanded_subset,
            'list_of_signals': self.signal_lst_expanded_subset,
            'initial_parameters': p0,
            'low_bounds': low_bounds,
            'high_bounds': high_bounds,
            'cp_value': self.cp_value,
            'fit_m1': self.fit_m_dep,
            'signal_ids':self.signal_ids,
            'baseline_native_fx': self.baseline_N_fx,
            'baseline_unfolded_fx': self.baseline_U_fx,
            'signal_fx' : signal_two_state_tc_unfolding
        }

        fit_fx = fit_tc_unfolding_shared_slopes_many_signals

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
            If True, model a scale factor for each denaturant concentration

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

            # Estimate the slope of bNs versus denaturant concentration
            m1, b1 = fit_line_robust(self.denaturant_concentrations, p1Ns)
            m1_low = m1 / 100 if m1 > 0 else 100 * m1
            m1_high = 100 * m1 if m1 > 0 else m1 / 100
            b1_low = b1 / 100 if b1 > 0 else 100 * b1
            b1_high = 100 * b1 if b1 > 0 else b1 / 100

            # Estimate the slope of bUs versus denaturant concentration
            m2, b2 = fit_line_robust(self.denaturant_concentrations, p1Us)
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

        params_names += ['denaturant_slope_term_native - ' + signal_name for signal_name in self.signal_names]
        params_names += ['denaturant_slope_term_unfolded - ' + signal_name for signal_name in self.signal_names]

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

            c_N_name = 'denaturant_slope_term_native - ' + signal_name
            c_U_name = 'denaturant_slope_term_unfolded - ' + signal_name

            c_N_idx = params_names.index(c_N_name)
            c_U_idx = params_names.index(c_U_name)

            low_bounds[c_N_idx] -= 5
            high_bounds[c_N_idx] += 5

            low_bounds[c_U_idx] -= 5
            high_bounds[c_U_idx] += 5

        # If required, include a scale factor for each denaturant concentration
        if model_scale_factor:
            # The last denaturant concentration is fixed to 1, the rest are fitted
            scale_factors = [1 for _ in range(self.nr_den - 1)]
            scale_factors_low = [0.5882 for _ in range(self.nr_den - 1)]
            scale_factors_high = [1.7 for _ in range(self.nr_den - 1)]

            p0 = np.concatenate([p0, scale_factors])
            low_bounds = np.concatenate([low_bounds, scale_factors_low])
            high_bounds = np.concatenate([high_bounds, scale_factors_high])

            params_names += ['Scale factor - ' + str(d) + ' (M). ID: ' + str(i) for
                             i, d in enumerate(self.denaturant_concentrations)]

            params_names.pop()  # Remove the last one, as it is fixed to 1

        scale_factor_exclude_ids = [self.nr_den - 1] if model_scale_factor else []

        # Do a prefit with a reduced dataset
        kwargs = {

            'list_of_temperatures' : self.temp_lst_expanded_subset,
            'list_of_signals' : self.signal_lst_expanded_subset,
            'signal_ids' : self.signal_ids,
            'denaturant_concentrations': self.denaturant_concentrations_expanded,
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
        Create a dataframe with three columns: Temperature, Signal, and Denaturant.
        Optimized for speed by avoiding per-curve DataFrame creation.

        Parameters
        ----------
        signal_type : {'raw', 'fitted', 'derivative'}, optional
            Which signal to include in the dataframe. 'raw' uses experimental data, 'fitted' uses model predictions,
            'derivative' uses the estimated derivative signal.
        scaled : bool, optional
            If True and signal_type == 'fitted' or 'raw', use the scaled versions if available.
        """

        # Flatten all arrays and repeat denaturant values accordingly

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
            np.full_like(temp_lst[i], self.denaturant_concentrations[i], dtype=np.float64)
            for i in range(len(temp_lst))
        ])

        # Add an ID column, so we can identify the curves, even with the same denaturant concentration
        id_all = np.concatenate([
            np.full_like(temp_lst[i], i, dtype=np.int32)
            for i in range(len(temp_lst))
        ])

        signal_df = pd.DataFrame({
            'Temperature': temp_all,
            'Signal': signal_all,
            'Denaturant': denat_all,
            'ID': id_all
        })

        return signal_df

