"""
Main class to handle thermal and chemical denaturation data
The current model assumes the protein is a monomer and that the unfolding is reversible
"""

import pandas as pd
import numpy as np

from .utils.files  import (

    detect_file_type,
    load_nanoDSF_xlsx,
    load_panta_xlsx,
    load_uncle_multi_channel,
    load_thermofluor_xlsx,
    load_quantstudio_txt,
    load_mx3005p_txt,
    load_supr_dsf,
    load_csv_file,
    load_aunty_xlsx

)

from .utils.math  import (
    temperature_to_celsius,
    is_evenly_spaced,
    first_derivative_savgol,
    constant_baseline,
    linear_baseline,
    quadratic_baseline,
    exponential_baseline
)

from .utils.processing import (
    subset_data,
    guess_Tm_from_derivative,
    clean_conditions_labels,
    subset_signal_by_temperature,
    estimate_signal_baseline_params
)


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

        self.signal_ids = None

        self.t_melting_init_multiple = None # Initial guess for tm based on derivative

        self.single_fit_done = False  # Individual thermodynamic parameters, baselines and slopes

        self.global_fit_done = False  # Global thermodynamic parameters, local baselines and slopes

        self.global_global_fit_done = False  # Global thermodynamic parameters, global slopes and local baselines

        self.global_global_global_fit_done = False  # Global thermodynamic parameters, global slopes and global baselines

        self.global_fit_params = None

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
            'csv': load_csv_file,
            'aunty': load_aunty_xlsx
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
        
        if hasattr(self, "predicted_lst_multiple"):
            # Calculate derivative of fitted function
            predicted_deriv_lst = []

            for i in range(self.nr_signals):
                signal_deriv = []

                for j in range(self.nr_den):
                    # Get fitted values and temperature
                    idx = i * self.nr_den + j
                    x = self.temp_lst_expanded[idx]
                    y_fit = self.predicted_lst_multiple[i][j]

                    # Calculate derivative using Savitzky-Golay filter
                    # window_length should be odd and >= 3
                    window = 9
                    deriv = first_derivative_savgol(x, y_fit, window_length)

                    signal_deriv.append(deriv)

                predicted_deriv_lst.append(signal_deriv)

            # Store in the object for plotting
            self.predicted_deriv_lst_multiple = predicted_deriv_lst

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

        #Creating 
        if not hasattr(self, "deriv_lst_multiple"):
            self.estimate_derivative()

        self.deriv_lst_expanded = []

        for i in range(len(self.signal_lst_multiple)):
            self.signal_lst_expanded += self.signal_lst_multiple[i]
            self.temp_lst_expanded += self.temp_lst_multiple[i]
            self.deriv_lst_expanded += self.deriv_lst_multiple[i]

        # Create a reduced dataset for faster fitting
        self.signal_lst_expanded_subset = [subset_data(x, 60) for x in self.signal_lst_expanded]
        self.temp_lst_expanded_subset = [subset_data(x, 60) for x in self.temp_lst_expanded]
        self.deriv_lst_expanded_subset = [subset_data(x, 60) for x in self.deriv_lst_expanded]

        if self.max_points is not None:

            self.signal_lst_expanded = [subset_data(x, self.max_points) for x in self.signal_lst_expanded]
            self.temp_lst_expanded = [subset_data(x, self.max_points) for x in self.temp_lst_expanded]
            self.deriv_lst_expanded = [subset_data(x, self.max_points) for x in self.deriv_lst_expanded]

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