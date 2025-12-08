def estimate_baseline_parameters_automatic(self, window_range_native=12, window_range_unfolded=12):
    # Count the number the linear model is better than the quadratic model
    linear_count_native = 0
    linear_count_unfolded = 0

    for s, t in zip(self.signal_lst, self.temp_lst):
        s_n = s[t < np.min(t) + window_range_native]
        t_n = t[t < np.min(t) + window_range_native]

        linear_count_native += compare_linear_to_quadratic(t_n, s_n)

        s_u = s[t > np.max(t) - window_range_unfolded]
        t_u = t[t > np.max(t) - window_range_unfolded]

        linear_count_unfolded += compare_linear_to_quadratic(t_u, s_u)

    poly_order_native = 1 if linear_count_native >= (self.nr_den / 2) else 2
    poly_order_unfolded = 1 if linear_count_unfolded >= (self.nr_den / 2) else 2

    self.estimate_baseline_parameters(
        window_range_native, window_range_unfolded,
        poly_order_native, poly_order_unfolded
    )

    return None


def compare_models(self, window_range_native=12, window_range_unfolded=12,
                   exclude_ids_lst=None):
    """
    Fit the thermal unfolding curves using four different models:

        Linear dependence for the native state - Linear dependence for the unfolded state
        Linear dependence for the native state - Quadratic dependence for the unfolded state
        Quadratic dependence for the native state - Linear dependence for the unfolded state
        Quadratic dependence for the native state - Quadratic dependence for the unfolded state

    Then compare them using the akaike criterion

    Parameters
    ----------
    window_range_native : int, optional
        Range of the window (in degrees) to estimate the baselines and slopes of the native state
    window_range_unfolded : int, optional
        Range of the window (in degrees) to estimate the baselines and slopes of the unfolded state
    exclude_ids_lst : list of int, optional
        List of ids to exclude from the fitting
    """

    self.signal_lst_orig = self.signal_lst.copy()
    self.temp_lst_orig = self.temp_lst.copy()

    self.denaturant_concentrations_orig = self.denaturant_concentrations.copy()

    if exclude_ids_lst is not None:
        # Remove the signals and temperatures from the lists
        self.signal_lst = [x for i, x in enumerate(self.signal_lst) if i not in exclude_ids_lst]
        self.temp_lst = [x for i, x in enumerate(self.temp_lst) if i not in exclude_ids_lst]
        self.denaturant_concentrations = [x for i, x in enumerate(self.denaturant_concentrations) if
                                          i not in exclude_ids_lst]

    self.estimate_baseline_parameters(
        window_range_native,
        window_range_unfolded,
        poly_order_native=1,
        poly_order_unfolded=1
    )

    self.fit_thermal_unfolding_local()

    akaikes_1 = []

    for y, y_pred in zip(self.signal_lst, self.predicted_lst):
        # Calculate the Akaike criterion
        akaikes_1.append(compute_aic(y, y_pred, 6))

    self.estimate_baseline_parameters(
        window_range_native,
        window_range_unfolded,
        poly_order_native=1,
        poly_order_unfolded=2
    )

    self.fit_thermal_unfolding_local()

    akaikes_2 = []
    for y, y_pred in zip(self.signal_lst, self.predicted_lst):
        # Calculate the Akaike criterion
        akaikes_2.append(compute_aic(y, y_pred, 7))

    self.estimate_baseline_parameters(
        window_range_native,
        window_range_unfolded,
        poly_order_native=2,
        poly_order_unfolded=1
    )

    self.fit_thermal_unfolding_local()

    akaikes_3 = []
    for y, y_pred in zip(self.signal_lst, self.predicted_lst):
        # Calculate the Akaike criterion
        akaikes_3.append(compute_aic(y, y_pred, 7))

    self.estimate_baseline_parameters(
        window_range_native,
        window_range_unfolded,
        poly_order_native=2,
        poly_order_unfolded=2
    )

    self.fit_thermal_unfolding_local()

    akaikes_4 = []
    for y, y_pred in zip(self.signal_lst, self.predicted_lst):
        # Calculate the Akaike criterion
        akaikes_4.append(compute_aic(y, y_pred, 8))

    # Show the overall best model
    best_model = compare_akaikes(akaikes_1, akaikes_2, akaikes_3, akaikes_4, self.denaturant_concentrations)

    # Go back to the original signal and temperature lists
    self.signal_lst = self.signal_lst_orig
    self.temp_lst = self.temp_lst_orig
    self.denaturant_concentrations = self.denaturant_concentrations_orig

    return None