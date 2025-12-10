def solve_one_root_quadratic(a,b,c):
    """
    Obtain one root of the quadratic equation of the form ax^2 + bx + c = 0.

    Parameters
    ----------
    a : float
        Coefficient of x^2
    b : float
        Coefficient of x
    c : float
        Constant term

    Returns
    -------
    float
        One root of the quadratic equation
    """
    return 2*c / (-b - np.sqrt(b**2 - 4*a*c))


def solve_one_root_depressed_cubic(p,q):

    """
    Obtain one root of the depressed cubic equation of the form x^3 + p x + q = 0.

    Parameters
    ----------
    p : float
        Coefficient of x
    q : float
        Constant term

    Returns
    -------
    float
        One real root of the cubic equation
    """

    delta = np.sqrt((q**2/4) + (p**3/27))

    return np.cbrt(-q/2+delta) + np.cbrt(-q/2-delta)

def residuals_squares_sum(y_true,y_pred):

    """
    Calculate the residual sum of squares.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        Residual sum of squares
    """

    # Convert to numpy arrays if it is a list
    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    rss = np.sum((y_true - y_pred)**2)

    return rss



def r_squared(y_true, y_pred):
    """
    Calculate the R-squared value for a regression model.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values

    Returns
    -------
    float
        R-squared value
    """

    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - ss_res / ss_total


def adjusted_r2(r2, n, p):
    """
    Calculate the adjusted R-squared value for a regression model.

    Parameters
    ----------
    r2 : float
        R-squared value
    n : int
        Number of observations
    p : int
        Number of predictors

    Returns
    -------
    float
        Adjusted R-squared value
    """

    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def compute_aic(y_true, y_pred, k):
    """
    Compute the Akaike Information Criterion (AIC) for a regression model.

    Parameters
    ----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    k : int
        Number of parameters in the model

    Returns
    -------
    float
        AIC value
    """

    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    return n * np.log(rss / n) + 2 * k


def compare_akaikes(akaikes_1, akaikes_2, akaikes_3, akaikes_4, denaturant_concentrations):
    model_names = ['Linear - Linear', 'Linear - Quadratic',
                   'Quadratic - Linear', 'Quadratic - Quadratic']

    akaikes_df = pd.DataFrame({
        'Model': model_names})

    i = 0
    for a1, a2, a3, a4 in zip(akaikes_1, akaikes_2, akaikes_3, akaikes_4):
        # Create a new column with the Akaike values
        # The name is the denaturant concentration

        # Compute delta AIC
        min_aic = np.min([a1, a2, a3, a4])
        a1 = a1 - min_aic
        a2 = a2 - min_aic
        a3 = a3 - min_aic
        a4 = a4 - min_aic

        akaikes_df[str(i) + '_' + str(denaturant_concentrations[i])] = [a1, a2, a3, a4]
        i += 1

    # Find the best model for each denaturant concentration
    best_models_ids = []
    for i in range(len(denaturant_concentrations)):

        # Get the column with the Akaike values
        aic_col = akaikes_df.iloc[:, i + 1].to_numpy()

        # Find index that sort them from min to max a numpy array
        sorted_idx = np.argsort(aic_col)

        first_model_id = np.arange(4)[sorted_idx][0]
        second_model_id = np.arange(4)[sorted_idx][1]
        third_model_id = np.arange(4)[sorted_idx][2]
        fourth_model_id = np.arange(4)[sorted_idx][3]

        best_models_ids.append(first_model_id)

        # Compare the AIC value of the second model to the first one
        if aic_col[second_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(second_model_id)

        # Compare the AIC value of the third model to the first one
        if aic_col[third_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(third_model_id)

        # Compare the AIC value of the fourth model to the first one
        if aic_col[fourth_model_id] - aic_col[first_model_id] < 2:
            best_models_ids.append(fourth_model_id)

    # Print the overall best model
    best_model_all = Counter(best_models_ids).most_common(1)[0][0]
    return model_names[best_model_all]


def rss_p(rrs0, n, p, alfa):

    """
    Given the residuals of the best fitted model,
    compute the desired residual sum of squares for a 1-alpha confidence interval.
    This is used to compute asymmetric confidence intervals for the fitted parameters.

    Parameters
    ----------
    rrs0 : float
        Residual sum of squares of the model with the best fit
    n : int
        Number of data points
    p : int
        Number of parameters
    alfa : float
        Desired significance level (alpha)

    Returns
    -------
    float
        Residual sum of squares for the desired confidence interval
    """

    critical_value = stats.f.ppf(q=1 - alfa, dfn=1, dfd=n - p)

    return rrs0 * (1 + critical_value / (n - p))


def get_desired_rss(y, y_fit, p,alpha=0.05):

    """
    Given the observed and fitted data, find the residual sum of squares required for a 1-alpha confidence interval.

    Parameters
    ----------
    y : array-like
        Observed values or list of arrays
    y_fit : array-like
        Fitted values or list of arrays
    p : int
        Number of parameters
    alpha : float, optional
        Desired significance level (default: 0.05)

    Returns
    -------
    float
        Residual sum of squares corresponding to the desired confidence interval
    """

    # If y is of type list, convert it to a numpy array by concatenating
    if isinstance(y, list):
        y = np.concatenate(y,axis=0)
    # If y_fit is of type list, convert it to a numpy array by concatenating
    if isinstance(y_fit, list):
        y_fit = np.concatenate(y_fit,axis=0)

    n = len(y)

    rss = get_rss(y, y_fit)

    return rss_p(rss, n, p, alpha)

def compare_linear_to_quadratic(x,y):

    """
    Compare the linear and quadratic fits to the data using an F-test.

    Parameters
    ----------
    x : array-like
        x data
    y : array-like
        y data

    Returns
    -------
    bool
        True if the linear model is statistically preferable to the quadratic model
    """

    m, b       = fit_line_robust(x, y)
    y_pred_lin = m * x + b

    a,b,c     = fit_quadratic_robust(x, y)
    y_pred_quad = a * x ** 2 + b * x + c

    # Residual sums
    rss_lin = np.sum((y - y_pred_lin) ** 2)
    rss_quad = np.sum((y - y_pred_quad) ** 2)

    # R² and Adjusted R²
    n = len(x)
    p_lin = 1
    p_quad = 2

    # F-test
    numerator   = (rss_lin - rss_quad) / (p_quad - p_lin)
    denominator = rss_quad / (n - (p_quad + 1))
    f_stat = numerator / denominator
    p_value = 1 - f_dist.cdf(f_stat, dfn=p_quad - p_lin, dfd=n - (p_quad + 1))

    # True if linear model is better
    return p_value > 0.05

def fu_two_state_dimer(K,C):
    """
    Given the equilibrium constant K of N2 <-> 2U and the concentration of dimer equivalent C,
    return the fraction of unfolded protein.

    Parameters
    ----------
    K : float
        Equilibrium constant of the reaction N2 <-> 2U
    C : float
        Concentration of dimer equivalent

    Returns
    -------
    float
        Fraction of unfolded protein
    """

    return solve_one_root_quadratic(4*C, K, -K)

def arrhenius(T, Tf, Ea):
    """
    Arrhenius equation: defines dependence of reaction rate constant k on temperature.
    In this version of the equation we use Tf (a temperature of k=1) to avoid specifying a pre-exponential constant A.

    Parameters
    ----------
    T : array-like
        Temperature (°C or K)
    Tf : float
        Reference temperature at which the reaction rate constant equals 1 (°C or K)
    Ea : float
        Activation energy (kcal/mol)

    Returns
    -------
    numpy.ndarray
        Reaction rate constant at the given temperature
    """

    T  = temperature_to_kelvin(T)
    Tf = temperature_to_kelvin(Tf)

    return np.exp(-Ea / R_gas * (1 / T - 1 / Tf))


def fit_tc_unfolding_many_signals_slow(
        list_of_temperatures,
        list_of_signals,
        signal_ids,
        denaturant_concentrations,
        initial_parameters,
        low_bounds, high_bounds,
        signal_fx,
        fit_slope_native_temp=True,
        fit_slope_unfolded_temp=True,
        fit_slope_native_den=True,
        fit_slope_unfolded_den=True,
        fit_quadratic_native=False,
        fit_quadratic_unfolded=False,
        oligomer_concentrations=None,
        fit_m1=False,
        model_scale_factor=False,
        scale_factor_exclude_ids=[]):
    """
    Fit thermochemical unfolding curves for many signals (slow variant).

    Parameters
    ----------
    list_of_temperatures : list of array-like
        List of temperature arrays for each dataset
    list_of_signals : list of array-like
        List of signal arrays for each dataset
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
    fit_slope_native_temp : bool, optional
        Whether to fit the temperature slope of the native baseline (per-signal)
    fit_slope_unfolded_temp : bool, optional
        Whether to fit the temperature slope of the unfolded baseline (per-signal)
    fit_slope_native_den : bool, optional
        Whether to fit the denaturant slope of the native baseline (per-signal)
    fit_slope_unfolded_den : bool, optional
        Whether to fit the denaturant slope of the unfolded baseline (per-signal)
    fit_quadratic_native : bool, optional
        Whether to fit a quadratic temperature term for the native baseline (per-signal)
    fit_quadratic_unfolded : bool, optional
        Whether to fit a quadratic temperature term for the unfolded baseline (per-signal)
    oligomer_concentrations : list, optional
        Oligomer concentrations per dataset (used by oligomeric models)
    fit_m1 : bool, optional
        Whether to include and fit temperature dependence of the m-value (m1)
    model_scale_factor : bool, optional
        If True, include a per-denaturant concentration scale factor to account for intensity differences
    scale_factor_exclude_ids : list, optional
        IDs of scale factors to exclude / fix to 1 (useful to avoid fitting trivial factors)

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

    n_signals = np.max(signal_ids) + 1

    nr_den = int(len(denaturant_concentrations) / n_signals)

    if len(scale_factor_exclude_ids) > 0 and model_scale_factor:
        # Sort them in ascending order to avoid issues when inserting
        scale_factor_exclude_ids = sorted(scale_factor_exclude_ids)

    # Find if highest concentration of denaturant has a higher signal or not
    if model_scale_factor:
        den_conc_simple = denaturant_concentrations[:nr_den]

        # Find the index that sorts them in descending order from highest to lowest
        sort_indeces = np.argsort(den_conc_simple)[::-1]

        signal_first = list_of_signals[:nr_den]

        signal_sort = [signal_first[i] for i in sort_indeces]

        higher_den_equal_higher_signal = signal_sort[0][0] > signal_sort[-1][0]

    def unfolding(dummyVariable, *args):

        Tm, DHm, Cp0, m0 = args[:4]  # Enthalpy of unfolding, Temperature of melting, Cp0, m0, m1

        id_param_init = 4 + fit_m1
        m1 = args[4] if fit_m1 else 0

        # First filter, verify that DG is not lower than 0 at 5C
        # In other words, we do not have cold denaturation at 5C
        """
        Tfive = temperature_to_kelvin(5)
        TmK   = temperature_to_kelvin(Tm)

        DGfive = DHm * (1 - Tfive / TmK) + Cp0 * (Tfive - TmK - Tfive * np.log(Tfive / TmK))

        if DGfive < 0:

            return np.zeros(len(all_signal))
        """

        a_Ns = args[id_param_init:id_param_init + n_signals]
        a_Us = args[id_param_init + n_signals:id_param_init + 2 * n_signals]

        id_param_init = id_param_init + 2 * n_signals
        if fit_slope_native_temp:
            b_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            b_Ns = [0] * n_signals

        if fit_slope_unfolded_temp:
            b_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            b_Us = [0] * n_signals

        if fit_slope_native_den:
            c_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            c_Ns = [0] * n_signals

        if fit_slope_unfolded_den:
            c_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            c_Us = [0] * n_signals

        if fit_quadratic_native:
            d_Ns = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            d_Ns = [0] * n_signals

        if fit_quadratic_unfolded:
            d_Us = args[id_param_init:id_param_init + n_signals]
            id_param_init += n_signals
        else:
            d_Us = [0] * n_signals

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
            a_N = a_Ns[signal_ids[i]]
            b_N = b_Ns[signal_ids[i]]
            c_N = c_Ns[signal_ids[i]]
            d_N = d_Ns[signal_ids[i]]

            a_U = a_Us[signal_ids[i]]
            b_U = b_Us[signal_ids[i]]
            c_U = c_Us[signal_ids[i]]
            d_U = d_Us[signal_ids[i]]

            d = denaturant_concentrations[i]

            c = 0 if oligomer_concentrations is None else oligomer_concentrations[i]

            d_factor = 1

            d = d * d_factor

            y = signal_fx(
                T, d, DHm, Tm, Cp0, m0, m1,
                a_N, b_N, c_N, d_N,
                a_U, b_U, c_U, d_U, c
            )

            scale_factor = 1 if not model_scale_factor else factors[i]

            y = y * scale_factor

            signal.append(y)

        # Second filter, verify that higher_den_equal_higher_signal is same in the raw and fitted signal
        if model_scale_factor:

            signal_first = signal[:nr_den]

            signal_sort = [signal_first[i] for i in sort_indeces]

            pred_higher_den_equal_higher_signal = signal_sort[0][0] > signal_sort[-1][0]

            if pred_higher_den_equal_higher_signal != higher_den_equal_higher_signal:
                return np.zeros(len(all_signal))

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