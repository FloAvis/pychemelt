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

