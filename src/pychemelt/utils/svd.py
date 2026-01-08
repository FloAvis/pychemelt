"""
Module containing functions to perform Singular Value Decomposition (SVD) and Principal Component Analysis (PCA)
on spectral data, along with utilities for manipulating basis spectra and coefficients.

Author: Osvaldo Burastero
"""

import numpy as np


def apply_svd(X):
    """
    Perform Singular Value Decomposition (SVD) on the input data matrix X.

    Parameters
    ----------
        X : numpy array of shape (n_wavelengths, n_measurements)
            The input data matrix to decompose.

    Returns
    -------
        explained_variance : numpy array
            The cumulative explained variance for each component.
        basis_spectra     : numpy array
            The left singular vectors (U matrix) representing the basis spectra.
        coefficients      : numpy array
            The coefficients associated with each basis spectrum.
    """

    U, S, _ = np.linalg.svd(X)

    # Calculate the total variance or correlation
    total_variance = np.sum(S ** 2)
    cumulative_variance = np.cumsum(S ** 2)

    # The matrix V contains the variation of each component against the temperature / measurement dimension

    a_is = []

    for i in range(U.shape[1]):
        def coefficients_bi(column):
            return U[:, i].dot(column)

        a_i = np.apply_along_axis(coefficients_bi, axis=0, arr=X)

        a_is.append(a_i)

    coefficients = np.array(a_is)

    # Basis spectra
    basis_spectra = U

    # Cumulated explained variance of the components
    explained_variance = cumulative_variance / total_variance * 100

    return explained_variance, basis_spectra, coefficients


def filter_basis_spectra(explained_variance, basis_spectra_all, coefficients_all, explained_variance_threshold=99):
    """
    Filter the basis spectra and coefficients based on the explained variance threshold
    Parameters
    ----------
        explained_variance         : numpy array
                                     The cumulative explained variance for each component.
        basis_spectra_all         : numpy array
                                     The left singular vectors (U matrix) representing the basis spectra.
        coefficients_all          : numpy array
                                     The coefficients associated with each basis spectrum.
        explained_variance_threshold : float, optional
                                     The threshold for explained variance to filter components. Default is 99.
    Returns
    -------
        basis_spectra  : numpy array
                         The filtered basis spectra.
        coefficients   : numpy array
                         The filtered coefficients.
        k              : int
                         The number of components that meet the explained variance threshold.
    """

    # Find the number of components (k) that capture at least threshold*100 percent of the variance or correlation
    k = np.sum(explained_variance < explained_variance_threshold) + 1

    basis_spectra = basis_spectra_all[:, :k]
    coefficients = coefficients_all[:k, :]

    return basis_spectra, coefficients, k


def align_basis_spectra_and_coefficients(X, basis_spectra, coefficients):
    """
    Align the basis spectra peaks to the original data
    Parameters
    ----------
        X              : numpy array of shape (n_samples, n_features)
                         The input data matrix.
        basis_spectra  : numpy array
                         The basis spectra obtained from SVD.
        coefficients   : numpy array
                         The coefficients associated with each basis spectrum.
    Returns
    -------
        basis_spectra  : numpy array
                         The aligned basis spectra.
        coefficients   : numpy array
                         The adjusted coefficients.

    """

    # Align basis spectra peaks to the original data
    # In other words, we want that if the original spectra has a peak with positive values of the CD signal,
    # so does our basis spectra

    # Fix the n_cutoff to remove the first n and last n rows of X before finding the peak

    n_cutoff = 5

    maxV_abs = np.abs(np.max(X[n_cutoff:-n_cutoff, :]))
    minV_abs = np.abs(np.min(X[n_cutoff:-n_cutoff, :]))

    positive_peak = maxV_abs > minV_abs

    k = basis_spectra.shape[1]

    for i in range(k):

        prcomp_i = basis_spectra[:, i]

        maxV_abs_prcomp_i = np.abs(np.max(prcomp_i[n_cutoff:-n_cutoff]))
        minV_abs_prcomp_i = np.abs(np.min(prcomp_i[n_cutoff:-n_cutoff]))

        positive_peak_prcomp_i = maxV_abs_prcomp_i > minV_abs_prcomp_i

        if positive_peak_prcomp_i != positive_peak:
            coeff_i = coefficients[i, :]

            basis_spectra[:, i] = - prcomp_i
            coefficients[i, :] = - coeff_i

    return basis_spectra, coefficients


def angle_from_cathets(adjacent_leg, opposite_leg):
    """
    Calculate the angle between the hypotenuse and the adjacent leg of a right triangle.
    Parameters
    ----------
        adjacent_leg : float
            Length of the adjacent leg.
        opposite_leg : float
            Length of the opposite leg.
    Returns
    -------
        angle_in_radians : float
            Angle in radians between the hypotenuse and the adjacent leg.
    """

    hypotenuse = np.sqrt(adjacent_leg ** 2 + opposite_leg ** 2)

    return np.arccos(adjacent_leg / hypotenuse)


def get_2d_counterclockwise_rot_matrix(angle_in_radians):
    """
    Obtain the rotation matrix for a 2d coordinates system using a counterclockwise direction
    Parameters
    ----------
        angle_in_radians : float
            Angle in radians for the rotation.
    Returns
    -------
        rotM : numpy array
            2x2 rotation matrix.
    """

    rotM = np.array([[np.cos(angle_in_radians), np.sin(angle_in_radians)],
                     [-np.sin(angle_in_radians), np.cos(angle_in_radians)]])

    return rotM


def get_3d_counterclockwise_rot_matrix_around_z_axis(angle_in_radians):
    """
    Obtain the rotation matrix for a 3d coordinates system around the z axis using a counterclockwise direction
    Parameters
    ----------
        angle_in_radians : float
            Angle in radians for the rotation.
    Returns
    -------
        rotM : numpy array
            3x3 rotation matrix.
    """

    rotM = np.array([[np.cos(angle_in_radians), np.sin(angle_in_radians), 0],
                     [-np.sin(angle_in_radians), np.cos(angle_in_radians), 0],
                     [0, 0, 1]])

    return rotM


def get_3d_clockwise_rot_matrix_around_y_axis(angle_in_radians):
    """
    Obtain the rotation matrix for a 3d coordinates system around the y axis using a clockwise direction
    Parameters
    ----------
        angle_in_radians : float
            Angle in radians for the rotation.
    Returns
    -------
        rotM : numpy array
            3x3 rotation matrix.
    """

    rotM = np.array([[np.cos(angle_in_radians), 0, np.sin(angle_in_radians)],
                     [0, 1, 0],
                     [-np.sin(angle_in_radians), 0, np.cos(angle_in_radians)]])

    return rotM


def rotate_two_basis_spectra(X, basis_spectra, pca_based=False):
    """
    Create a new basis spectra using a linear combination of the first and second basis spectra

    Parameters
    ----------
        X : numpy array
            The raw data matrix of size n*m, where 'n' is the number of measured wavelengths
            and 'm' is the number of acquired spectra.
        basis_spectra : numpy array
            The matrix containing the set of basis spectra.
        pca_based : bool, optional
            Boolean to decide if we need to center the matrix X. Default is False.

    Returns
    -------
        basis_spectra_new : numpy array
            The new set of basis spectra.
        coefficients : numpy array
            The new set of associated coefficients.
    """

    if pca_based:
        X_mean = np.mean(X, axis=1, keepdims=True)
        X = X - X_mean

    first_spectrum = X[:, 0]

    c1 = first_spectrum.dot(basis_spectra[:, 0])
    c2 = first_spectrum.dot(basis_spectra[:, 1])

    rotAngle = angle_from_cathets(c1, c2)

    rotM = get_2d_counterclockwise_rot_matrix(rotAngle)

    basis_spectra_new = np.dot(basis_spectra[:, :2], rotM)
    coefficients = np.dot(basis_spectra_new.T, X)

    return basis_spectra_new, coefficients


def rotate_three_basis_spectra(X, basis_spectra, pca_based=False):
    """
    Create a new basis spectra using a linear combination from the first, second and third basis spectra

    Parameters
    ----------
        X : numpy array
            The raw data matrix of size n*m, where 'n' is the number of measured wavelengths
            and 'm' is the number of acquired spectra.
        basis_spectra : numpy array
            The matrix containing the set of basis spectra.
        pca_based : bool, optional
            Boolean to decide if we need to center the matrix X. Default is False.

    Returns
    -------
        basis_spectra_new : numpy array
            The new set of basis spectra.
        coefficients_subset : numpy array
            The new set of associated coefficients.

    """

    if pca_based:
        X_mean = np.mean(X, axis=1, keepdims=True)
        X = X - X_mean

    first_spectrum = X[:, 0]

    c1 = first_spectrum.dot(basis_spectra[:, 0])
    c2 = first_spectrum.dot(basis_spectra[:, 1])
    c3 = first_spectrum.dot(basis_spectra[:, 2])

    zAngle = angle_from_cathets(c1, c2)
    yAngle = angle_from_cathets(np.sqrt(c1 ** 2 + c2 ** 2), c3)

    rotZaxis = get_3d_counterclockwise_rot_matrix_around_z_axis(zAngle)
    rotYaxis = get_3d_clockwise_rot_matrix_around_y_axis(yAngle)

    basis_z_rot = np.dot(basis_spectra[:, :3], rotZaxis)
    basis_spectra_new = np.dot(basis_z_rot, rotYaxis)
    coefficients = np.dot(basis_spectra_new.T, X)

    return basis_spectra_new, coefficients


def reconstruct_spectra(basis_spectra, coefficients, X=None, pca_based=False):
    """
    Reconstruct the original spectra based on the set of basis spectra and the associated coefficients

    Parameters
    ----------
        basis_spectra       : numpy array
            The matrix containing the set of basis spectra.
        coefficients        : numpy array
            The associated coefficients of each basis spectrum.
        X                   : numpy array, optional
            Only used if pca_based equals TRUE!
            X is the raw data matrix of size n*m, where
            'n' is the number of measured wavelengths and
            'm' is the number of acquired spectra.
        pca_based           : bool, optional
            Boolean to decide if we need to extract the mean from the the X raw data matrix. Default is False.

        Returns
        -------
        fitted : numpy array
            The reconstructed matrix which should be close the original raw data.
    """

    fitted = (basis_spectra @ coefficients)

    # Add the mean, if needed
    if pca_based:
        X_mean = np.mean(X, axis=1, keepdims=True)
        fitted = fitted + X_mean

    return fitted


def explained_variance_from_orthogonal_vectors(vectors, coefficients, total_variance):
    """
    Useful to get the percentage of variance, not in the coordinate space provided by PCA/SVD,
    but against a different set of (rotated) vectors.

    Parameters
    ----------
        vectors        : numpy array
            The set of orthogonal vectors.
        coefficients   : numpy array
            The associated coefficients of each orthogonal vector.
        total_variance : float
            The total variance of the original data (mean subtracted if we performed PCA...).

    Returns
    -------
        explained_variance : list
            The amount of explained variance by each orthogonal vector.
    """

    explained_variance = []

    for i in range(vectors.shape[1]):
        a = np.linalg.norm(coefficients[i, :]) ** 2
        b = np.linalg.norm(vectors[:, i]) ** 2

        explained_variance.append(a / b)

    return 100 * np.cumsum(explained_variance) / total_variance


def apply_pca(X):
    """
    Perform Principal Component Analysis (PCA) on the input data matrix X.
    Parameters
    ----------
        X : numpy array of shape (n_wavelengths, n_measurements)
            The input data matrix to decompose.
    Returns
    -------
        cum_sum_eigenvalues : numpy array
            The cumulative explained variance for each principal component.
        principal_components : numpy array
            The principal components (eigenvectors) representing the basis spectra.
        coefficients : numpy array
            The coefficients associated with each principal component.
    """

    X = X.T  # We need to transpose X to have samples as rows and features as columns

    X_mean = np.mean(X, axis=0)
    X = X - X_mean

    # compute the covariance matrix
    cov_mat = np.cov(X, rowvar=False)

    # find the eigen vectors and associated eigen values
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # sort the eigenvalues in descending order
    sorted_index = np.argsort(eigen_values)[::-1]

    sorted_eigenvalue = eigen_values[sorted_index]

    # similarly sort the eigenvectors
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # compute the total variance
    total_eigenvalues = np.sum(sorted_eigenvalue)

    # compute the explained variance
    exp_var_pca = (sorted_eigenvalue / total_eigenvalues * 100)

    # compute the cumulative explained variance
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    principal_components = sorted_eigenvectors

    a_is = []

    for i in range(principal_components.shape[1]):
        def coefficients_bi(column):
            # Your custom logic here
            return principal_components[:, i].dot(column)

        a_i = np.apply_along_axis(coefficients_bi, axis=1, arr=X)

        a_is.append(a_i)

    coefficients = np.array(a_is)

    return cum_sum_eigenvalues, principal_components, coefficients


def recalc_explained_variance(basis_spectra, coefficients, X, pca_based=False):
    """
    Recalculate the explained variance of a set of basis spectra and associated coefficients
    Parameters
    ----------
        basis_spectra : numpy array
                         The basis spectra.
        coefficients  : numpy array
                         The associated coefficients of each basis spectrum.
        X             : numpy array
                         The raw data matrix of size n*m, where 'n' is the number of measured wavelengths
                         and 'm' is the number of acquired spectra.
        pca_based     : bool, optional
                         Boolean to decide if we need to center the matrix X. Default is False.
    Returns
    -------
        explained_variance : numpy array
                             The cumulative explained variance for each component.
    """

    if pca_based:
        X_mean = np.mean(X, axis=1, keepdims=True)
        X = X - X_mean

    total_variance = np.linalg.norm(X) ** 2  # Total variance in the data

    explained_variance = explained_variance_from_orthogonal_vectors(
        basis_spectra,
        coefficients,
        total_variance)

    return explained_variance