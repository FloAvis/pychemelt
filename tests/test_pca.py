import numpy as np

from pychemelt.utils.svd import (
    apply_pca,
    reconstruct_spectra,
    filter_basis_spectra,
    rotate_two_basis_spectra,
    rotate_three_basis_spectra,
    recalc_explained_variance
)


# Generate test data
def gaussian(x, amp, ctr, std):
    return amp * np.exp(-(x - ctr) ** 2 / (2 * (std ** 2)))


x = np.linspace(100, 400, 300)

B1 = gaussian(x, 1, 210, 7)
B2 = gaussian(x, 1, 240, 12)
B3 = gaussian(x, 1, 270, 5)

A = np.array([
    [0.0, 1.0, 0.1],
    [0.3, 1.0, 0.4],
    [0.2, 0.5, 1.0],
    [0.9, 0.1, 0.2]
])

X_2 = (A[:, :2] @ np.vstack([B1, B2])).T + 5
X_3 = (A @ np.vstack([B1, B2, B3])).T + 5  # No need to transpose here, like in test_svd.py


def test_apply_pca():
    cum_sum_eigenvalues, principal_components, coefficients = apply_pca(X_3)

    # Check the coefficients shape
    assert len(coefficients) == X_3.shape[0]

    # Verify that the reconstructed data matches the original data
    X_reconstructed = reconstruct_spectra(principal_components, coefficients, X=X_3, pca_based=True)

    assert np.allclose(X_3, X_reconstructed)

    # We expect three significant components
    assert cum_sum_eigenvalues[1] < 100
    assert np.allclose(cum_sum_eigenvalues[2], 100, rtol=1e-5)


def test_filter_basis_spectra():
    explained_variance, basis_spectra, coefficients = apply_pca(X_2)

    _, _, k = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    assert k == 2

    explained_variance, basis_spectra, coefficients = apply_pca(X_3)

    _, _, k = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    assert k == 3


def test_rotate_spectra():
    # Rotate two basis spectra
    explained_variance, basis_spectra, coefficients = apply_pca(X_2)

    basis_spectra_new, _, _ = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    basis_spectra_rot, coefficients_rot = rotate_two_basis_spectra(X_2, basis_spectra_new, pca_based=True)

    # Verify that the reconstructed data still matches the original data
    X_reconstructed = reconstruct_spectra(basis_spectra_rot, coefficients_rot, X=X_2, pca_based=True)

    assert np.allclose(X_2, X_reconstructed)

    # Rotate three basis spectra
    explained_variance, basis_spectra, coefficients = apply_pca(X_3)

    basis_spectra_new, _, _ = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    basis_spectra_rot, coefficients_rot = rotate_three_basis_spectra(X_3, basis_spectra_new, pca_based=True)

    # Verify that the reconstructed data still matches the original data
    X_reconstructed = reconstruct_spectra(basis_spectra_rot, coefficients_rot, X=X_3, pca_based=True)

    assert np.allclose(X_3, X_reconstructed)


def test_variance_change_after_rotation():
    # Rotate two basis spectra
    explained_variance, basis_spectra, coefficients = apply_pca(X_2)

    basis_spectra_new, _, _ = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    basis_spectra_rot, coefficients_rot = rotate_two_basis_spectra(X_2, basis_spectra_new, pca_based=True)

    # Recalculate explained variance
    explained_variance_rot = recalc_explained_variance(basis_spectra_rot, coefficients_rot, X_2, pca_based=True)

    assert explained_variance_rot[0] < explained_variance[0]
    assert np.allclose(explained_variance_rot[1], explained_variance[1], rtol=1e-5)