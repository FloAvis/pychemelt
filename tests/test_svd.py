import numpy as np

from pychemelt.utils.svd import (
    apply_svd,
    align_basis_spectra_and_coefficients,
    reconstruct_spectra,
    filter_basis_spectra,
    rotate_two_basis_spectra,
    rotate_three_basis_spectra
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

X_2 = (A[:, :2] @ np.vstack([B1, B2])).T
X_3 = (A @ np.vstack([B1, B2, B3])).T


def test_apply_svd():
    explained_variance, basis_spectra, coefficients = apply_svd(X_3)

    # Check the coefficients shape
    assert coefficients.shape == X_3.shape

    # Verify that the reconstructed data matches the original data
    X_reconstructed = reconstruct_spectra(basis_spectra, coefficients)

    assert np.allclose(X_3, X_reconstructed)

    # We expect three significant components
    assert explained_variance[1] < 100
    assert explained_variance[2] == 100

    basis_aligned, _ = align_basis_spectra_and_coefficients(
        X_3, basis_spectra, coefficients
    )

    # Check that they where correctly aligned
    basis_1_is_positive = np.abs(np.max(basis_aligned[:, 0])) > np.abs(np.min(basis_aligned[:, 0]))

    assert basis_1_is_positive


def test_filter_basis_spectra():
    explained_variance, basis_spectra, coefficients = apply_svd(X_2)

    _, _, k = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    assert k == 2

    explained_variance, basis_spectra, coefficients = apply_svd(X_3)

    _, _, k = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    assert k == 3


def test_rotate_spectra():
    # Rotate two basis spectra
    explained_variance, basis_spectra, coefficients = apply_svd(X_2)

    basis_spectra_new, _, k = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    basis_spectra_rot, coefficients_rot = rotate_two_basis_spectra(X_2, basis_spectra_new)

    # Verify that the reconstructed data still matches the original data
    X_reconstructed = reconstruct_spectra(basis_spectra_rot, coefficients_rot)

    assert np.allclose(X_2, X_reconstructed)

    # Rotate three basis spectra
    explained_variance, basis_spectra, coefficients = apply_svd(X_3)

    basis_spectra_new, _, k = filter_basis_spectra(explained_variance, basis_spectra, coefficients, 100)

    basis_spectra_rot, coefficients_rot = rotate_three_basis_spectra(X_3, basis_spectra_new)

    # Verify that the reconstructed data still matches the original data
    X_reconstructed = reconstruct_spectra(basis_spectra_rot, coefficients_rot)

    assert np.allclose(X_3, X_reconstructed)