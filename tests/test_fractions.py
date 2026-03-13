import pytest
import numpy as np

from pychemelt.utils.fractions import (
    fn_two_state_monomer,
    fu_two_state_dimer,
    fu_two_state_trimer,
    fu_two_state_tetramer,
    fi_three_state_tetramer_monomeric_intermediate,
    fi_three_state_dimer_monomeric_intermediate,
    fu_three_state_dimer_dimeric_intermediate,
    fi_three_state_dimer_dimeric_intermediate,
    fi_three_state_trimer_monomeric_intermediate,
    fu_three_state_trimer_trimeric_intermediate,
    fi_three_state_trimer_trimeric_intermediate
)


class TestMonomerFraction:
    """Tests for fn_two_state_monomer function"""

    def test_monomer_fully_folded(self):
        """When K=0 (strongly favors native), fraction folded should be 1"""
        K = 0.0
        fn = fn_two_state_monomer(K)
        assert fn == pytest.approx(1.0)

    def test_monomer_fully_unfolded(self):
        """When K is very large (strongly favors unfolded), fraction folded should approach 0"""
        K = 1e10
        fn = fn_two_state_monomer(K)
        assert fn == pytest.approx(0.0, abs=1e-9)

    def test_monomer_equal_populations(self):
        """When K=1 (equal populations), fraction folded should be 0.5"""
        K = 1.0
        fn = fn_two_state_monomer(K)
        assert fn == pytest.approx(0.5)

    def test_monomer_typical_values(self):
        """Test with typical equilibrium constant values"""
        # K=0.1 should give fn ≈ 0.909
        assert fn_two_state_monomer(0.1) == pytest.approx(0.909, abs=1e-3)

        # K=10 should give fn ≈ 0.091
        assert fn_two_state_monomer(10.0) == pytest.approx(0.091, abs=1e-3)

        # K=0.25 should give fn = 0.8
        assert fn_two_state_monomer(0.25) == pytest.approx(0.8)

    def test_monomer_array_input(self):
        """Test that function works with numpy arrays"""
        K = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
        fn = fn_two_state_monomer(K)

        expected = np.array([1.0, 2/3, 0.5, 1/3, 1/11])
        np.testing.assert_allclose(fn, expected, rtol=1e-10)


class TestDimerFraction:
    """Tests for fu_two_state_dimer function (N2 <-> 2U)"""

    def test_dimer_fully_folded(self):
        """When K=0, all dimer should be folded, fu=0"""
        K = 1e-30
        C = 1e-6  # 1 µM
        fu = fu_two_state_dimer(K, C)
        assert fu == pytest.approx(0.0, abs=1e-10)

    def test_dimer_fully_unfolded(self):
        """When K is very large, dimer should be fully unfolded, fu→1"""
        K = 1e10
        C = 1e-6
        fu = fu_two_state_dimer(K, C)
        assert fu == pytest.approx(1.0, abs=1e-6)

    def test_dimer_typical_values(self):
        """Test with typical K and concentration values"""
        # Test case 1: Moderate K and concentration
        K = 1e-2
        C = 1
        fu = fu_two_state_dimer(K, C)
        # For dimer: fu² = K/(4C(1-fu)) -> should be partially unfolded
        assert 0.0 < fu < 1.0


    def test_dimer_concentration_dependence(self):
        """Test that higher concentration shifts equilibrium toward folded state"""
        K = 1e-6
        C_low = 1e-7
        C_high = 1e-5

        fu_low = fu_two_state_dimer(K, C_low)
        fu_high = fu_two_state_dimer(K, C_high)

        # Higher concentration should favor folded state (lower fu)
        assert fu_low > fu_high

    def test_dimer_array_input(self):
        """Test that function works with numpy arrays"""
        K = 1e-6
        C = np.array([1e-7, 1e-6, 1e-5])
        fu = fu_two_state_dimer(K, C)

        assert len(fu) == len(C)
        assert np.all(fu >= 0) and np.all(fu <= 1)


class TestTrimerFraction:
    """Tests for fu_two_state_trimer function (N3 <-> 3U)"""

    def test_trimer_fully_folded(self):
        """When K=0, all trimer should be folded, fu=0"""
        K = 0.00000000000000000000000000000000000000000000001
        C = 1e-6
        fu = fu_two_state_trimer(K, C)
        assert fu == pytest.approx(0.0, abs=1e-10)

    def test_trimer_fully_unfolded(self):
        """When K is very large, trimer should be fully unfolded, fu→1"""
        K = 1e15
        C = 1e-6
        fu = fu_two_state_trimer(K, C)
        assert fu == pytest.approx(1.0, abs=1e-3)

    def test_trimer_typical_values(self):
        """Test with typical K and concentration values"""
        K = 1e-12
        C = 1e-6
        fu = fu_two_state_trimer(K, C)

        # Should be partially unfolded
        assert 0.0 < fu < 1.0


    def test_trimer_concentration_dependence(self):
        """Test that higher concentration shifts equilibrium toward folded state"""
        K = 1e-12
        C_low = 1e-7
        C_high = 1e-5

        fu_low = fu_two_state_trimer(K, C_low)
        fu_high = fu_two_state_trimer(K, C_high)

        # Higher concentration should favor folded state (lower fu)
        assert fu_low > fu_high

    def test_trimer_array_input(self):
        """Test that function works with numpy arrays"""
        K = 1e-12
        C = np.array([1e-7, 1e-6, 1e-5])
        fu = fu_two_state_trimer(K, C)

        assert len(fu) == len(C)
        assert np.all(fu >= 0) and np.all(fu <= 1)


class TestTetramerFraction:
    """Tests for fu_two_state_tetramer function (N4 <-> 4U)"""

    def test_tetramer_fully_folded(self):
        """When K=0, all tetramer should be folded, fu=0"""
        K = 0.0
        C = 1e-6
        fu = fu_two_state_tetramer(K, C)
        assert fu == pytest.approx(0.0, abs=1e-10)

    def test_tetramer_fully_unfolded(self):
        """When K is very large, tetramer should be fully unfolded, fu→1"""
        K = 1e20
        C = 1e-6
        fu = fu_two_state_tetramer(K, C)
        assert fu == pytest.approx(1.0, abs=1e-2)

    def test_tetramer_typical_values(self):
        """Test with typical K and concentration values"""
        K = 1e-18
        C = 1e-6
        fu = fu_two_state_tetramer(K, C)

        # Should be partially unfolded
        assert 0.0 < fu < 1.0

        # Verify the equilibrium relationship: K = 256*C³*fu⁴/(1-fu)⁴
        fn = 1 - fu
        if fn > 0:
            K_calc = 256 * C**3 * fu**4 / (fn**4)
            assert K_calc == pytest.approx(K, rel=1e-2)

    def test_tetramer_concentration_dependence(self):
        """Test that higher concentration shifts equilibrium toward folded state"""
        K = 1e-18
        C_low = 1e-7
        C_high = 1e-5

        fu_low = fu_two_state_tetramer(K, C_low)
        fu_high = fu_two_state_tetramer(K, C_high)

        # Higher concentration should favor folded state (lower fu)
        assert fu_low > fu_high

    def test_tetramer_array_input(self):
        """Test that function works with numpy arrays"""
        K = 1e-18
        C = np.array([1e-7, 1e-6, 1e-5])
        fu = fu_two_state_tetramer(K, C)

        assert len(fu) == len(C)
        # All values should be valid fractions (allowing for numerical edge cases)
        assert np.all((fu >= 0) & (fu <= 1.01))

    def test_tetramer_numerical_stability(self):
        """Test that tetramer calculation handles edge cases properly"""
        # Very small K
        fu = fu_two_state_tetramer(1e-50, 1e-6)
        assert fu == pytest.approx(0.0, abs=1e-5)

        # Moderate values that might cause numerical issues
        K = np.array([1e-20, 1e-18, 1e-16])
        C = 1e-6
        fu = fu_two_state_tetramer(K, C)

        # Should all be valid fractions
        assert np.all((fu >= 0) & (fu <= 1.01))
        # Should be monotonically increasing with K
        assert np.all(np.diff(fu) > 0)


class TestThreeStateFractions:
    """Tests for three-state fraction functions"""

    def test_tetramer_monomeric_intermediate(self):
        """Test fi_three_state_tetramer_monomeric_intermediate (N4 <-> 4I <-> 4U)"""
        Ct = 1e-6
        # When K1 is very small,so fi should be small
        assert fi_three_state_tetramer_monomeric_intermediate(1e-50, 1.0, Ct) == pytest.approx(0.0, abs=1e-9)

        # When K1 is very large and K2 is small, so fi should be large (~1)
        K2 = 0.1
        # Use a large but not extremely large K1 to avoid numerical issues
        fi = fi_three_state_tetramer_monomeric_intermediate(1e5, K2, Ct)
        assert fi == pytest.approx(1 / (1 + K2), rel=1e-5)

    def test_dimer_monomeric_intermediate(self):
        """Test fi_three_state_dimer_monomeric_intermediate (N2 <-> 2I <-> 2U)"""
        C = 1e-6
        # When K1 is small, mostly N2, fi should be small
        assert fi_three_state_dimer_monomeric_intermediate(1e-30, 1.0, C) == pytest.approx(0.0, abs=1e-10)

        # When K1 is large, 4*C*fi**2 + K1*(1+K2)*fi - K1 = 0
        # fi = (-K1(1+K2) + sqrt(K1**2(1+K2)**2 + 16*C*K1)) / (8*C)
        # As K1 -> inf, fi -> 1/(1+K2)
        K2 = 0.5
        fi = fi_three_state_dimer_monomeric_intermediate(1e10, K2, C)
        assert fi == pytest.approx(1 / (1 + K2), rel=1e-5)

    def test_dimer_dimeric_intermediate(self):
        """Test dimer with dimeric intermediate (N2 <-> I2 <-> 2U)"""
        C = 1e-6
        K1 = 1.0
        K2 = 1e-6

        # When K2 is small, fu should be small
        fu = fu_three_state_dimer_dimeric_intermediate(K1, K2, C)

        # fu = (-1 + sqrt(1 + 32)) / 16 = (-1 + sqrt(33)) / 16 ≈ 0.296
        expected_fu = (-1 + np.sqrt(33)) / 16
        assert fu == pytest.approx(expected_fu, rel=1e-5)

        # When K2 is very large, fu should be large
        fu_large = fu_three_state_dimer_dimeric_intermediate(K1, 1e10, C)
        assert fu_large == pytest.approx(1.0, rel=1e-5)

        # Test fi calculation from fu
        fi = fi_three_state_dimer_dimeric_intermediate(fu_large, 1e10, C)
        # fi = 4 * fu**2 * C / K2
        assert fi == pytest.approx(4 * 1.0**2 * C / 1e10)

    def test_trimer_monomeric_intermediate(self):
        """Test fi_three_state_trimer_monomeric_intermediate (N3 <-> 3I <-> 3U)"""
        C = 1e-6
        # When K1 is small, fi should be small
        assert fi_three_state_trimer_monomeric_intermediate(1e-50, 1.0, C) == pytest.approx(0.0, abs=1e-10)

        # When K1 is large, fi should approach 1/(1+K2)
        K2 = 0.2
        fi = fi_three_state_trimer_monomeric_intermediate(1e20, K2, C)
        assert fi == pytest.approx(1 / (1 + K2), rel=1e-5)

    def test_trimer_trimeric_intermediate(self):
        """Test trimer with trimeric intermediate (N3 <-> I3 <-> 3U)"""
        C = 1e-6
        K1 = 1.0

        # fu**3 + 1.85e-10*fu - 1.85e-10 = 0 => fu ≈ (1.85e-10)^(1/3) ≈ 0.000569

        fu = fu_three_state_trimer_trimeric_intermediate(K1, 1e-20, C)
        p = 1e-20 / (27 * C**2 * (1 + K1))
        # Solve fu**3 + p*fu - p = 0. For small p, fu ≈ p**(1/3)
        assert fu == pytest.approx(p**(1/3), rel=1e-2)

        # When K2 is large, fu should be large
        fu_large = fu_three_state_trimer_trimeric_intermediate(K1, 1e15, C)
        assert fu_large == pytest.approx(1.0, rel=1e-3)

        # Test fi calculation from fu
        fi = fi_three_state_trimer_trimeric_intermediate(fu_large, 1e15, C)
        # fi = 27 * C**2 * fu**3 / K2
        assert fi == pytest.approx(27 * C**2 * 1.0**3 / 1e15, rel=1e-3)



