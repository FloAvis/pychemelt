import pytest
import numpy as np

from pychemelt.utils.fractions import (
    fn_two_state_monomer,
    fu_two_state_dimer,
    fu_two_state_trimer,
    fu_two_state_tetramer
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
        K = 0.0
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



