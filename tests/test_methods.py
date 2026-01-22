"""Tests for estimation methods."""

import numpy as np
import pytest

from diwasp.methods import BDM, DFTM, EMLM, EMEP, IMLM, EstimationMethodBase
from diwasp.methods.base import compute_kx


class TestComputeKx:
    """Tests for spatial phase lag computation."""

    def test_shape(self):
        """Test output shape."""
        k = np.array([0.1, 0.2, 0.3])
        theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        sensor_x = np.array([0, 10, 20])
        sensor_y = np.array([0, 0, 10])

        kx = compute_kx(k, theta, sensor_x, sensor_y)

        assert kx.shape == (3, 36, 3)

    def test_zero_at_origin(self):
        """Sensor at origin should have zero phase lag."""
        k = np.array([0.1, 0.2])
        theta = np.array([0, np.pi / 2, np.pi])
        sensor_x = np.array([0, 10])
        sensor_y = np.array([0, 0])

        kx = compute_kx(k, theta, sensor_x, sensor_y)

        # First sensor at origin should have zero phase lag
        np.testing.assert_allclose(kx[:, :, 0], 0)

    def test_phase_proportional_to_k(self):
        """Phase lag should be proportional to wavenumber."""
        k = np.array([0.1, 0.2])
        theta = np.array([0])  # Looking in x-direction
        sensor_x = np.array([10])
        sensor_y = np.array([0])

        kx = compute_kx(k, theta, sensor_x, sensor_y)

        # kx = k * x * cos(0) = k * 10
        expected = k * 10
        np.testing.assert_allclose(kx[:, 0, 0], expected)


class TestEstimationMethodBase:
    """Tests for estimation method base class."""

    def test_is_abstract(self):
        """Base class should not be instantiable."""
        with pytest.raises(TypeError):
            EstimationMethodBase()


class TestDFTM:
    """Tests for DFTM estimation method."""

    def test_output_shape(self):
        """Test output has correct shape."""
        method = DFTM()

        n_freqs, n_dirs, n_sensors = 10, 36, 3
        csd = np.random.randn(n_freqs, n_sensors, n_sensors) + 1j * np.random.randn(
            n_freqs, n_sensors, n_sensors
        )
        # Make Hermitian
        for fi in range(n_freqs):
            csd[fi] = (csd[fi] + csd[fi].conj().T) / 2

        transfer = np.random.randn(n_freqs, n_dirs, n_sensors) + 1j * np.random.randn(
            n_freqs, n_dirs, n_sensors
        )
        kx = np.random.randn(n_freqs, n_dirs, n_sensors)

        S = method.estimate(csd, transfer, kx)

        assert S.shape == (n_freqs, n_dirs)

    def test_non_negative(self):
        """Output should be non-negative."""
        method = DFTM()

        n_freqs, n_dirs, n_sensors = 10, 36, 3
        csd = np.random.randn(n_freqs, n_sensors, n_sensors) + 1j * np.random.randn(
            n_freqs, n_sensors, n_sensors
        )
        for fi in range(n_freqs):
            csd[fi] = (csd[fi] + csd[fi].conj().T) / 2

        transfer = np.random.randn(n_freqs, n_dirs, n_sensors) + 1j * np.random.randn(
            n_freqs, n_dirs, n_sensors
        )
        kx = np.random.randn(n_freqs, n_dirs, n_sensors)

        S = method.estimate(csd, transfer, kx)

        assert np.all(S >= 0)

    def test_normalized(self):
        """Each frequency row should be normalized."""
        method = DFTM()

        n_freqs, n_dirs, n_sensors = 10, 36, 3
        # Create positive-definite CSD
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T

        transfer = np.ones((n_freqs, n_dirs, n_sensors), dtype=complex)
        kx = np.zeros((n_freqs, n_dirs, n_sensors))

        S = method.estimate(csd, transfer, kx)

        # Each row should sum to 1 (normalized)
        row_sums = np.sum(S, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)


class TestEMLM:
    """Tests for EMLM estimation method."""

    def test_output_shape(self):
        """Test output has correct shape."""
        method = EMLM()

        n_freqs, n_dirs, n_sensors = 10, 36, 3
        # Create positive-definite CSD
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T + 0.1 * np.eye(n_sensors)

        transfer = np.random.randn(n_freqs, n_dirs, n_sensors) + 1j * np.random.randn(
            n_freqs, n_dirs, n_sensors
        )
        kx = np.random.randn(n_freqs, n_dirs, n_sensors)

        S = method.estimate(csd, transfer, kx)

        assert S.shape == (n_freqs, n_dirs)

    def test_non_negative(self):
        """Output should be non-negative."""
        method = EMLM()

        n_freqs, n_dirs, n_sensors = 5, 18, 3
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T + 0.1 * np.eye(n_sensors)

        transfer = np.ones((n_freqs, n_dirs, n_sensors), dtype=complex)
        kx = np.zeros((n_freqs, n_dirs, n_sensors))

        S = method.estimate(csd, transfer, kx)

        assert np.all(S >= 0)


class TestIMLM:
    """Tests for IMLM estimation method."""

    def test_parameters(self):
        """Test custom parameters."""
        method = IMLM(max_iter=50, gamma=0.2, alpha=0.2)

        assert method.max_iter == 50
        assert method.gamma == 0.2
        assert method.alpha == 0.2

    def test_output_shape(self):
        """Test output has correct shape."""
        method = IMLM(max_iter=10)

        n_freqs, n_dirs, n_sensors = 5, 18, 3
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T + 0.1 * np.eye(n_sensors)

        transfer = np.ones((n_freqs, n_dirs, n_sensors), dtype=complex)
        kx = np.zeros((n_freqs, n_dirs, n_sensors))

        S = method.estimate(csd, transfer, kx)

        assert S.shape == (n_freqs, n_dirs)


class TestEMEP:
    """Tests for EMEP estimation method."""

    def test_output_shape(self):
        """Test output has correct shape."""
        method = EMEP(max_iter=10)

        n_freqs, n_dirs, n_sensors = 5, 36, 3
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T + 0.1 * np.eye(n_sensors)

        transfer = np.ones((n_freqs, n_dirs, n_sensors), dtype=complex)
        kx = np.zeros((n_freqs, n_dirs, n_sensors))

        S = method.estimate(csd, transfer, kx)

        assert S.shape == (n_freqs, n_dirs)


class TestBDM:
    """Tests for BDM estimation method."""

    def test_output_shape(self):
        """Test output has correct shape."""
        method = BDM(max_iter=10, n_reg_steps=5)

        n_freqs, n_dirs, n_sensors = 5, 36, 3
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T + 0.1 * np.eye(n_sensors)

        transfer = np.ones((n_freqs, n_dirs, n_sensors), dtype=complex)
        kx = np.zeros((n_freqs, n_dirs, n_sensors))

        S = method.estimate(csd, transfer, kx)

        assert S.shape == (n_freqs, n_dirs)

    def test_non_negative(self):
        """Output should be non-negative."""
        method = BDM(max_iter=10, n_reg_steps=5)

        n_freqs, n_dirs, n_sensors = 3, 18, 3
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T + 0.1 * np.eye(n_sensors)

        transfer = np.ones((n_freqs, n_dirs, n_sensors), dtype=complex)
        kx = np.zeros((n_freqs, n_dirs, n_sensors))

        S = method.estimate(csd, transfer, kx)

        assert np.all(S >= 0)


class TestMethodComparison:
    """Tests comparing different estimation methods."""

    @pytest.fixture
    def simple_input(self):
        """Create simple test input."""
        np.random.seed(42)

        n_freqs, n_dirs, n_sensors = 10, 36, 3

        # Create positive-definite CSD
        csd = np.zeros((n_freqs, n_sensors, n_sensors), dtype=complex)
        for fi in range(n_freqs):
            A = np.random.randn(n_sensors, n_sensors) + 1j * np.random.randn(n_sensors, n_sensors)
            csd[fi] = A @ A.conj().T + 0.1 * np.eye(n_sensors)

        transfer = np.ones((n_freqs, n_dirs, n_sensors), dtype=complex)
        kx = np.zeros((n_freqs, n_dirs, n_sensors))

        return csd, transfer, kx

    def test_all_methods_run(self, simple_input):
        """All methods should run without error."""
        csd, transfer, kx = simple_input

        methods = [DFTM(), EMLM(), IMLM(max_iter=10), EMEP(max_iter=10), BDM(n_reg_steps=5)]

        for method in methods:
            S = method.estimate(csd, transfer, kx)
            assert S.shape == (10, 36)
            assert np.all(np.isfinite(S))

    def test_method_names(self):
        """Methods should have correct names."""
        assert DFTM().name == "DFTM"
        assert EMLM().name == "EMLM"
        assert IMLM().name == "IMLM"
        assert EMEP().name == "EMEP"
        assert BDM().name == "BDM"
