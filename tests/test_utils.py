"""Tests for utility functions."""

import numpy as np
import pytest

from diwasp.utils import (
    angular_to_frequency,
    compute_csd,
    compute_csd_matrix,
    detrend_data,
    direction_cart_to_naut,
    direction_naut_to_cart,
    directional_spread,
    frequency_to_angular,
    hsig,
    mean_direction,
    peak_direction,
    peak_frequency,
    wavenumber,
)


class TestWavenumber:
    """Tests for wavenumber calculation."""

    def test_deep_water_limit(self):
        """In deep water, k ~ sigma^2 / g."""
        sigma = np.array([1.0, 2.0, 3.0])
        depth = 1000.0  # Very deep

        k = wavenumber(sigma, depth)

        # Deep water approximation
        k_deep = sigma**2 / 9.81
        np.testing.assert_allclose(k, k_deep, rtol=0.01)

    def test_shallow_water_limit(self):
        """In shallow water, k ~ sigma / sqrt(g*d)."""
        sigma = np.array([0.1, 0.2])  # Low frequency for shallow water
        depth = 1.0  # Very shallow

        k = wavenumber(sigma, depth)

        # Shallow water approximation: k = sigma / sqrt(g*d)
        k_shallow = sigma / np.sqrt(9.81 * depth)
        np.testing.assert_allclose(k, k_shallow, rtol=0.1)

    def test_dispersion_relation(self):
        """Verify wavenumber satisfies dispersion relation."""
        sigma = np.array([0.5, 1.0, 1.5, 2.0])
        depth = 10.0

        k = wavenumber(sigma, depth)

        # Check: sigma^2 = g * k * tanh(k * d)
        lhs = sigma**2
        rhs = 9.81 * k * np.tanh(k * depth)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-6)

    def test_scalar_input(self):
        """Test with scalar input."""
        sigma = 1.0
        depth = 10.0

        k = wavenumber(sigma, depth)

        assert isinstance(k, np.ndarray)
        assert k.shape == (1,)


class TestFrequencyConversion:
    """Tests for frequency/angular frequency conversion."""

    def test_frequency_to_angular(self):
        """Test Hz to rad/s conversion."""
        freq = np.array([1.0, 2.0, 0.5])
        sigma = frequency_to_angular(freq)

        expected = 2 * np.pi * freq
        np.testing.assert_allclose(sigma, expected)

    def test_angular_to_frequency(self):
        """Test rad/s to Hz conversion."""
        sigma = np.array([2 * np.pi, 4 * np.pi, np.pi])
        freq = angular_to_frequency(sigma)

        expected = np.array([1.0, 2.0, 0.5])
        np.testing.assert_allclose(freq, expected)

    def test_roundtrip(self):
        """Test conversion roundtrip."""
        freq = np.array([0.1, 0.5, 1.0, 2.0])
        freq_back = angular_to_frequency(frequency_to_angular(freq))
        np.testing.assert_allclose(freq, freq_back)


class TestCSD:
    """Tests for cross-spectral density computation."""

    def test_auto_spectrum_positive(self):
        """Auto-spectrum should be real and non-negative."""
        np.random.seed(42)
        x = np.random.randn(1024)

        freqs, csd = compute_csd(x, x, fs=10.0)

        # Auto-spectrum should be real
        np.testing.assert_allclose(np.imag(csd), 0, atol=1e-10)

        # Auto-spectrum should be non-negative
        assert np.all(np.real(csd) >= 0)

    def test_cross_spectrum_symmetry(self):
        """Cross-spectrum Pxy should equal conjugate of Pyx."""
        np.random.seed(42)
        x = np.random.randn(1024)
        y = np.random.randn(1024)

        _, csd_xy = compute_csd(x, y, fs=10.0)
        _, csd_yx = compute_csd(y, x, fs=10.0)

        np.testing.assert_allclose(csd_xy, np.conj(csd_yx), rtol=1e-10)

    def test_csd_matrix_hermitian(self):
        """CSD matrix should be Hermitian at each frequency."""
        np.random.seed(42)
        data = np.random.randn(1024, 3)

        freqs, csd_matrix = compute_csd_matrix(data, fs=10.0)

        for fi in range(len(freqs)):
            C = csd_matrix[fi, :, :]
            np.testing.assert_allclose(C, np.conj(C.T), rtol=1e-10)


class TestSpectralStatistics:
    """Tests for spectral statistics calculations."""

    def test_hsig_uniform(self):
        """Test Hsig calculation with known energy."""
        freqs = np.linspace(0.05, 0.5, 50)
        dirs = np.linspace(0, 360, 180, endpoint=False)

        # Uniform spectrum with known total energy
        # m0 = 1, so Hsig = 4 * sqrt(1) = 4
        df = freqs[1] - freqs[0]
        ddir = dirs[1] - dirs[0]
        target_m0 = 1.0
        S_value = target_m0 / (len(freqs) * df * len(dirs) * ddir)
        S = np.full((len(freqs), len(dirs)), S_value)

        hs = hsig(S, freqs, dirs)
        np.testing.assert_allclose(hs, 4.0, rtol=0.01)

    def test_peak_frequency_unimodal(self):
        """Test peak frequency with known peak."""
        freqs = np.linspace(0.05, 0.5, 50)
        dirs = np.linspace(0, 360, 180, endpoint=False)

        S = np.zeros((len(freqs), len(dirs)))

        # Put peak at specific frequency
        peak_idx = 20
        S[peak_idx, :] = 1.0

        fp = peak_frequency(S, freqs)
        np.testing.assert_allclose(fp, freqs[peak_idx])

    def test_peak_direction_unimodal(self):
        """Test peak direction with known peak."""
        freqs = np.linspace(0.05, 0.5, 50)
        dirs = np.linspace(0, 360, 180, endpoint=False)

        S = np.zeros((len(freqs), len(dirs)))

        # Put peak at specific direction
        dir_idx = 45  # 90 degrees
        S[:, dir_idx] = 1.0

        dp = peak_direction(S, freqs, dirs)
        np.testing.assert_allclose(dp, dirs[dir_idx])

    def test_mean_direction_uniform(self):
        """Mean direction of uniform spectrum should handle wrapping."""
        freqs = np.linspace(0.05, 0.5, 50)
        dirs = np.linspace(0, 360, 180, endpoint=False)

        # Spectrum concentrated around 350-10 degrees (wrapping)
        S = np.zeros((len(freqs), len(dirs)))
        for i, d in enumerate(dirs):
            if d < 20 or d > 340:
                S[:, i] = 1.0

        dm = mean_direction(S, dirs)

        # Mean should be near 0/360 degrees
        assert dm < 20 or dm > 340


class TestDirectionConversion:
    """Tests for direction convention conversion."""

    def test_cart_to_naut_east(self):
        """East in Cartesian (0) should be East in nautical (90)."""
        cart = np.array([0.0])
        naut = direction_cart_to_naut(cart)
        np.testing.assert_allclose(naut, 90.0)

    def test_cart_to_naut_north(self):
        """North in Cartesian (90) should be North in nautical (0)."""
        cart = np.array([90.0])
        naut = direction_cart_to_naut(cart)
        np.testing.assert_allclose(naut, 0.0)

    def test_roundtrip(self):
        """Test conversion roundtrip."""
        cart = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        cart_back = direction_naut_to_cart(direction_cart_to_naut(cart))
        np.testing.assert_allclose(cart_back % 360, cart % 360)


class TestDetrend:
    """Tests for detrending."""

    def test_removes_linear_trend(self):
        """Detrending should remove linear trend."""
        t = np.arange(100)
        trend = 0.5 * t + 10
        signal = np.sin(2 * np.pi * t / 10) + trend

        detrended = detrend_data(signal)

        # Should remove the trend, leaving just the sinusoid
        # Mean should be close to zero
        np.testing.assert_allclose(np.mean(detrended), 0, atol=0.1)

    def test_2d_input(self):
        """Test with multi-column input."""
        data = np.random.randn(100, 3)
        data[:, 0] += np.arange(100) * 0.1  # Add trend to first column

        detrended = detrend_data(data)

        # Each column should be detrended
        for col in range(3):
            np.testing.assert_allclose(np.mean(detrended[:, col]), 0, atol=0.1)
