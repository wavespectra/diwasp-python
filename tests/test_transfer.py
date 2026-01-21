"""Tests for transfer functions."""

import numpy as np
import pytest

from diwasp.transfer import (
    TRANSFER_FUNCTIONS,
    AccelerationXTransfer,
    AccelerationYTransfer,
    AccelerationZTransfer,
    ElevationTransfer,
    PressureTransfer,
    SlopeXTransfer,
    SlopeYTransfer,
    SurfaceVelocityTransfer,
    VelocityXTransfer,
    VelocityYTransfer,
    VelocityZTransfer,
    compute_transfer_matrix,
    get_transfer_function,
)
from diwasp.types import SensorType


class TestElevationTransfer:
    """Tests for elevation transfer function."""

    def test_identity(self):
        """Elevation transfer should be identity (1.0)."""
        tf = ElevationTransfer()

        sigma = np.array([1.0, 2.0])
        k = np.array([0.1, 0.4])
        theta = np.array([0.0, np.pi / 2, np.pi])
        depth = 10.0
        z = 10.0

        H = tf(sigma, k, theta, depth, z)

        assert H.shape == (2, 3)
        np.testing.assert_allclose(np.abs(H), 1.0)


class TestPressureTransfer:
    """Tests for pressure transfer function."""

    def test_surface_unity(self):
        """At surface (z=d), pressure transfer should approach 1."""
        tf = PressureTransfer()

        sigma = np.array([1.0])
        k = np.array([0.1])  # Long wavelength
        theta = np.array([0.0])
        depth = 10.0
        z = depth  # At surface

        H = tf(sigma, k, theta, depth, z)

        # At surface, cosh(k*d)/cosh(k*d) = 1
        np.testing.assert_allclose(np.abs(H[0, 0]), 1.0, rtol=0.01)

    def test_depth_attenuation(self):
        """Pressure should attenuate with depth."""
        tf = PressureTransfer()

        sigma = np.array([1.0])
        k = np.array([0.5])  # Medium wavelength
        theta = np.array([0.0])
        depth = 10.0

        # At bottom
        H_bottom = tf(sigma, k, theta, depth, z=0.0)
        # At surface
        H_surface = tf(sigma, k, theta, depth, z=depth)

        assert np.abs(H_bottom[0, 0]) < np.abs(H_surface[0, 0])

    def test_cutoff(self):
        """Pressure transfer should have minimum cutoff."""
        tf = PressureTransfer(min_cutoff=0.1)

        sigma = np.array([10.0])  # Very high frequency
        k = np.array([10.0])  # Large wavenumber
        theta = np.array([0.0])
        depth = 10.0
        z = 0.0  # At bottom

        H = tf(sigma, k, theta, depth, z)

        # Should not drop below cutoff
        assert np.abs(H[0, 0]) >= 0.1


class TestVelocityTransfer:
    """Tests for velocity transfer functions."""

    def test_velx_directional(self):
        """VelocityX should vary with cos(theta)."""
        tf = VelocityXTransfer()

        sigma = np.array([1.0])
        k = np.array([0.1])
        theta = np.array([0.0, np.pi / 2, np.pi])  # 0, 90, 180 deg
        depth = 10.0
        z = 5.0

        H = tf(sigma, k, theta, depth, z)

        # At theta=0, cos=1 (max)
        # At theta=pi/2, cos=0 (zero)
        # At theta=pi, cos=-1 (negative max)
        assert np.abs(H[0, 0]) > np.abs(H[0, 1])  # 0 > 90 deg
        np.testing.assert_allclose(H[0, 1], 0, atol=1e-10)  # 90 deg ~ 0

    def test_vely_directional(self):
        """VelocityY should vary with sin(theta)."""
        tf = VelocityYTransfer()

        sigma = np.array([1.0])
        k = np.array([0.1])
        theta = np.array([0.0, np.pi / 2, np.pi])
        depth = 10.0
        z = 5.0

        H = tf(sigma, k, theta, depth, z)

        # At theta=0, sin=0 (zero)
        # At theta=pi/2, sin=1 (max)
        # At theta=pi, sin=0 (zero)
        np.testing.assert_allclose(H[0, 0], 0, atol=1e-10)
        assert np.abs(H[0, 1]) > np.abs(H[0, 0])

    def test_velz_isotropic(self):
        """VelocityZ should not depend on direction."""
        tf = VelocityZTransfer()

        sigma = np.array([1.0])
        k = np.array([0.1])
        theta = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        depth = 10.0
        z = 5.0

        H = tf(sigma, k, theta, depth, z)

        # Should be same for all directions
        np.testing.assert_allclose(H[0, :], H[0, 0])


class TestAccelerationTransfer:
    """Tests for acceleration transfer functions."""

    def test_acceleration_vs_velocity(self):
        """Acceleration should be sigma * velocity transfer."""
        vel_tf = VelocityXTransfer()
        acc_tf = AccelerationXTransfer()

        sigma = np.array([1.0, 2.0])
        k = np.array([0.1, 0.4])
        theta = np.array([0.0, np.pi / 4])
        depth = 10.0
        z = 5.0

        H_vel = vel_tf(sigma, k, theta, depth, z)
        H_acc = acc_tf(sigma, k, theta, depth, z)

        # Acceleration = sigma * velocity
        expected = H_vel * sigma[:, np.newaxis]
        np.testing.assert_allclose(H_acc, expected)


class TestSlopeTransfer:
    """Tests for slope transfer functions."""

    def test_slope_imaginary(self):
        """Slope transfer should be purely imaginary."""
        tf_x = SlopeXTransfer()
        tf_y = SlopeYTransfer()

        sigma = np.array([1.0])
        k = np.array([0.1])
        theta = np.array([np.pi / 4])  # 45 degrees
        depth = 10.0
        z = depth

        H_x = tf_x(sigma, k, theta, depth, z)
        H_y = tf_y(sigma, k, theta, depth, z)

        # Should be purely imaginary
        np.testing.assert_allclose(np.real(H_x), 0, atol=1e-10)
        np.testing.assert_allclose(np.real(H_y), 0, atol=1e-10)


class TestTransferFunctionRegistry:
    """Tests for transfer function registry."""

    def test_all_sensor_types_registered(self):
        """All sensor types should have a transfer function."""
        for sensor_type in SensorType:
            tf = get_transfer_function(sensor_type)
            assert tf is not None

    def test_get_transfer_function(self):
        """Test get_transfer_function returns correct types."""
        assert isinstance(get_transfer_function(SensorType.ELEV), ElevationTransfer)
        assert isinstance(get_transfer_function(SensorType.PRES), PressureTransfer)
        assert isinstance(get_transfer_function(SensorType.VELX), VelocityXTransfer)


class TestComputeTransferMatrix:
    """Tests for transfer matrix computation."""

    def test_shape(self):
        """Test output shape is correct."""
        sensor_types = [SensorType.PRES, SensorType.VELX, SensorType.VELY]
        sensor_z = np.array([5.0, 5.0, 5.0])
        sigma = np.array([1.0, 2.0, 3.0])
        k = np.array([0.1, 0.4, 0.9])
        theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        depth = 10.0

        H = compute_transfer_matrix(sensor_types, sensor_z, sigma, k, theta, depth)

        assert H.shape == (3, 36, 3)  # (n_freqs, n_dirs, n_sensors)

    def test_complex_output(self):
        """Output should be complex."""
        sensor_types = [SensorType.SLPX]
        sensor_z = np.array([10.0])
        sigma = np.array([1.0])
        k = np.array([0.1])
        theta = np.array([np.pi / 4])
        depth = 10.0

        H = compute_transfer_matrix(sensor_types, sensor_z, sigma, k, theta, depth)

        assert H.dtype == np.complex128
