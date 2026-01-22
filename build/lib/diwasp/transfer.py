"""Transfer functions for different sensor types.

Transfer functions convert sensor measurements to equivalent surface elevation
spectra. Each sensor type has a specific transfer function based on linear
wave theory.

The transfer functions are complex-valued and depend on:
- Angular frequency (sigma)
- Wavenumber (k)
- Water depth (d)
- Sensor elevation from seabed (z)
- Direction (theta) for directional sensors
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .types import SensorType

if TYPE_CHECKING:
    pass


class TransferFunction(ABC):
    """Base class for sensor transfer functions."""

    @abstractmethod
    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Calculate transfer function.

        Args:
            sigma: Angular frequency in rad/s [n_freqs].
            k: Wavenumber in rad/m [n_freqs].
            theta: Direction in radians [n_dirs].
            depth: Water depth in meters.
            z: Sensor elevation from seabed in meters.

        Returns:
            Complex transfer function [n_freqs x n_dirs].
        """
        pass


class ElevationTransfer(TransferFunction):
    """Transfer function for surface elevation measurements.

    H(f, theta) = 1 (identity)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Surface elevation transfer function (identity)."""
        n_freqs = len(sigma)
        n_dirs = len(theta)
        return np.ones((n_freqs, n_dirs), dtype=np.complex128)


class PressureTransfer(TransferFunction):
    """Transfer function for pressure measurements.

    H(f, theta) = cosh(k*z) / cosh(k*d)

    with a minimum cutoff to prevent amplification at high frequencies.
    """

    def __init__(self, min_cutoff: float = 0.1):
        """Initialize pressure transfer function.

        Args:
            min_cutoff: Minimum transfer function value to prevent excessive
                amplification at high frequencies.
        """
        self.min_cutoff = min_cutoff

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Pressure transfer function."""
        n_dirs = len(theta)

        # Calculate hyperbolic terms
        cosh_kz = np.cosh(k * z)
        cosh_kd = np.cosh(k * depth)

        # Transfer function (real, same for all directions)
        H = cosh_kz / cosh_kd

        # Apply minimum cutoff
        H = np.maximum(H, self.min_cutoff)

        # Broadcast to [n_freqs x n_dirs]
        return np.tile(H[:, np.newaxis], (1, n_dirs)).astype(np.complex128)


class VelocityXTransfer(TransferFunction):
    """Transfer function for horizontal velocity (x-component).

    H(f, theta) = sigma * k_z * cos(theta)

    where k_z = cosh(k*z) / sinh(k*d)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Horizontal x-velocity transfer function."""
        # Calculate hyperbolic term
        cosh_kz = np.cosh(k * z)
        sinh_kd = np.sinh(k * depth)

        # Prevent division by zero for very small sinh
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = cosh_kz / sinh_kd

        # H = sigma * k_z * cos(theta)
        # Shape: [n_freqs x n_dirs]
        H = np.outer(sigma * k_z, np.cos(theta))

        return H.astype(np.complex128)


class VelocityYTransfer(TransferFunction):
    """Transfer function for horizontal velocity (y-component).

    H(f, theta) = sigma * k_z * sin(theta)

    where k_z = cosh(k*z) / sinh(k*d)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Horizontal y-velocity transfer function."""
        cosh_kz = np.cosh(k * z)
        sinh_kd = np.sinh(k * depth)
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = cosh_kz / sinh_kd

        # H = sigma * k_z * sin(theta)
        H = np.outer(sigma * k_z, np.sin(theta))

        return H.astype(np.complex128)


class VelocityZTransfer(TransferFunction):
    """Transfer function for vertical velocity.

    H(f, theta) = sigma * k_z

    where k_z = sinh(k*z) / sinh(k*d)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Vertical velocity transfer function."""
        sinh_kz = np.sinh(k * z)
        sinh_kd = np.sinh(k * depth)
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = sinh_kz / sinh_kd

        # H = sigma * k_z (same for all directions)
        H = sigma * k_z
        n_dirs = len(theta)

        return np.tile(H[:, np.newaxis], (1, n_dirs)).astype(np.complex128)


class SurfaceVelocityTransfer(TransferFunction):
    """Transfer function for surface velocity measurements.

    H(f, theta) = sigma (at surface, z = d)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Surface velocity transfer function."""
        n_dirs = len(theta)
        H = np.tile(sigma[:, np.newaxis], (1, n_dirs))
        return H.astype(np.complex128)


class AccelerationXTransfer(TransferFunction):
    """Transfer function for horizontal acceleration (x-component).

    H(f, theta) = sigma^2 * k_z * cos(theta)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Horizontal x-acceleration transfer function."""
        cosh_kz = np.cosh(k * z)
        sinh_kd = np.sinh(k * depth)
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = cosh_kz / sinh_kd

        # H = sigma^2 * k_z * cos(theta)
        H = np.outer(sigma**2 * k_z, np.cos(theta))

        return H.astype(np.complex128)


class AccelerationYTransfer(TransferFunction):
    """Transfer function for horizontal acceleration (y-component).

    H(f, theta) = sigma^2 * k_z * sin(theta)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Horizontal y-acceleration transfer function."""
        cosh_kz = np.cosh(k * z)
        sinh_kd = np.sinh(k * depth)
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = cosh_kz / sinh_kd

        # H = sigma^2 * k_z * sin(theta)
        H = np.outer(sigma**2 * k_z, np.sin(theta))

        return H.astype(np.complex128)


class AccelerationZTransfer(TransferFunction):
    """Transfer function for vertical acceleration.

    H(f, theta) = sigma^2 * k_z

    where k_z = sinh(k*z) / sinh(k*d)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Vertical acceleration transfer function."""
        sinh_kz = np.sinh(k * z)
        sinh_kd = np.sinh(k * depth)
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = sinh_kz / sinh_kd

        H = sigma**2 * k_z
        n_dirs = len(theta)

        return np.tile(H[:, np.newaxis], (1, n_dirs)).astype(np.complex128)


class SurfaceAccelerationTransfer(TransferFunction):
    """Transfer function for surface acceleration.

    H(f, theta) = sigma^2
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Surface acceleration transfer function."""
        n_dirs = len(theta)
        H = np.tile((sigma**2)[:, np.newaxis], (1, n_dirs))
        return H.astype(np.complex128)


class SlopeXTransfer(TransferFunction):
    """Transfer function for surface slope (x-component).

    H(f, theta) = -i * k * cos(theta)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Surface x-slope transfer function."""
        # H = -i * k * cos(theta)
        H = -1j * np.outer(k, np.cos(theta))
        return H


class SlopeYTransfer(TransferFunction):
    """Transfer function for surface slope (y-component).

    H(f, theta) = -i * k * sin(theta)
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Surface y-slope transfer function."""
        # H = -i * k * sin(theta)
        H = -1j * np.outer(k, np.sin(theta))
        return H


class DisplacementXTransfer(TransferFunction):
    """Transfer function for horizontal displacement (x-component).

    Same as pressure transfer divided by (i * sigma).
    """

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Horizontal x-displacement transfer function."""
        cosh_kz = np.cosh(k * z)
        sinh_kd = np.sinh(k * depth)
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = cosh_kz / sinh_kd

        # H = k_z * cos(theta) / (i * sigma) = -i * k_z * cos(theta) / sigma
        # Avoid division by zero at sigma = 0
        sigma_safe = np.maximum(np.abs(sigma), 1e-10)
        H = np.outer(-1j * k_z / sigma_safe, np.cos(theta))

        return H


class DisplacementYTransfer(TransferFunction):
    """Transfer function for horizontal displacement (y-component)."""

    def __call__(
        self,
        sigma: NDArray[np.floating],
        k: NDArray[np.floating],
        theta: NDArray[np.floating],
        depth: float,
        z: float,
    ) -> NDArray[np.complexfloating]:
        """Horizontal y-displacement transfer function."""
        cosh_kz = np.cosh(k * z)
        sinh_kd = np.sinh(k * depth)
        sinh_kd = np.maximum(sinh_kd, 1e-10)

        k_z = cosh_kz / sinh_kd

        sigma_safe = np.maximum(np.abs(sigma), 1e-10)
        H = np.outer(-1j * k_z / sigma_safe, np.sin(theta))

        return H


# Mapping from sensor types to transfer function classes
TRANSFER_FUNCTIONS: dict[SensorType, type[TransferFunction]] = {
    SensorType.ELEV: ElevationTransfer,
    SensorType.PRES: PressureTransfer,
    SensorType.VELX: VelocityXTransfer,
    SensorType.VELY: VelocityYTransfer,
    SensorType.VELZ: VelocityZTransfer,
    SensorType.VELS: SurfaceVelocityTransfer,
    SensorType.ACCX: AccelerationXTransfer,
    SensorType.ACCY: AccelerationYTransfer,
    SensorType.ACCZ: AccelerationZTransfer,
    SensorType.ACCS: SurfaceAccelerationTransfer,
    SensorType.SLPX: SlopeXTransfer,
    SensorType.SLPY: SlopeYTransfer,
    SensorType.DSPX: DisplacementXTransfer,
    SensorType.DSPY: DisplacementYTransfer,
}


def get_transfer_function(sensor_type: SensorType) -> TransferFunction:
    """Get the transfer function for a given sensor type.

    Args:
        sensor_type: The sensor type.

    Returns:
        Transfer function instance.

    Raises:
        ValueError: If sensor type is not supported.
    """
    if sensor_type not in TRANSFER_FUNCTIONS:
        raise ValueError(f"Unsupported sensor type: {sensor_type}")

    return TRANSFER_FUNCTIONS[sensor_type]()


def compute_transfer_matrix(
    sensor_types: list[SensorType],
    sensor_z: NDArray[np.floating],
    sigma: NDArray[np.floating],
    k: NDArray[np.floating],
    theta: NDArray[np.floating],
    depth: float,
) -> NDArray[np.complexfloating]:
    """Compute transfer function matrix for all sensors.

    Args:
        sensor_types: List of sensor types.
        sensor_z: Sensor elevations from seabed [n_sensors].
        sigma: Angular frequencies [n_freqs].
        k: Wavenumbers [n_freqs].
        theta: Directions in radians [n_dirs].
        depth: Water depth.

    Returns:
        Transfer matrix [n_freqs x n_dirs x n_sensors].
    """
    n_sensors = len(sensor_types)
    n_freqs = len(sigma)
    n_dirs = len(theta)

    H = np.zeros((n_freqs, n_dirs, n_sensors), dtype=np.complex128)

    for i, (stype, z) in enumerate(zip(sensor_types, sensor_z)):
        transfer_func = get_transfer_function(stype)
        H[:, :, i] = transfer_func(sigma, k, theta, depth, z)

    return H
