"""Base class for directional spectrum estimation methods."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EstimationMethodBase(ABC):
    """Abstract base class for directional spectrum estimation methods.

    All estimation methods take cross-spectral density data and transfer
    functions as input and produce a directional spectrum estimate.

    Subclasses must implement the `estimate` method.
    """

    def __init__(self, max_iter: int = 100):
        """Initialize estimation method.

        Args:
            max_iter: Maximum iterations for iterative methods.
        """
        self.max_iter = max_iter

    @abstractmethod
    def estimate(
        self,
        csd_matrix: NDArray[np.complexfloating],
        transfer_matrix: NDArray[np.complexfloating],
        kx: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Estimate directional spectrum from cross-spectral density.

        Args:
            csd_matrix: Cross-spectral density matrix [n_freqs x n_sensors x n_sensors].
            transfer_matrix: Transfer functions [n_freqs x n_dirs x n_sensors].
            kx: Spatial phase lags [n_freqs x n_dirs x n_sensors].
                Phase lag = k * (x_i * cos(theta) + y_i * sin(theta))

        Returns:
            Directional spectrum estimate [n_freqs x n_dirs].
        """
        pass

    def _compute_phase_weights(
        self,
        transfer_matrix: NDArray[np.complexfloating],
        kx: NDArray[np.floating],
    ) -> NDArray[np.complexfloating]:
        """Compute complex phase weights for each sensor.

        H_i(f, theta) * exp(i * kx_i(f, theta))

        Args:
            transfer_matrix: Transfer functions [n_freqs x n_dirs x n_sensors].
            kx: Spatial phase lags [n_freqs x n_dirs x n_sensors].

        Returns:
            Complex weights [n_freqs x n_dirs x n_sensors].
        """
        return transfer_matrix * np.exp(1j * kx)

    @property
    def name(self) -> str:
        """Return the method name."""
        return self.__class__.__name__


def compute_kx(
    k: NDArray[np.floating],
    theta: NDArray[np.floating],
    sensor_x: NDArray[np.floating],
    sensor_y: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute spatial phase lags for all sensors.

    kx_i(f, theta) = k(f) * (x_i * cos(theta) + y_i * sin(theta))

    Args:
        k: Wavenumbers [n_freqs].
        theta: Directions in radians [n_dirs].
        sensor_x: Sensor x-coordinates [n_sensors].
        sensor_y: Sensor y-coordinates [n_sensors].

    Returns:
        Spatial phase lags [n_freqs x n_dirs x n_sensors].
    """
    n_freqs = len(k)
    n_dirs = len(theta)
    n_sensors = len(sensor_x)

    # Compute cos and sin of directions
    cos_theta = np.cos(theta)  # [n_dirs]
    sin_theta = np.sin(theta)  # [n_dirs]

    # Initialize output array
    kx = np.zeros((n_freqs, n_dirs, n_sensors))

    for i in range(n_sensors):
        # Spatial offset for this sensor: x_i * cos(theta) + y_i * sin(theta)
        spatial_offset = sensor_x[i] * cos_theta + sensor_y[i] * sin_theta  # [n_dirs]

        # Phase lag: k(f) * spatial_offset
        # Broadcast k [n_freqs] with spatial_offset [n_dirs]
        kx[:, :, i] = np.outer(k, spatial_offset)

    return kx
