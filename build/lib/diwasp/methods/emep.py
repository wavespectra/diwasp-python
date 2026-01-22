"""Extended Maximum Entropy Principle (EMEP) for directional spectrum estimation.

The EMEP uses a model selection approach based on the Akaike Information
Criterion (AIC) to determine the optimal model order for representing
the directional spectrum.

Reference:
    Hashimoto, N. (1997) "Analysis of the directional wave spectrum from
    field data" in Advances in Coastal Engineering Vol. 3, World Scientific.
"""

import numpy as np
from numpy.typing import NDArray

from .base import EstimationMethodBase


class EMEP(EstimationMethodBase):
    """Extended Maximum Entropy Principle method.

    The EMEP estimates the directional spectrum using:

    1. Separate co- and quadrature components from cross-spectra
    2. Expand using cosine/sine basis functions
    3. Iteratively fit models of increasing order
    4. Select optimal order using Akaike Information Criterion (AIC)

    This provides automatic model order selection for robust estimates.
    """

    def __init__(self, max_iter: int = 100, max_order: int | None = None):
        """Initialize EMEP method.

        Args:
            max_iter: Maximum iterations per model order.
            max_order: Maximum model order to try. If None, uses n_sensors - 1.
        """
        super().__init__(max_iter)
        self.max_order = max_order

    def estimate(
        self,
        csd_matrix: NDArray[np.complexfloating],
        transfer_matrix: NDArray[np.complexfloating],
        kx: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Estimate directional spectrum using EMEP.

        Args:
            csd_matrix: Cross-spectral density matrix [n_freqs x n_sensors x n_sensors].
            transfer_matrix: Transfer functions [n_freqs x n_dirs x n_sensors].
            kx: Spatial phase lags [n_freqs x n_dirs x n_sensors].

        Returns:
            Directional spectrum estimate [n_freqs x n_dirs].
        """
        n_freqs, n_dirs, n_sensors = transfer_matrix.shape

        # Determine maximum model order
        max_order = self.max_order if self.max_order else n_sensors - 1
        max_order = min(max_order, n_dirs // 2 - 1)  # Limit by Nyquist

        # Initialize output spectrum
        S = np.zeros((n_freqs, n_dirs))

        # Direction grid in radians
        theta = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)

        # Loop over frequencies
        for fi in range(n_freqs):
            # Get CSD matrix for this frequency
            C = csd_matrix[fi, :, :]

            # Compute weights matrix W [n_dirs x n_sensors]
            W = np.zeros((n_dirs, n_sensors), dtype=np.complex128)
            for di in range(n_dirs):
                H = transfer_matrix[fi, di, :]
                kx_d = kx[fi, di, :]
                W[di, :] = H * np.exp(1j * kx_d)

            # Extract co- and quadrature spectra from cross-spectral matrix
            # Compute theoretical cross-spectrum for each direction
            # This gives us observations to fit

            # Build observation vector from cross-spectra
            # Use upper triangle of CSD matrix
            obs_real = []
            obs_imag = []
            sensor_pairs = []

            for i in range(n_sensors):
                for j in range(i, n_sensors):
                    obs_real.append(np.real(C[i, j]))
                    obs_imag.append(np.imag(C[i, j]))
                    sensor_pairs.append((i, j))

            n_obs = len(obs_real)
            obs = np.array(obs_real + obs_imag)

            # Try different model orders and select best by AIC
            best_aic = np.inf
            best_coeffs = None
            best_order = 1

            for order in range(1, max_order + 1):
                # Number of Fourier coefficients: a0, a1..an, b1..bn
                n_coeffs = 2 * order + 1

                if n_coeffs > n_obs:
                    break

                # Build design matrix for Fourier series
                # S(theta) = a0 + sum_k [ak * cos(k*theta) + bk * sin(k*theta)]
                A = self._build_design_matrix(W, theta, order, sensor_pairs, n_obs)

                # Solve least squares with regularization
                try:
                    coeffs, residuals, rank, singular = np.linalg.lstsq(
                        A, obs, rcond=None
                    )

                    # Compute AIC
                    if len(residuals) > 0:
                        rss = residuals[0]
                    else:
                        rss = np.sum((obs - A @ coeffs) ** 2)

                    n_data = len(obs)
                    aic = n_data * np.log(rss / n_data + 1e-20) + 2 * n_coeffs

                    if aic < best_aic:
                        best_aic = aic
                        best_coeffs = coeffs
                        best_order = order

                except np.linalg.LinAlgError:
                    continue

            # Reconstruct spectrum from best coefficients
            if best_coeffs is not None:
                S_dir = self._reconstruct_spectrum(theta, best_coeffs, best_order)
            else:
                # Fallback to uniform distribution
                S_dir = np.ones(n_dirs) / n_dirs

            # Ensure non-negative and normalize
            S_dir = np.maximum(S_dir, 0.0)
            total = np.sum(S_dir)
            if total > 0:
                S_dir = S_dir / total

            S[fi, :] = S_dir

        return S

    def _build_design_matrix(
        self,
        W: NDArray[np.complexfloating],
        theta: NDArray[np.floating],
        order: int,
        sensor_pairs: list[tuple[int, int]],
        n_obs: int,
    ) -> NDArray[np.floating]:
        """Build design matrix for Fourier series fit.

        Args:
            W: Complex weight matrix [n_dirs x n_sensors].
            theta: Direction grid in radians [n_dirs].
            order: Fourier series order.
            sensor_pairs: List of (i, j) sensor pairs.
            n_obs: Number of observations (number of sensor pairs).

        Returns:
            Design matrix [2*n_obs x (2*order+1)].
        """
        n_dirs = len(theta)
        n_coeffs = 2 * order + 1

        # Design matrix: maps Fourier coefficients to observed cross-spectra
        A = np.zeros((2 * n_obs, n_coeffs))

        # Build basis functions for directional distribution
        # S(theta) = a0 + sum_k [ak * cos(k*theta) + bk * sin(k*theta)]
        basis = np.zeros((n_dirs, n_coeffs))
        basis[:, 0] = 1.0  # a0
        for k in range(1, order + 1):
            basis[:, k] = np.cos(k * theta)  # ak
            basis[:, order + k] = np.sin(k * theta)  # bk

        # For each sensor pair, compute how each basis function contributes
        # to the cross-spectrum
        for p, (i, j) in enumerate(sensor_pairs):
            for c in range(n_coeffs):
                # Contribution to real part
                contrib_real = 0.0
                contrib_imag = 0.0

                for di in range(n_dirs):
                    # Cross-spectrum contribution: W_i * W_j^* * basis
                    cross = W[di, i] * W[di, j].conj()
                    contrib_real += np.real(cross) * basis[di, c]
                    contrib_imag += np.imag(cross) * basis[di, c]

                A[p, c] = contrib_real / n_dirs
                A[n_obs + p, c] = contrib_imag / n_dirs

        return A

    def _reconstruct_spectrum(
        self,
        theta: NDArray[np.floating],
        coeffs: NDArray[np.floating],
        order: int,
    ) -> NDArray[np.floating]:
        """Reconstruct directional spectrum from Fourier coefficients.

        Args:
            theta: Direction grid in radians [n_dirs].
            coeffs: Fourier coefficients [2*order+1].
            order: Fourier series order.

        Returns:
            Directional spectrum [n_dirs].
        """
        n_dirs = len(theta)
        S = np.full(n_dirs, coeffs[0])  # a0 term

        for k in range(1, order + 1):
            if k < len(coeffs):
                S += coeffs[k] * np.cos(k * theta)
            if order + k < len(coeffs):
                S += coeffs[order + k] * np.sin(k * theta)

        return S
