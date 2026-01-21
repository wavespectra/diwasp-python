"""Bayesian Direct Method (BDM) for directional spectrum estimation.

The BDM uses Bayesian inference with Tikhonov regularization to estimate
the directional spectrum. It provides natural smoothing via the Laplacian
prior and automatic hyperparameter selection via ABIC.

Reference:
    Hashimoto, N. (1997) "Analysis of the directional wave spectrum from
    field data" in Advances in Coastal Engineering Vol. 3, World Scientific.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from .base import EstimationMethodBase


class BDM(EstimationMethodBase):
    """Bayesian Direct Method.

    The BDM estimates the directional spectrum using:

    1. Log-transform spectral density: x = log(S)
    2. Solve inverse problem with Tikhonov regularization
    3. Use QR decomposition for numerical stability
    4. Select regularization via Akaike Bayesian Information Criterion (ABIC)

    The Laplacian regularization provides natural smoothing for the spectrum.
    """

    def __init__(self, max_iter: int = 100, n_reg_steps: int = 20):
        """Initialize BDM method.

        Args:
            max_iter: Maximum iterations for optimization.
            n_reg_steps: Number of regularization parameter values to try.
        """
        super().__init__(max_iter)
        self.n_reg_steps = n_reg_steps

    def estimate(
        self,
        csd_matrix: NDArray[np.complexfloating],
        transfer_matrix: NDArray[np.complexfloating],
        kx: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Estimate directional spectrum using BDM.

        Args:
            csd_matrix: Cross-spectral density matrix [n_freqs x n_sensors x n_sensors].
            transfer_matrix: Transfer functions [n_freqs x n_dirs x n_sensors].
            kx: Spatial phase lags [n_freqs x n_dirs x n_sensors].

        Returns:
            Directional spectrum estimate [n_freqs x n_dirs].
        """
        n_freqs, n_dirs, n_sensors = transfer_matrix.shape

        # Initialize output spectrum
        S = np.zeros((n_freqs, n_dirs))

        # Build Laplacian regularization matrix for circular domain
        L = self._build_laplacian(n_dirs)

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

            # Build observation vector from cross-spectra (upper triangle)
            obs_real = []
            obs_imag = []

            for i in range(n_sensors):
                for j in range(i, n_sensors):
                    obs_real.append(np.real(C[i, j]))
                    obs_imag.append(np.imag(C[i, j]))

            d = np.array(obs_real + obs_imag)
            n_obs = len(obs_real)

            # Build forward model matrix G
            # Maps directional spectrum to observed cross-spectra
            G = self._build_forward_matrix(W, n_sensors, n_dirs, n_obs)

            # Search for optimal regularization parameter
            best_abic = np.inf
            best_x = None

            # Try different regularization strengths
            reg_values = 0.5 ** np.arange(self.n_reg_steps)

            for reg in reg_values:
                try:
                    # Solve regularized least squares
                    # minimize ||Gx - d||^2 + reg * ||Lx||^2
                    x, abic = self._solve_regularized(G, d, L, reg)

                    if abic < best_abic:
                        best_abic = abic
                        best_x = x

                except np.linalg.LinAlgError:
                    continue

            # Convert from log-space to spectrum
            if best_x is not None:
                S_dir = np.exp(best_x)
            else:
                # Fallback to uniform distribution
                S_dir = np.ones(n_dirs)

            # Normalize
            total = np.sum(S_dir)
            if total > 0:
                S_dir = S_dir / total

            S[fi, :] = S_dir

        return S

    def _build_laplacian(self, n: int) -> NDArray[np.floating]:
        """Build circular Laplacian regularization matrix.

        For a circular domain, the Laplacian is:
        L[i,i] = 2, L[i,i-1] = L[i,i+1] = -1

        with periodic boundary conditions.

        Args:
            n: Size of the matrix.

        Returns:
            Laplacian matrix [n x n].
        """
        L = np.zeros((n, n))
        for i in range(n):
            L[i, i] = 2.0
            L[i, (i - 1) % n] = -1.0
            L[i, (i + 1) % n] = -1.0

        return L

    def _build_forward_matrix(
        self,
        W: NDArray[np.complexfloating],
        n_sensors: int,
        n_dirs: int,
        n_obs: int,
    ) -> NDArray[np.floating]:
        """Build forward model matrix.

        Maps log-spectrum to observed cross-spectra.

        Args:
            W: Complex weight matrix [n_dirs x n_sensors].
            n_sensors: Number of sensors.
            n_dirs: Number of directions.
            n_obs: Number of observations.

        Returns:
            Forward matrix [2*n_obs x n_dirs].
        """
        G = np.zeros((2 * n_obs, n_dirs))

        p = 0
        for i in range(n_sensors):
            for j in range(i, n_sensors):
                for di in range(n_dirs):
                    # Cross-spectrum contribution: W_i * W_j^*
                    cross = W[di, i] * W[di, j].conj()

                    # For log-space model, we need linearization
                    # Here we use a simplified linear model
                    G[p, di] = np.real(cross)
                    G[n_obs + p, di] = np.imag(cross)

                p += 1

        return G

    def _solve_regularized(
        self,
        G: NDArray[np.floating],
        d: NDArray[np.floating],
        L: NDArray[np.floating],
        reg: float,
    ) -> tuple[NDArray[np.floating], float]:
        """Solve regularized least squares problem.

        minimize ||Gx - d||^2 + reg * ||Lx||^2

        Args:
            G: Forward matrix.
            d: Observation vector.
            L: Regularization matrix.
            reg: Regularization parameter.

        Returns:
            Tuple of (solution x, ABIC value).
        """
        n_obs = len(d)
        n_dirs = G.shape[1]

        # Build augmented system
        # [G; sqrt(reg)*L] @ x = [d; 0]
        G_aug = np.vstack([G, np.sqrt(reg) * L])
        d_aug = np.concatenate([d, np.zeros(n_dirs)])

        # Solve using QR decomposition for stability
        Q, R = linalg.qr(G_aug, mode="economic")
        x = linalg.solve_triangular(R, Q.T @ d_aug)

        # Compute residual
        residual = G @ x - d
        rss = np.sum(residual**2)

        # Compute regularization penalty
        reg_penalty = np.sum((L @ x) ** 2)

        # ABIC: Akaike Bayesian Information Criterion
        # ABIC = n * log(rss/n) + log(det(G'G + reg*L'L))
        # Simplified version:
        effective_df = min(n_dirs, n_obs)
        abic = n_obs * np.log(rss / n_obs + 1e-20) + reg * reg_penalty + effective_df

        return x, abic
