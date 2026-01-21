"""Iterated Maximum Likelihood Method (IMLM) for directional spectrum estimation.

The IMLM iteratively refines the maximum likelihood solution with relaxation
to improve convergence and stability.

Reference:
    Hashimoto, N. (1997) "Analysis of the directional wave spectrum from
    field data" in Advances in Coastal Engineering Vol. 3, World Scientific.
"""

import numpy as np
from numpy.typing import NDArray

from .base import EstimationMethodBase


class IMLM(EstimationMethodBase):
    """Iterated Maximum Likelihood Method.

    The IMLM improves upon EMLM by using iterative refinement:

    1. Start with initial estimate from EMLM
    2. Iteratively update using weighted likelihood
    3. Apply relaxation to ensure convergence

    Parameters for iteration control:
    - gamma: Step size relaxation (default 0.1)
    - beta: Regularization parameter (default 1.0)
    - alpha: Convergence factor (default 0.1)
    """

    def __init__(
        self,
        max_iter: int = 100,
        gamma: float = 0.1,
        beta: float = 1.0,
        alpha: float = 0.1,
        tol: float = 1e-6,
    ):
        """Initialize IMLM method.

        Args:
            max_iter: Maximum number of iterations.
            gamma: Step size relaxation parameter.
            beta: Regularization parameter.
            alpha: Convergence factor.
            tol: Convergence tolerance.
        """
        super().__init__(max_iter)
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha
        self.tol = tol

    def estimate(
        self,
        csd_matrix: NDArray[np.complexfloating],
        transfer_matrix: NDArray[np.complexfloating],
        kx: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Estimate directional spectrum using IMLM.

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

        # Loop over frequencies
        for fi in range(n_freqs):
            # Get CSD matrix for this frequency [n_sensors x n_sensors]
            C = csd_matrix[fi, :, :]

            # Compute weights matrix W [n_dirs x n_sensors]
            W = np.zeros((n_dirs, n_sensors), dtype=np.complex128)
            for di in range(n_dirs):
                H = transfer_matrix[fi, di, :]
                kx_d = kx[fi, di, :]
                W[di, :] = H * np.exp(1j * kx_d)

            # Initial estimate using uniform distribution
            S_dir = np.ones(n_dirs) / n_dirs

            # Iterative refinement
            for _ in range(self.max_iter):
                S_old = S_dir.copy()

                # Compute weighted CSD estimate
                # C_model = sum_theta S(theta) * W(theta) * W(theta)^H
                C_model = np.zeros((n_sensors, n_sensors), dtype=np.complex128)
                for di in range(n_dirs):
                    C_model += S_dir[di] * np.outer(W[di, :], W[di, :].conj())

                # Regularize model covariance
                reg = self.beta * np.trace(C_model) / n_sensors
                if reg < 1e-20:
                    reg = 1e-10
                C_model_reg = C_model + reg * np.eye(n_sensors)

                try:
                    C_model_inv = np.linalg.inv(C_model_reg)
                except np.linalg.LinAlgError:
                    C_model_inv = np.linalg.pinv(C_model_reg)

                # Update spectrum estimate
                S_new = np.zeros(n_dirs)
                for di in range(n_dirs):
                    w = W[di, :]
                    # IMLM update: S_new = S_old * sqrt(w^H @ C_model_inv @ C @ C_model_inv @ w)
                    #                              / (w^H @ C_model_inv @ w)
                    w_Cinv = np.dot(w.conj(), C_model_inv)

                    numerator = np.real(np.dot(w_Cinv, np.dot(C, np.dot(C_model_inv, w))))
                    denominator = np.real(np.dot(w_Cinv, w))

                    if denominator > 1e-20 and numerator > 0:
                        update = np.sqrt(numerator / denominator)
                        S_new[di] = S_old[di] * ((1 - self.gamma) + self.gamma * update)
                    else:
                        S_new[di] = S_old[di]

                # Ensure non-negative
                S_new = np.maximum(S_new, 0.0)

                # Normalize
                total = np.sum(S_new)
                if total > 0:
                    S_new = S_new / total

                # Apply relaxation
                S_dir = (1 - self.alpha) * S_old + self.alpha * S_new

                # Check convergence
                if np.max(np.abs(S_dir - S_old)) < self.tol:
                    break

            S[fi, :] = S_dir

        return S
