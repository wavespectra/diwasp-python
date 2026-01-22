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

    1. Start with initial EMLM estimate (inverse CSD method)
    2. Iteratively update: ei = gamma * ((Eo - T) + alpha * (T - Told))
    3. Additive update: E = E + ei

    Parameters for iteration control:
    - gamma: Step size relaxation (default 0.1)
    - alpha: Momentum factor for smoothing updates (default 0.1)
    """

    def __init__(
        self,
        max_iter: int = 100,
        gamma: float = 0.1,
        alpha: float = 0.1,
    ):
        """Initialize IMLM method.

        Args:
            max_iter: Maximum number of iterations.
            gamma: Step size relaxation parameter.
            alpha: Momentum factor for update smoothing.
        """
        super().__init__(max_iter)
        self.gamma = gamma
        self.alpha = alpha

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
        ddir = 2.0 * np.pi / n_dirs

        # Initialize output spectrum
        S = np.zeros((n_freqs, n_dirs))

        for ff in range(n_freqs):
            # Precompute transfer function products using broadcasting
            # H[n_dirs, n_sensors], Hs[n_dirs, n_sensors]
            H = transfer_matrix[ff, :, :]  # [n_dirs x n_sensors]
            Hs = np.conj(transfer_matrix[ff, :, :])  # [n_dirs x n_sensors]

            # Phase differences: [n_dirs x n_sensors x n_sensors]
            # phase_diff[d, m, n] = kx[ff, d, m] - kx[ff, d, n]
            kx_ff = kx[ff, :, :]  # [n_dirs x n_sensors]
            phase_diff = kx_ff[:, :, np.newaxis] - kx_ff[:, np.newaxis, :]

            expx = np.exp(1j * phase_diff)  # [n_dirs x n_sensors x n_sensors]
            iexpx = np.exp(-1j * phase_diff)

            # Htemp[d, m, n] = H[d, n] * Hs[d, m] * expx[d, m, n]
            Htemp = H[:, np.newaxis, :] * Hs[:, :, np.newaxis] * expx
            iHtemp = H[:, np.newaxis, :] * Hs[:, :, np.newaxis] * iexpx

            # Initial EMLM estimate
            try:
                invcps = np.linalg.inv(csd_matrix[ff])
            except np.linalg.LinAlgError:
                invcps = np.linalg.pinv(csd_matrix[ff])

            # Sftmp[d] = sum_mn invcps[m,n] * Htemp[d,m,n]
            Sftmp = np.einsum("mn,dmn->d", invcps, Htemp)

            # Initial estimate Eo (normalized to sum=1)
            Eo = 1.0 / np.maximum(np.real(Sftmp), 1e-20)
            Eo = Eo / np.sum(Eo)

            E = Eo.copy()
            T = Eo.copy()

            # Iterative refinement
            for _ in range(self.max_iter):
                # Compute model covariance: ixps[m,n] = sum_d iHtemp[d,m,n] * E[d] * ddir
                ixps = np.einsum("dmn,d->mn", iHtemp, E) * ddir

                # Invert model covariance
                try:
                    invcps = np.linalg.inv(ixps)
                except np.linalg.LinAlgError:
                    invcps = np.linalg.pinv(ixps)

                # Compute new target T: Sftmp[d] = sum_mn invcps[m,n] * Htemp[d,m,n]
                Sftmp = np.einsum("mn,dmn->d", invcps, Htemp)

                Told = T.copy()
                T = 1.0 / np.maximum(np.real(Sftmp), 1e-20)
                T = T / np.sum(T)

                # IMLM update: ei = gamma * ((Eo - T) + alpha * (T - Told))
                ei = self.gamma * ((Eo - T) + self.alpha * (T - Told))
                E = E + ei

                # Ensure non-negative
                E = np.maximum(E, 1e-20)

                # Normalize to sum=1
                E = E / np.sum(E)

            S[ff, :] = E

        return S
