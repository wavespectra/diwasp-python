"""Extended Maximum Likelihood Method (EMLM) for directional spectrum estimation.

The EMLM inverts the cross-spectral density matrix to estimate the directional
spectrum with improved resolution compared to DFTM.

Reference:
    Hashimoto, N. (1997) "Analysis of the directional wave spectrum from
    field data" in Advances in Coastal Engineering Vol. 3, World Scientific.
"""

import numpy as np
from numpy.typing import NDArray

from .base import EstimationMethodBase


class EMLM(EstimationMethodBase):
    """Extended Maximum Likelihood Method.

    The EMLM estimates the directional spectrum using matrix inversion:

    E(theta) = 1 / sum_nm [H_n * H_m* * C_inv_nm * exp(i * kx_nm)]

    This method provides better directional resolution than DFTM but is
    more sensitive to noise due to the matrix inversion.
    """

    def estimate(
        self,
        csd_matrix: NDArray[np.complexfloating],
        transfer_matrix: NDArray[np.complexfloating],
        kx: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Estimate directional spectrum using EMLM.

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

            # Invert CSD matrix with regularization for stability
            try:
                # Add small regularization to diagonal for numerical stability
                reg = 1e-10 * np.trace(C) / n_sensors
                C_reg = C + reg * np.eye(n_sensors)
                C_inv = np.linalg.inv(C_reg)
            except np.linalg.LinAlgError:
                # If inversion fails, use pseudo-inverse
                C_inv = np.linalg.pinv(C)

            # Loop over directions
            for di in range(n_dirs):
                # Get transfer function and phase for this direction
                H = transfer_matrix[fi, di, :]
                kx_d = kx[fi, di, :]

                # Complex weights: H * exp(i * kx)
                W = H * np.exp(1j * kx_d)

                # EMLM estimate: 1 / (W^H @ C_inv @ W)
                denominator = np.dot(W.conj(), np.dot(C_inv, W))

                # Take real part and ensure positive
                denom_real = np.real(denominator)

                if denom_real > 1e-20:
                    S[fi, di] = 1.0 / denom_real
                else:
                    S[fi, di] = 0.0

        # Ensure non-negative spectrum
        S = np.maximum(S, 0.0)

        # Normalize each frequency to unit directional distribution
        for fi in range(n_freqs):
            row_sum = np.sum(S[fi, :])
            if row_sum > 0:
                S[fi, :] = S[fi, :] / row_sum

        return S
