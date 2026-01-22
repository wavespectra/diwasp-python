"""Direct Fourier Transform Method (DFTM) for directional spectrum estimation.

The DFTM is the simplest directional estimation method. It directly integrates
the cross-spectra with transfer functions without iteration.

Reference:
    Hashimoto, N. (1997) "Analysis of the directional wave spectrum from
    field data" in Advances in Coastal Engineering Vol. 3, World Scientific.
"""

import numpy as np
from numpy.typing import NDArray

from .base import EstimationMethodBase


class DFTM(EstimationMethodBase):
    """Direct Fourier Transform Method.

    The DFTM estimates the directional spectrum through direct integration:

    S(f, theta) ~ sum_n sum_m [H_n * H_m* * C_nm * exp(i * kx_nm)]

    This is a non-iterative method that provides quick estimates but may
    produce spectra with negative values due to noise.
    """

    def estimate(
        self,
        csd_matrix: NDArray[np.complexfloating],
        transfer_matrix: NDArray[np.complexfloating],
        kx: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Estimate directional spectrum using DFTM.

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

            # Loop over directions
            for di in range(n_dirs):
                # Get transfer function and phase for this direction
                # H: [n_sensors], kx_d: [n_sensors]
                H = transfer_matrix[fi, di, :]
                kx_d = kx[fi, di, :]

                # Complex weights: H * exp(i * kx)
                W = H * np.exp(1j * kx_d)

                # Spectral estimate: sum_nm W_n * C_nm * W_m*
                # This is equivalent to W^H @ C @ W (Hermitian quadratic form)
                S_complex = np.dot(W.conj(), np.dot(C, W))

                # Take real part (should be real for valid spectrum)
                S[fi, di] = np.real(S_complex)

        # Ensure non-negative spectrum
        S = np.maximum(S, 0.0)

        # Normalize to have unit energy distribution over directions
        # (actual energy is in the frequency spectrum)
        for fi in range(n_freqs):
            row_sum = np.sum(S[fi, :])
            if row_sum > 0:
                S[fi, :] = S[fi, :] / row_sum

        return S
