"""Bayesian Direct Method (BDM) for directional spectrum estimation.

The BDM uses Bayesian inference with Tikhonov regularization to estimate
the directional spectrum. It provides natural smoothing via a second-derivative
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
    2. Newton-Raphson iterative optimization with exponential model
    3. QR decomposition for numerical stability
    4. Select regularization via Akaike Bayesian Information Criterion (ABIC)

    Uses second-derivative regularization with circular boundary conditions.
    """

    def __init__(self, max_iter: int = 100, n_reg_steps: int = 6):
        """Initialize BDM method.

        Args:
            max_iter: Maximum iterations per regularization parameter.
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
        ddir = 2.0 * np.pi / n_dirs

        # Build second-derivative regularization matrix (circular boundary)
        dd = self._build_second_derivative(n_dirs)

        # Extract Co and Quad spectra
        Co = np.real(csd_matrix)
        Quad = -np.imag(csd_matrix)

        # Compute normalization factors (standard deviations)
        sigCo = np.zeros((n_sensors, n_sensors, n_freqs))
        sigQuad = np.zeros((n_sensors, n_sensors, n_freqs))

        for ff in range(n_freqs):
            xpsx = np.real(np.outer(np.diag(csd_matrix[ff]), np.diag(csd_matrix[ff]).conj()))
            sigCo[:, :, ff] = np.sqrt(np.maximum(0.5 * (xpsx + Co[ff] ** 2 - Quad[ff] ** 2), 1e-20))
            sigQuad[:, :, ff] = np.sqrt(
                np.maximum(0.5 * (xpsx - Co[ff] ** 2 + Quad[ff] ** 2), 1e-20)
            )

        # Initialize output spectrum
        S = np.zeros((n_freqs, n_dirs))

        for ff in range(n_freqs):
            # Build observation vector phi and transfer matrix H
            phi_list = []
            H_list = []

            for m in range(n_sensors):
                for n in range(m, n_sensors):
                    # Compute transfer function product
                    Hh = transfer_matrix[ff, :, m]
                    Hhs = np.conj(transfer_matrix[ff, :, n])
                    expx = np.exp(-1j * (kx[ff, :, m] - kx[ff, :, n]))
                    Htemp = Hh * Hhs * expx

                    # Check if this pair provides useful information
                    if not np.allclose(Htemp[0], Htemp[1]):
                        # Real part (Co-spectrum)
                        sig_co = np.real(sigCo[m, n, ff])
                        if sig_co > 1e-20:
                            phi_list.append(np.real(csd_matrix[ff, m, n]) / sig_co)
                            H_list.append(np.real(Htemp) / sig_co)

                        # Imaginary part (Quad-spectrum) if spatially separated
                        if not np.allclose(kx[ff, 0, m], kx[ff, 0, n]):
                            sig_quad = np.real(sigQuad[m, n, ff])
                            if sig_quad > 1e-20:
                                phi_list.append(np.imag(csd_matrix[ff, m, n]) / sig_quad)
                                H_list.append(np.imag(Htemp) / sig_quad)

            if len(phi_list) == 0:
                S[ff, :] = np.ones(n_dirs) / n_dirs
                continue

            M = len(phi_list)
            B = np.array(phi_list)  # [M]
            A = np.array(H_list) * ddir  # [M x n_dirs]

            # Model order selection with ABIC
            ABIC_values = []
            x_held = []
            keepgoing = True
            n_model = 0

            while keepgoing and n_model < self.n_reg_steps:
                n_model += 1
                u = 0.5**n_model  # Regularization parameter

                # Initialize x = log(1/(2*pi)) for uniform distribution
                x = np.full(n_dirs, np.log(1.0 / (2 * np.pi)))

                rlx = 1.0
                count = 0
                converged = False

                while not converged:
                    count += 1

                    # Exponential model: F = exp(x), with bounds to prevent overflow
                    x_clipped = np.clip(x, -20, 20)
                    F = np.exp(x_clipped)
                    E = np.diag(F)

                    # Linearized system: A @ E @ dx = B - A @ F + A @ E @ x
                    A2 = A @ E  # [M x n_dirs]
                    B2 = B - A @ F + A @ E @ x  # [M]

                    # Check for numerical issues
                    if not np.all(np.isfinite(A2)) or not np.all(np.isfinite(B2)):
                        if rlx > 0.0625:
                            rlx *= 0.5
                            x = np.full(n_dirs, np.log(1.0 / (2 * np.pi)))
                            count = 0
                            continue
                        else:
                            break

                    # Build augmented system for QR decomposition
                    # [A2; u*dd] @ x = [B2; 0]
                    Z = np.zeros((M + n_dirs, n_dirs + 1))
                    Z[:M, :n_dirs] = A2
                    Z[M:, :n_dirs] = u * dd
                    Z[:M, n_dirs] = B2
                    Z[M:, n_dirs] = 0.0

                    # QR decomposition
                    try:
                        Q, U = linalg.qr(Z, mode="full")
                    except (linalg.LinAlgError, ValueError):
                        break

                    # Extract triangular system
                    TA = U[:n_dirs, :n_dirs]
                    Tb = U[:n_dirs, n_dirs]

                    # Check for numerical issues in result
                    if not np.all(np.isfinite(Tb)):
                        if rlx > 0.0625:
                            rlx *= 0.5
                            x = np.full(n_dirs, np.log(1.0 / (2 * np.pi)))
                            count = 0
                            continue
                        else:
                            break

                    # Solve for x1
                    try:
                        x1 = linalg.solve_triangular(TA, Tb, lower=False)
                    except (linalg.LinAlgError, ValueError):
                        try:
                            x1 = np.linalg.lstsq(TA, Tb, rcond=None)[0]
                        except np.linalg.LinAlgError:
                            break

                    if not np.all(np.isfinite(x1)):
                        if rlx > 0.0625:
                            rlx *= 0.5
                            x = np.full(n_dirs, np.log(1.0 / (2 * np.pi)))
                            count = 0
                            continue
                        else:
                            break

                    stddiff = np.std(x - x1)
                    x_new = (1 - rlx) * x + rlx * x1

                    # Check for divergence or non-finite values
                    if count > self.max_iter or not np.all(np.isfinite(x_new)):
                        if rlx > 0.0625:
                            rlx *= 0.5
                            x = np.full(n_dirs, np.log(1.0 / (2 * np.pi)))
                            count = 0
                        else:
                            if n_model > 1:
                                keepgoing = False
                            break
                    else:
                        x = x_new
                        if stddiff < 0.001:
                            converged = True

                if not converged:
                    if n_model > 1:
                        keepgoing = False
                    continue

                # Compute ABIC for this regularization
                # sig2 = (||A2*x - B2||^2 + u*||dd*x||^2) / M
                residual_norm = np.linalg.norm(A2 @ x - B2)
                reg_norm = np.linalg.norm(dd @ x)
                sig2 = (residual_norm**2 + u * reg_norm**2) / M

                # ABIC = M*(log(2*pi*sig2) + 1) - k*log(u^2) + sum(log(diag(TA)^2))
                diag_TA = np.diag(TA)
                diag_TA = np.where(np.abs(diag_TA) < 1e-20, 1e-20, diag_TA)
                abic = (
                    M * (np.log(2 * np.pi * sig2) + 1)
                    - n_dirs * np.log(u * u)
                    + np.sum(np.log(diag_TA**2))
                )

                ABIC_values.append(abic)
                x_held.append(x.copy())

                # Check if ABIC is increasing (stop if worse)
                if len(ABIC_values) > 1:
                    if ABIC_values[-1] > ABIC_values[-2]:
                        keepgoing = False

            # Select best model
            if len(ABIC_values) > 0:
                best_idx = np.argmin(ABIC_values)
                x = x_held[best_idx]
            else:
                x = np.full(n_dirs, np.log(1.0 / (2 * np.pi)))

            # Reconstruct spectrum: G = exp(x), with clipping for stability
            x_clipped = np.clip(x, -20, 20)
            G = np.exp(x_clipped)

            # Normalize to sum=1 (directional distribution)
            total = np.sum(G)
            if total > 0 and np.isfinite(total):
                SG = G / total
            else:
                SG = np.ones(n_dirs) / n_dirs

            S[ff, :] = SG

        return S

    def _build_second_derivative(self, n: int) -> NDArray[np.floating]:
        """Build second-derivative regularization matrix with circular boundary.

        The matrix implements: d^2/dtheta^2 with periodic boundary conditions.
        dd[i,i] = 1, dd[i,i-1] = -2, dd[i,i-2] = 1

        Args:
            n: Size of the matrix (number of directions).

        Returns:
            Second derivative matrix [n x n].
        """
        dd = np.zeros((n, n))

        # Main diagonal
        dd += np.diag(np.ones(n))

        # -2 on first sub-diagonal (with wrap)
        dd += np.diag(-2 * np.ones(n - 1), -1)
        dd[0, n - 1] = -2

        # +1 on second sub-diagonal (with wrap)
        dd += np.diag(np.ones(n - 2), -2)
        dd[0, n - 2] = 1
        dd[1, n - 1] = 1

        return dd
