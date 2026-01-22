"""Extended Maximum Entropy Principle (EMEP) for directional spectrum estimation.

The EMEP uses an exponential model with iterative coefficient estimation and
automatic model order selection via the Akaike Information Criterion (AIC).

Reference:
    Hashimoto, N. (1997) "Analysis of the directional wave spectrum from
    field data" in Advances in Coastal Engineering Vol. 3, World Scientific.
"""

import numpy as np
from numpy.typing import NDArray

from .base import EstimationMethodBase


class EMEP(EstimationMethodBase):
    """Extended Maximum Entropy Principle method.

    The EMEP estimates the directional spectrum using an exponential model:

        G(theta) = exp(sum_n [a_n * cos(n*theta) + b_n * sin(n*theta)])

    The algorithm:
    1. Separates co- and quadrature components from cross-spectra
    2. Normalizes by standard deviations for numerical stability
    3. Iteratively solves for Fourier coefficients using gradient descent
    4. Selects optimal model order using AIC
    """

    def __init__(self, max_iter: int = 100, max_order: int | None = None):
        """Initialize EMEP method.

        Args:
            max_iter: Maximum iterations per model order.
            max_order: Maximum model order to try. If None, determined by data.
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
        ddir = 2.0 * np.pi / n_dirs

        # Direction grid in radians
        theta = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)

        # Precompute cos/sin basis for all possible orders
        max_possible_order = n_dirs // 2
        cosn = np.array([np.cos(n * theta) for n in range(1, max_possible_order + 2)])
        sinn = np.array([np.sin(n * theta) for n in range(1, max_possible_order + 2)])

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
                    # Phase from kx - need to handle the indexing properly
                    # kx is [n_freqs x n_dirs x n_sensors], need phase difference
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
            phi = np.array(phi_list)  # [M]
            Hi = np.array(H_list).T  # [n_dirs x M]

            # Determine maximum model order
            max_order = self.max_order if self.max_order else M // 2 + 1
            max_order = min(max_order, M // 2 + 1, max_possible_order)

            # Precompute cos/sin matrices for efficiency
            cosnt = np.zeros((n_dirs, M, max_order + 1))
            sinnt = np.zeros((n_dirs, M, max_order + 1))
            for eni in range(max_order + 1):
                cosnt[:, :, eni] = np.outer(np.cos((eni + 1) * theta), np.ones(M))
                sinnt[:, :, eni] = np.outer(np.sin((eni + 1) * theta), np.ones(M))

            Phione = np.outer(np.ones(n_dirs), phi)  # [n_dirs x M]

            # Model order selection with AIC
            AIC_values = []
            a1_held = []
            b1_held = []
            keepgoing = True
            order = 0

            while keepgoing and order < max_order:
                order += 1

                # Initialize coefficients for this order
                a1 = np.zeros(order)
                b1 = np.zeros(order)

                rlx = 1.0
                count = 0
                converged = False

                while not converged:
                    count += 1

                    # Compute exponential: Fn = sum_n (a_n * cos(n*theta) + b_n * sin(n*theta))
                    Fn = np.zeros(n_dirs)
                    for n_idx in range(order):
                        Fn += a1[n_idx] * cosn[n_idx] + b1[n_idx] * sinn[n_idx]

                    Fnexp = np.exp(Fn)[:, np.newaxis] * np.ones((1, M))  # [n_dirs x M]
                    PhiHF = (Phione - Hi) * Fnexp

                    # Compute Z
                    sum_Fnexp = np.sum(Fnexp, axis=0)
                    sum_PhiHF = np.sum(PhiHF, axis=0)
                    sum_PhiHF = np.where(np.abs(sum_PhiHF) < 1e-20, 1e-20, sum_PhiHF)
                    Z = sum_PhiHF / sum_Fnexp

                    # Build gradient matrices X and Y
                    X = np.zeros((order, M))
                    Y = np.zeros((order, M))

                    for eni in range(order):
                        sum_Fnexp_cos = np.sum(Fnexp * cosnt[:, :, eni], axis=0)
                        sum_PhiHF_cos = np.sum(PhiHF * cosnt[:, :, eni], axis=0)
                        sum_Fnexp_sin = np.sum(Fnexp * sinnt[:, :, eni], axis=0)
                        sum_PhiHF_sin = np.sum(PhiHF * sinnt[:, :, eni], axis=0)

                        X[eni, :] = Z * (sum_Fnexp_cos / sum_Fnexp - sum_PhiHF_cos / sum_PhiHF)
                        Y[eni, :] = Z * (sum_Fnexp_sin / sum_Fnexp - sum_PhiHF_sin / sum_PhiHF)

                    # Build coefficient matrix C and solve
                    C_mat = np.hstack([X.T, Y.T])  # [M x 2*order]

                    try:
                        out, _, _, _ = np.linalg.lstsq(C_mat, Z, rcond=None)
                    except np.linalg.LinAlgError:
                        break

                    a2 = out[:order]
                    b2 = out[order : 2 * order]

                    # Check for divergence
                    if (
                        np.any(np.abs(a2) > 100)
                        or np.any(np.abs(b2) > 100)
                        or count > self.max_iter
                    ):
                        if rlx > 0.0625:
                            rlx *= 0.5
                            count = 0
                            a1 = np.zeros(order)
                            b1 = np.zeros(order)
                        else:
                            keepgoing = False
                            break
                    else:
                        a1 = a1 + rlx * a2
                        b1 = b1 + rlx * b2

                        # Check convergence
                        if np.max(np.abs(a2)) < 0.01 and np.max(np.abs(b2)) < 0.01:
                            converged = True

                if not converged and not keepgoing:
                    break

                # Compute AIC for this model order
                error = Z - X.T @ a2 - Y.T @ b2
                var_error = np.var(error)
                if var_error > 0:
                    aic = M * (np.log(2 * np.pi * var_error) + 1) + 4 * order + 2
                else:
                    aic = np.inf

                AIC_values.append(aic)
                a1_held.append(a1.copy())
                b1_held.append(b1.copy())

                # Check if AIC is increasing (stop if worse)
                if len(AIC_values) > 1:
                    if np.isnan(aic) or aic > AIC_values[-2]:
                        keepgoing = False

            # Select best model
            if len(AIC_values) > 0:
                best_idx = np.argmin(AIC_values)
                a1 = a1_held[best_idx]
                b1 = b1_held[best_idx]
                best_order = len(a1)
            else:
                a1 = np.array([0.0])
                b1 = np.array([0.0])
                best_order = 1

            # Reconstruct spectrum: G = exp(sum_n [a_n * cos + b_n * sin])
            Fn = np.zeros(n_dirs)
            for n_idx in range(best_order):
                Fn += a1[n_idx] * cosn[n_idx] + b1[n_idx] * sinn[n_idx]
            G = np.exp(Fn)

            # Normalize to sum=1 (directional distribution)
            SG = G / np.sum(G)
            S[ff, :] = SG

        return S
