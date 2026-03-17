"""Utility functions for DIWASP.

This module provides core mathematical utilities used throughout the package:
- Wavenumber calculation (dispersion relation)
- Cross-spectral density computation
- Significant wave height calculation
- Spectral statistics
"""

import numpy as np
from numpy.typing import NDArray
from scipy import signal

# Gravitational acceleration (m/s^2)
G = 9.81


def wavenumber(
    sigma: NDArray[np.floating] | float,
    depth: float,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> NDArray[np.floating]:
    """Calculate wavenumber from angular frequency using dispersion relation.

    Solves the linear dispersion relation: sigma^2 = g * k * tanh(k * d)
    using Newton-Raphson iteration.

    Args:
        sigma: Angular frequency in rad/s (scalar or array).
        depth: Water depth in meters.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        Wavenumber k in rad/m (same shape as sigma).

    Raises:
        ValueError: If iteration does not converge.
    """
    sigma = np.atleast_1d(np.asarray(sigma, dtype=np.float64))

    # Initial guess using deep water approximation: k = sigma^2 / g
    k = sigma**2 / G

    # Newton-Raphson iteration
    for _ in range(max_iter):
        tanh_kd = np.tanh(k * depth)
        # f(k) = sigma^2 - g*k*tanh(k*d)
        f = sigma**2 - G * k * tanh_kd
        # f'(k) = -g*tanh(k*d) - g*k*d*sech^2(k*d)
        sech2_kd = 1.0 / np.cosh(k * depth) ** 2
        fp = -G * tanh_kd - G * k * depth * sech2_kd

        # Newton step
        dk = -f / fp

        # Update k with safety check for negative values
        k_new = k + dk
        k_new = np.maximum(k_new, 1e-10)  # Prevent negative wavenumbers

        # Check convergence
        if np.all(np.abs(dk) < tol * np.abs(k_new)):
            return k_new

        k = k_new

    raise ValueError(f"Wavenumber iteration did not converge after {max_iter} iterations")


def frequency_to_angular(freq: NDArray[np.floating] | float) -> NDArray[np.floating]:
    """Convert frequency in Hz to angular frequency in rad/s.

    Args:
        freq: Frequency in Hz.

    Returns:
        Angular frequency in rad/s.
    """
    return 2.0 * np.pi * np.asarray(freq)


def angular_to_frequency(sigma: NDArray[np.floating] | float) -> NDArray[np.floating]:
    """Convert angular frequency in rad/s to frequency in Hz.

    Args:
        sigma: Angular frequency in rad/s.

    Returns:
        Frequency in Hz.
    """
    return np.asarray(sigma) / (2.0 * np.pi)


def compute_csd(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    fs: float,
    nfft: int | None = None,
    window: str = "hann",
    overlap: float = 0.5,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Compute cross-spectral density between two signals.

    Uses Welch's method with specified windowing and overlap.

    Args:
        x: First signal array.
        y: Second signal array.
        fs: Sampling frequency in Hz.
        nfft: FFT length. If None, uses length of x.
        window: Window function name (default 'hann').
        overlap: Overlap fraction between segments (default 0.5).

    Returns:
        Tuple of (frequencies, csd) where frequencies is in Hz and
        csd is the complex cross-spectral density.
    """
    if nfft is None:
        nfft = len(x)

    noverlap = int(nfft * overlap)

    freqs, csd = signal.csd(
        x,
        y,
        fs=fs,
        window=window,
        nperseg=nfft,
        noverlap=noverlap,
        return_onesided=True,
    )

    return freqs, csd


def compute_csd_matrix(
    data: NDArray[np.floating],
    fs: float,
    nfft: int | None = None,
    window: str = "hann",
    overlap: float = 0.5,
) -> tuple[NDArray[np.floating], NDArray[np.complexfloating]]:
    """Compute cross-spectral density matrix for multi-sensor data.

    Computes the CSD between all pairs of sensors.

    Args:
        data: Sensor data array [n_samples x n_sensors].
        fs: Sampling frequency in Hz.
        nfft: FFT length. If None, uses number of samples.
        window: Window function name.
        overlap: Overlap fraction between segments.

    Returns:
        Tuple of (frequencies, csd_matrix) where csd_matrix has shape
        [n_freqs x n_sensors x n_sensors] and contains complex CSD values.
    """
    n_samples, n_sensors = data.shape

    if nfft is None:
        nfft = n_samples

    # Compute CSD for first pair to get frequency array
    freqs, _ = compute_csd(data[:, 0], data[:, 0], fs, nfft, window, overlap)
    n_freqs = len(freqs)

    # Initialize CSD matrix
    csd_matrix = np.zeros((n_freqs, n_sensors, n_sensors), dtype=np.complex128)

    # Compute CSD for all sensor pairs
    for i in range(n_sensors):
        for j in range(n_sensors):
            _, csd = compute_csd(data[:, i], data[:, j], fs, nfft, window, overlap)
            csd_matrix[:, i, j] = csd

    return freqs, csd_matrix


def hsig(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    dirs: NDArray[np.floating],
) -> float:
    """Calculate significant wave height from directional spectrum.

    Hs = 4 * sqrt(m0) where m0 is the zeroth moment (total energy).

    Args:
        S: Spectral density matrix [n_freqs x n_dirs] in m^2/(Hz*degree).
        freqs: Frequency bins in Hz.
        dirs: Direction bins in degrees.

    Returns:
        Significant wave height in meters.
    """
    # Calculate frequency and direction resolution
    df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0
    ddir = np.mean(np.diff(dirs)) if len(dirs) > 1 else 1.0

    # Integrate spectrum (convert degrees to appropriate units)
    # S is in m^2/(Hz*degree), so integrate over freq and dir
    m0 = np.sum(S) * df * ddir

    return 4.0 * np.sqrt(m0)


def peak_frequency(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    dirs: NDArray[np.floating] | None = None,
) -> float:
    """Calculate peak frequency from directional spectrum.

    Args:
        S: Spectral density matrix [n_freqs x n_dirs].
        freqs: Frequency bins in Hz.

    Returns:
        Peak frequency in Hz.
    """
    # Sum over directions to get frequency spectrum
    S_f = np.sum(S, axis=1)

    # Find peak
    peak_idx = np.argmax(S_f)
    return float(freqs[peak_idx])


def peak_direction(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    dirs: NDArray[np.floating],
) -> float:
    """Calculate peak direction from directional spectrum.

    Peak direction is the direction at the spectral peak.

    Args:
        S: Spectral density matrix [n_freqs x n_dirs].
        freqs: Frequency bins in Hz.
        dirs: Direction bins in degrees.

    Returns:
        Peak direction in degrees.
    """
    # Find peak in 2D spectrum
    peak_idx = np.unravel_index(np.argmax(S), S.shape)
    return float(dirs[peak_idx[1]])


def mean_direction(
    S: NDArray[np.floating],
    freqs_or_dirs: NDArray[np.floating],
    dirs: NDArray[np.floating] | None = None,
) -> float:
    """Calculate energy-weighted mean direction.

    Uses circular mean to properly handle direction wrapping.

    Args:
        S: Spectral density matrix [n_freqs x n_dirs].
        freqs_or_dirs: Either direction bins in degrees (2-arg form) or
            frequency bins in Hz (3-arg form, freqs is unused).
        dirs: Direction bins in degrees (3-arg form only).

    Returns:
        Mean direction in degrees.
    """
    if dirs is None:
        dirs = freqs_or_dirs
    # Convert to radians
    dirs_rad = np.deg2rad(dirs)

    # Energy weights (sum over frequencies)
    weights = np.sum(S, axis=0)
    weights = weights / np.sum(weights)

    # Circular mean
    sin_mean = np.sum(weights * np.sin(dirs_rad))
    cos_mean = np.sum(weights * np.cos(dirs_rad))

    mean_dir = np.rad2deg(np.arctan2(sin_mean, cos_mean))

    # Normalize to [0, 360)
    return float(mean_dir % 360.0)


def directional_spread(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    dirs: NDArray[np.floating],
) -> float:
    """Calculate directional spread (standard deviation).

    Args:
        S: Spectral density matrix [n_freqs x n_dirs].
        dirs: Direction bins in degrees.

    Returns:
        Directional spread in degrees.
    """
    # Convert to radians
    dirs_rad = np.deg2rad(dirs)

    # Energy weights (sum over frequencies)
    weights = np.sum(S, axis=0)
    weights = weights / np.sum(weights)

    # Calculate circular spread
    sin_mean = np.sum(weights * np.sin(dirs_rad))
    cos_mean = np.sum(weights * np.cos(dirs_rad))

    r = np.sqrt(sin_mean**2 + cos_mean**2)

    # Circular standard deviation
    spread_rad = np.sqrt(-2.0 * np.log(r)) if r > 0 else np.pi

    return float(np.rad2deg(spread_rad))


def one_sided_directional_spread(
    S: NDArray[np.floating],
    freqs: NDArray[np.floating],
    dirs: NDArray[np.floating],
) -> float:
    """One-sided directional spread Dspr.

    Matches the wavespectra definition: the one-sided directional width
    of the spectrum in degrees.

    The formula is: dspr = sqrt(2 * R2D^2 * (1 - sqrt(a^2 + b^2) / m0))

    Where:
        - R2D = 180/pi (radians to degrees conversion)
        - a = integral of S * sin(theta) over freq and dir
        - b = integral of S * cos(theta) over freq and dir
        - m0 = integral of S over freq and dir (total energy)
        - theta = 180 + 90 - dir (wavespectra convention with theta=90)

    Args:
        S: Directional spectrum [n_freqs x n_dirs] in m^2/Hz/deg.
        freqs: Frequency array in Hz.
        dirs: Direction array in degrees.

    Returns:
        Directional spread in degrees.
    """
    R2D = 180.0 / np.pi  # Radians to degrees

    # Direction resolution (constant for circular grid)
    if len(dirs) > 1:
        dd = np.abs(np.mean(np.diff(dirs)))
    else:
        dd = 1.0

    # Frequency resolution array (like wavespectra dfarr using gradient)
    if len(freqs) > 1:
        dfarr = np.gradient(freqs)
    else:
        dfarr = np.array([1.0])

    # Convert directions to radians with wavespectra convention
    # wavespectra uses: theta = 180 + 90 - dir (with theta=90 default)
    dirs_rad = np.deg2rad(180.0 + 90.0 - dirs)

    sin_dirs = np.sin(dirs_rad)
    cos_dirs = np.cos(dirs_rad)

    # Step 1: Compute directional moments per frequency (like wavespectra momd)
    # moms[f] = sum_d(S[f,d] * sin(theta) * dd)
    moms_f = np.sum(S * sin_dirs[np.newaxis, :], axis=1) * dd  # [n_freqs]
    momc_f = np.sum(S * cos_dirs[np.newaxis, :], axis=1) * dd  # [n_freqs]

    # Step 2: Integrate over frequency (like wavespectra momf on moms/momc)
    # a = sum_f(moms[f] * df[f])
    a = np.sum(moms_f * dfarr)
    b = np.sum(momc_f * dfarr)

    # Step 3: Total energy (like wavespectra mom0)
    # First compute 1D directional spectrum: sum_f(S[f,d] * df[f]) for each d
    dir_spectrum = np.sum(S * dfarr[:, np.newaxis], axis=0)  # [n_dirs]
    # Then integrate over directions: sum_d(dir_spectrum * dd)
    m0 = np.sum(dir_spectrum) * dd

    # Directional spread formula from wavespectra
    if m0 > 0:
        r = np.sqrt(a**2 + b**2) / m0
        # Clamp r to [0, 1] to avoid sqrt of negative
        r = np.clip(r, 0.0, 1.0)
        dspr = np.sqrt(2.0 * R2D**2 * (1.0 - r))
    else:
        dspr = 0.0

    return float(dspr)


def detrend_data(data: NDArray[np.floating]) -> NDArray[np.floating]:
    """Remove linear trend from data.

    Args:
        data: Input data array [n_samples] or [n_samples x n_sensors].

    Returns:
        Detrended data array.
    """
    return signal.detrend(data, axis=0)


def direction_cart_to_naut(theta_cart: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert direction from Cartesian to nautical convention.

    Cartesian: 0 = East, counter-clockwise positive
    Nautical: 0 = North, clockwise positive, direction waves come FROM

    Args:
        theta_cart: Direction in Cartesian convention (degrees).

    Returns:
        Direction in nautical convention (degrees).
    """
    # Convert counter-clockwise from East to clockwise from North
    theta_naut = 90.0 - theta_cart
    return theta_naut % 360.0


def direction_naut_to_cart(theta_naut: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert direction from nautical to Cartesian convention.

    Args:
        theta_naut: Direction in nautical convention (degrees).

    Returns:
        Direction in Cartesian convention (degrees).
    """
    theta_cart = 90.0 - theta_naut
    return theta_cart % 360.0
