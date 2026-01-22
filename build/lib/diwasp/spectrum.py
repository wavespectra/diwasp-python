"""Spectrum manipulation and generation utilities.

This module provides functions for:
- Interpolating spectra to different frequency/direction grids
- Generating synthetic directional spectra
- Creating synthetic sensor data for testing
"""

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

from .types import InstrumentData, SensorType, SpectralMatrix
from .utils import G, frequency_to_angular, wavenumber


def interpspec(
    spectrum: SpectralMatrix,
    freqs_out: NDArray[np.floating] | None = None,
    dirs_out: NDArray[np.floating] | None = None,
) -> SpectralMatrix:
    """Interpolate spectrum to a new frequency/direction grid.

    Uses 2D interpolation in a Cartesian spectral basis to handle
    the circular nature of the direction dimension.

    Args:
        spectrum: Input spectral matrix.
        freqs_out: Output frequency grid in Hz. If None, keeps original.
        dirs_out: Output direction grid in degrees. If None, keeps original.

    Returns:
        Interpolated spectral matrix.
    """
    if freqs_out is None:
        freqs_out = spectrum.freqs.copy()
    if dirs_out is None:
        dirs_out = spectrum.dirs.copy()

    # Convert to Cartesian spectral basis for interpolation
    # This handles the circular direction dimension properly
    dirs_rad = np.deg2rad(spectrum.dirs)
    dirs_out_rad = np.deg2rad(dirs_out)

    # Create meshgrid for input
    F_in, D_in = np.meshgrid(spectrum.freqs, dirs_rad, indexing="ij")

    # Convert to Cartesian coordinates (freq*cos, freq*sin)
    X_in = F_in * np.cos(D_in)
    Y_in = F_in * np.sin(D_in)

    # Flatten for interpolation
    points = np.column_stack([X_in.ravel(), Y_in.ravel()])
    values = spectrum.S.ravel()

    # Create output grid
    F_out, D_out = np.meshgrid(freqs_out, dirs_out_rad, indexing="ij")
    X_out = F_out * np.cos(D_out)
    Y_out = F_out * np.sin(D_out)

    # Interpolate
    S_out = interpolate.griddata(
        points,
        values,
        (X_out, Y_out),
        method="linear",
        fill_value=0.0,
    )

    # Handle any NaN values
    S_out = np.nan_to_num(S_out, nan=0.0)

    return SpectralMatrix(
        freqs=freqs_out,
        dirs=dirs_out,
        S=S_out,
        xaxisdir=spectrum.xaxisdir,
        funit=spectrum.funit,
        dunit=spectrum.dunit,
    )


def makespec(
    freq_range: tuple[float, float, float],
    theta: float | list[float],
    spread: float | list[float],
    weights: float | list[float] | None = None,
    hsig: float = 1.0,
    depth: float = 20.0,
    n_freqs: int = 64,
    n_dirs: int = 180,
) -> SpectralMatrix:
    """Generate a synthetic directional wave spectrum.

    Creates a spectrum using the TMA spectral model for frequency distribution
    and cosine power spreading for directionality.

    Args:
        freq_range: Tuple of (low_freq, peak_freq, high_freq) in Hz.
        theta: Mean wave direction(s) in degrees. Can be a list for
            multi-modal spectra.
        spread: Directional spreading parameter(s). Higher values = narrower
            spread. Typical range: 25-100.
        weights: Relative weights for each directional component. If None,
            uses equal weights.
        hsig: Target significant wave height in meters.
        depth: Water depth in meters.
        n_freqs: Number of frequency bins.
        n_dirs: Number of direction bins.

    Returns:
        Synthetic directional spectrum.

    Example:
        >>> # Single modal spectrum
        >>> spectrum = makespec(
        ...     freq_range=(0.05, 0.1, 0.3),
        ...     theta=45.0,
        ...     spread=50.0,
        ...     hsig=2.0
        ... )
        >>>
        >>> # Bimodal spectrum (wind sea + swell)
        >>> spectrum = makespec(
        ...     freq_range=(0.04, 0.08, 0.3),
        ...     theta=[270.0, 180.0],
        ...     spread=[25.0, 75.0],
        ...     weights=[0.3, 0.7],
        ...     hsig=3.0
        ... )
    """
    # Ensure inputs are lists
    if isinstance(theta, (int, float)):
        theta = [theta]
    if isinstance(spread, (int, float)):
        spread = [spread]
    if weights is None:
        weights = [1.0 / len(theta)] * len(theta)
    elif isinstance(weights, (int, float)):
        weights = [weights]

    # Normalize weights
    weights = np.array(weights) / np.sum(weights)

    # Create frequency and direction grids
    freqs = np.linspace(freq_range[0], freq_range[2], n_freqs)
    dirs = np.linspace(0, 360, n_dirs, endpoint=False)

    # Calculate TMA frequency spectrum
    S_f = _tma_spectrum(freqs, freq_range[1], depth)

    # Build directional spectrum
    S = np.zeros((n_freqs, n_dirs))

    for th, sp, w in zip(theta, spread, weights):
        # Create directional distribution for this component
        D = _cosine_spread(dirs, th, sp)

        # Combine frequency and direction
        S += w * np.outer(S_f, D)

    # Scale to target Hsig
    # Hsig = 4 * sqrt(m0), m0 = integral of S
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    ddir = dirs[1] - dirs[0] if len(dirs) > 1 else 1.0
    m0_current = np.sum(S) * df * ddir
    hsig_current = 4.0 * np.sqrt(m0_current)

    if hsig_current > 0:
        scale = (hsig / hsig_current) ** 2
        S = S * scale

    return SpectralMatrix(
        freqs=freqs,
        dirs=dirs,
        S=S,
        xaxisdir=90.0,
        funit="hz",
        dunit="cart",
    )


def make_wave_data(
    spectrum: SpectralMatrix,
    instrument_data: InstrumentData,
    n_samples: int,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.floating]:
    """Generate synthetic sensor data from a directional spectrum.

    Creates time series measurements that would be observed by sensors
    measuring the wave field described by the spectrum.

    Args:
        spectrum: Directional wave spectrum.
        instrument_data: Sensor configuration (used for layout and types).
        n_samples: Number of time samples to generate.
        noise_level: Standard deviation of Gaussian noise to add.
        seed: Random seed for reproducibility.

    Returns:
        Synthetic sensor data [n_samples x n_sensors].
    """
    if seed is not None:
        np.random.seed(seed)

    n_sensors = instrument_data.n_sensors
    fs = instrument_data.fs
    depth = instrument_data.depth

    # Time vector
    t = np.arange(n_samples) / fs

    # Initialize output
    data = np.zeros((n_samples, n_sensors))

    # Frequency and direction grids
    freqs = spectrum.freqs
    dirs_rad = np.deg2rad(spectrum.dirs)

    # Calculate wavenumbers
    sigma = frequency_to_angular(freqs)
    k = wavenumber(sigma, depth)

    # Generate random phases for each frequency/direction component
    phases = np.random.uniform(0, 2 * np.pi, (len(freqs), len(dirs_rad)))

    # Amplitude from spectrum
    df = np.mean(np.diff(freqs)) if len(freqs) > 1 else 1.0
    ddir = np.mean(np.diff(dirs_rad)) if len(dirs_rad) > 1 else 1.0
    amplitudes = np.sqrt(2 * spectrum.S * df * ddir)

    # Generate wave components for each sensor
    from .transfer import get_transfer_function

    for si in range(n_sensors):
        sensor_type = instrument_data.datatypes[si]
        x = instrument_data.layout[0, si]
        y = instrument_data.layout[1, si]
        z = instrument_data.layout[2, si]

        transfer_func = get_transfer_function(sensor_type)

        # Loop over frequency and direction
        for fi, f in enumerate(freqs):
            for di, d in enumerate(dirs_rad):
                if amplitudes[fi, di] < 1e-10:
                    continue

                # Transfer function for this sensor
                H = transfer_func(
                    np.array([sigma[fi]]),
                    np.array([k[fi]]),
                    np.array([d]),
                    depth,
                    z,
                )

                # Phase from position
                kx = k[fi] * (x * np.cos(d) + y * np.sin(d))

                # Generate signal
                wave = (
                    amplitudes[fi, di]
                    * np.abs(H[0, 0])
                    * np.cos(2 * np.pi * f * t - kx + phases[fi, di] + np.angle(H[0, 0]))
                )

                data[:, si] += wave

    # Add noise
    if noise_level > 0:
        data += np.random.normal(0, noise_level, data.shape)

    return data


def _tma_spectrum(
    freqs: NDArray[np.floating],
    fp: float,
    depth: float,
    gamma: float = 3.3,
) -> NDArray[np.floating]:
    """Calculate TMA frequency spectrum.

    TMA spectrum (Bouws et al., 1985) is a depth-limited modification
    of the JONSWAP spectrum.

    Args:
        freqs: Frequency array in Hz.
        fp: Peak frequency in Hz.
        depth: Water depth in meters.
        gamma: Peak enhancement factor (default 3.3 for JONSWAP).

    Returns:
        Spectral density at each frequency.
    """
    # JONSWAP parameters
    alpha = 0.0081  # Phillips constant

    # JONSWAP spectrum
    sigma = np.where(freqs <= fp, 0.07, 0.09)

    # Peak enhancement
    r = np.exp(-0.5 * ((freqs / fp - 1) / sigma) ** 2)
    enhancement = gamma**r

    # PM-type spectrum
    S = alpha * G**2 / (2 * np.pi) ** 4 / freqs**5
    S = S * np.exp(-1.25 * (fp / freqs) ** 4)
    S = S * enhancement

    # TMA transformation (depth limitation)
    sigma_f = frequency_to_angular(freqs)
    k = wavenumber(sigma_f, depth)
    kd = k * depth

    # Kitaigorodskii shape factor
    phi = np.where(
        kd <= 1, 0.5 * kd**2, 1 - 0.5 * (2 - kd) ** 2 * (kd < 2) + (kd >= 2) * 1.0
    )

    S = S * phi

    # Handle edge cases
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    return S


def _cosine_spread(
    dirs: NDArray[np.floating],
    theta_mean: float,
    s: float,
) -> NDArray[np.floating]:
    """Calculate cosine power directional spreading function.

    D(theta) = A * cos^(2s)((theta - theta_mean) / 2)

    where A is a normalization constant and s is the spreading parameter.

    Args:
        dirs: Direction array in degrees.
        theta_mean: Mean wave direction in degrees.
        s: Spreading parameter (higher = narrower spread).

    Returns:
        Directional distribution (integrates to 1 over 360 degrees).
    """
    # Convert to radians
    theta_diff = np.deg2rad(dirs - theta_mean)

    # Cosine power spreading
    D = np.cos(theta_diff / 2) ** (2 * s)

    # Handle negative values (from angles > 180 from mean)
    D = np.maximum(D, 0.0)

    # Normalize to integrate to 1
    ddir = np.mean(np.diff(dirs)) if len(dirs) > 1 else 1.0
    integral = np.sum(D) * ddir / 360.0

    if integral > 0:
        D = D / (integral * 360.0)

    return D
