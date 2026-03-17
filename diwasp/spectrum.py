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
    spectrum: SpectralMatrix | None = None,
    instrument_data: InstrumentData | None = None,
    n_samples: int | None = None,
    noise_level: float = 0.0,
    seed: int | None = None,
    *,
    layout: NDArray[np.floating] | None = None,
    datatypes: list | None = None,
    depth: float | None = None,
    fs: float | None = None,
    duration: float | None = None,
) -> NDArray[np.floating]:
    """Generate synthetic sensor data from a directional spectrum.

    Creates time series measurements that would be observed by sensors
    measuring the wave field described by the spectrum.

    Uses FFT-based synthesis to correctly handle energy when summing
    components at the same frequency but different directions.

    Args:
        spectrum: Directional wave spectrum.
        instrument_data: Sensor configuration (used for layout and types).
            Alternatively, provide layout/datatypes/depth/fs/duration directly.
        n_samples: Number of time samples to generate.
        noise_level: Standard deviation of Gaussian noise to add.
        seed: Random seed for reproducibility.
        layout: Sensor positions [3 x n_sensors] (alternative to instrument_data).
        datatypes: List of sensor type strings (alternative to instrument_data).
        depth: Water depth in meters (alternative to instrument_data).
        fs: Sampling frequency in Hz (alternative to instrument_data).
        duration: Duration in seconds (alternative to n_samples).

    Returns:
        Synthetic sensor data [n_samples x n_sensors].
    """
    if spectrum is None:
        raise ValueError("Must provide spectrum")

    if instrument_data is None:
        if layout is None or datatypes is None or depth is None or fs is None:
            raise ValueError("Must provide either instrument_data or layout/datatypes/depth/fs")
        sensor_types = [SensorType(dt) if isinstance(dt, str) else dt for dt in datatypes]
        instrument_data = InstrumentData(
            data=np.zeros((1, len(sensor_types))),
            layout=np.asarray(layout),
            datatypes=sensor_types,
            depth=depth,
            fs=fs,
        )

    if duration is not None and n_samples is None:
        n_samples = int(duration * instrument_data.fs)
    elif n_samples is None:
        raise ValueError("Must provide either n_samples or duration")

    if seed is not None:
        np.random.seed(seed)

    n_sensors = instrument_data.n_sensors
    fs = instrument_data.fs
    depth = instrument_data.depth

    # Initialize output
    data = np.zeros((n_samples, n_sensors))

    # Frequency and direction grids from spectrum
    freqs_spec = spectrum.freqs
    dirs_rad = np.deg2rad(spectrum.dirs)
    n_dirs = len(dirs_rad)

    # Calculate wavenumbers for spectrum frequencies
    sigma_spec = frequency_to_angular(freqs_spec)
    k_spec = wavenumber(sigma_spec, depth)

    # FFT frequency grid (one-sided, positive frequencies only)
    # For n_samples, FFT has frequencies 0, df, 2*df, ... up to Nyquist
    df_fft = fs / n_samples
    n_fft_pos = n_samples // 2 + 1  # Number of positive frequency bins
    freqs_fft = np.arange(n_fft_pos) * df_fft

    # Amplitude from spectrum: A = sqrt(2 * S * df * ddir)
    # For complex FFT coefficient: |X(f)| = A * n_samples / 2
    # Note: S is in m^2/(Hz*degree), so ddir must be in degrees
    df_spec = np.mean(np.diff(freqs_spec)) if len(freqs_spec) > 1 else 1.0
    ddir_deg = np.mean(np.diff(spectrum.dirs)) if len(spectrum.dirs) > 1 else 1.0

    # Interpolate spectrum to FFT frequencies
    # For frequencies outside spectrum range, use zero
    S_interp = np.zeros((n_fft_pos, n_dirs))
    k_interp = np.zeros(n_fft_pos)
    sigma_interp = frequency_to_angular(freqs_fft)

    # Only compute for frequencies within spectrum range
    freq_mask = (freqs_fft >= freqs_spec[0]) & (freqs_fft <= freqs_spec[-1])
    if np.any(freq_mask):
        for di in range(n_dirs):
            S_interp[freq_mask, di] = np.interp(freqs_fft[freq_mask], freqs_spec, spectrum.S[:, di])
        k_interp[freq_mask] = np.interp(freqs_fft[freq_mask], freqs_spec, k_spec)

    # Generate random phases for each frequency/direction component
    phases = np.random.uniform(0, 2 * np.pi, (n_fft_pos, n_dirs))

    # Amplitude from interpolated spectrum
    # Use df_fft for proper scaling to FFT frequencies
    amplitudes = np.sqrt(2 * S_interp * df_fft * ddir_deg)

    # Generate wave data for each sensor using FFT
    from .transfer import get_transfer_function

    for si in range(n_sensors):
        sensor_type = instrument_data.datatypes[si]
        x = instrument_data.layout[0, si]
        y = instrument_data.layout[1, si]
        z = instrument_data.layout[2, si]

        transfer_func = get_transfer_function(sensor_type)

        # Build complex FFT spectrum by summing directional components
        X_fft = np.zeros(n_fft_pos, dtype=np.complex128)

        for fi in range(n_fft_pos):
            if freqs_fft[fi] == 0 or not freq_mask[fi]:
                continue

            # Get transfer function for all directions at this frequency
            H = transfer_func(
                np.array([sigma_interp[fi]]),
                np.array([k_interp[fi]]),
                dirs_rad,
                depth,
                z,
            )  # Shape: [1, n_dirs]

            # Sum contributions from all directions
            for di in range(n_dirs):
                if amplitudes[fi, di] < 1e-10:
                    continue

                # Phase from position
                kx = k_interp[fi] * (x * np.cos(dirs_rad[di]) + y * np.sin(dirs_rad[di]))

                # Complex amplitude including transfer function and spatial phase
                # The factor n_samples/2 converts amplitude to FFT coefficient
                X_fft[fi] += (
                    amplitudes[fi, di]
                    * H[0, di]
                    * np.exp(1j * (phases[fi, di] - kx))
                    * n_samples
                    / 2
                )

        # Build two-sided spectrum for IFFT
        X_full = np.zeros(n_samples, dtype=np.complex128)
        X_full[0:n_fft_pos] = X_fft

        # Conjugate symmetry for real signal (negative frequencies)
        # X[-f] = conj(X[f])
        if n_samples % 2 == 0:
            # Even length: Nyquist bin is real, don't duplicate
            X_full[n_fft_pos:] = np.conj(X_fft[-2:0:-1])
        else:
            # Odd length
            X_full[n_fft_pos:] = np.conj(X_fft[-1:0:-1])

        # IFFT to get time series
        data[:, si] = np.real(np.fft.ifft(X_full))

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
    phi = np.where(kd <= 1, 0.5 * kd**2, 1 - 0.5 * (2 - kd) ** 2 * (kd < 2) + (kd >= 2) * 1.0)

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
