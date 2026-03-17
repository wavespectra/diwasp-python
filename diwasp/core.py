"""Core driver function for directional wave spectrum analysis.

This module provides the main entry point `dirspec` that orchestrates
the directional spectrum estimation pipeline.
"""

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from .methods import BDM, DFTM, EMLM, EMEP, IMLM, EstimationMethodBase
from .methods.base import compute_kx
from .transfer import compute_transfer_matrix
from .types import (
    EstimationMethod,
    EstimationParameters,
    InstrumentData,
    SpectralInfo,
    SpectralMatrix,
)
from .utils import (
    compute_csd_matrix,
    detrend_data,
    directional_spread,
    frequency_to_angular,
    hsig,
    mean_direction,
    peak_direction,
    peak_frequency,
    wavenumber,
)

if TYPE_CHECKING:
    pass


# Mapping from estimation method enum to class
METHOD_CLASSES: dict[EstimationMethod, type[EstimationMethodBase]] = {
    EstimationMethod.DFTM: DFTM,
    EstimationMethod.EMLM: EMLM,
    EstimationMethod.IMLM: IMLM,
    EstimationMethod.EMEP: EMEP,
    EstimationMethod.BDM: BDM,
}


def dirspec(
    instrument_data: InstrumentData,
    estimation_params: EstimationParameters | None = None,
    freqs: NDArray[np.floating] | None = None,
    dirs: NDArray[np.floating] | None = None,
    verbose: int = 1,
) -> tuple[SpectralMatrix, SpectralInfo]:
    """Estimate directional wave spectrum from multi-sensor measurements.

    This is the main driver function for directional spectrum analysis.
    It orchestrates the following pipeline:

    1. Validate and preprocess input data
    2. Compute cross-spectral density matrix
    3. Calculate wavenumbers from dispersion relation
    4. Compute transfer functions for each sensor
    5. Calculate spatial phase lags
    6. Apply selected estimation method
    7. Interpolate to output frequency/direction grid
    8. Apply optional smoothing
    9. Calculate spectral statistics

    Args:
        instrument_data: Sensor measurements and configuration.
        estimation_params: Analysis parameters. If None, uses defaults.
        freqs: Output frequency grid in Hz. If None, uses frequencies from CSD.
        dirs: Output direction grid in degrees. If None, uses uniform 0-360.
        verbose: Verbosity level (0=silent, 1=normal, 2=detailed).

    Returns:
        Tuple of (SpectralMatrix, SpectralInfo) containing the estimated
        directional spectrum and computed statistics.

    Example:
        >>> from diwasp import dirspec, InstrumentData, SensorType
        >>> import numpy as np
        >>>
        >>> # Create instrument data
        >>> data = np.random.randn(1024, 3)  # 3 sensors, 1024 samples
        >>> layout = np.array([[0, 10, 0], [0, 0, 10], [10, 10, 10]])  # x, y, z
        >>> datatypes = [SensorType.PRES, SensorType.VELX, SensorType.VELY]
        >>>
        >>> id = InstrumentData(
        ...     data=data,
        ...     layout=layout.T,
        ...     datatypes=datatypes,
        ...     depth=20.0,
        ...     fs=2.0
        ... )
        >>>
        >>> # Estimate spectrum
        >>> spectrum, info = dirspec(id)
    """
    # Default estimation parameters
    if estimation_params is None:
        estimation_params = EstimationParameters()

    # Validate inputs
    _validate_inputs(instrument_data, estimation_params)

    if verbose >= 1:
        print(f"DIWASP Directional Spectrum Analysis")
        print(f"Method: {estimation_params.method.value}")
        print(f"Sensors: {instrument_data.n_sensors}")
        print(f"Samples: {instrument_data.n_samples}")

    # Single-sensor fallback: return uniform directional spectrum
    if instrument_data.n_sensors == 1:
        return _single_sensor_spectrum(instrument_data, estimation_params, freqs, dirs)

    # Step 1: Detrend data
    if verbose >= 2:
        print("Detrending data...")
    data = detrend_data(instrument_data.data)

    # Step 2: Determine FFT length
    nfft = estimation_params.nfft
    if nfft is None:
        # Auto-calculate: use power of 2 close to data length
        nfft = min(instrument_data.n_samples, 256)
        nfft = int(2 ** np.floor(np.log2(nfft)))

    if verbose >= 2:
        print(f"FFT length: {nfft}")

    # Step 3: Compute cross-spectral density matrix
    if verbose >= 2:
        print("Computing cross-spectral density...")
    csd_freqs, csd_matrix = compute_csd_matrix(
        data,
        fs=instrument_data.fs,
        nfft=nfft,
    )

    # Step 4: Set up output grids
    if freqs is None:
        # Use frequencies from CSD, excluding DC and very low frequencies
        min_freq = 0.04  # Typical minimum for ocean waves
        freq_mask = csd_freqs >= min_freq
        freqs = csd_freqs[freq_mask]
        csd_matrix = csd_matrix[freq_mask, :, :]
    else:
        freqs = np.asarray(freqs)
        # Interpolate CSD to requested frequencies
        csd_matrix = _interpolate_csd(csd_matrix, csd_freqs, freqs)

    if dirs is None:
        dirs = np.linspace(0, 360, estimation_params.dres, endpoint=False)
    else:
        dirs = np.asarray(dirs)

    n_freqs = len(freqs)
    n_dirs = len(dirs)

    if verbose >= 2:
        print(f"Frequency range: {freqs[0]:.3f} - {freqs[-1]:.3f} Hz ({n_freqs} bins)")
        print(f"Direction range: {dirs[0]:.1f} - {dirs[-1]:.1f} deg ({n_dirs} bins)")

    # Step 5: Calculate wavenumbers
    if verbose >= 2:
        print("Calculating wavenumbers...")
    sigma = frequency_to_angular(freqs)
    k = wavenumber(sigma, instrument_data.depth)

    # Step 6: Set up direction grid in radians
    theta = np.deg2rad(dirs)

    # Step 7: Compute transfer functions
    if verbose >= 2:
        print("Computing transfer functions...")
    sensor_z = instrument_data.layout[2, :]  # z-coordinates
    transfer_matrix = compute_transfer_matrix(
        instrument_data.datatypes,
        sensor_z,
        sigma,
        k,
        theta,
        instrument_data.depth,
    )

    # Step 8: Compute spatial phase lags
    sensor_x = instrument_data.layout[0, :]
    sensor_y = instrument_data.layout[1, :]
    kx = compute_kx(k, theta, sensor_x, sensor_y)

    # Step 9: Compute surface-elevation-equivalent spectra for energy scaling
    # Following MATLAB: Ss(m,:) = Sxps ./ (max(tfn') .* conj(max(tfn')))
    # This divides raw sensor auto-spectra by the maximum transfer function
    # squared to correct for depth attenuation
    if verbose >= 2:
        print("Computing surface-equivalent spectra...")

    Ss = np.zeros((instrument_data.n_sensors, n_freqs))
    for i in range(instrument_data.n_sensors):
        # Get maximum transfer function magnitude across all directions for this sensor
        # transfer_matrix shape: [n_freqs x n_dirs x n_sensors]
        H_max = np.max(np.abs(transfer_matrix[:, :, i]), axis=1)  # [n_freqs]

        # Prevent division by very small values
        H_max = np.maximum(H_max, 0.1)

        # Compute surface-elevation-equivalent spectrum
        # Ss = raw_auto_spectrum / |H_max|^2
        Ss[i, :] = np.real(csd_matrix[:, i, i]) / (H_max * H_max)

    # Step 10: Apply estimation method
    if verbose >= 1:
        print(f"Estimating spectrum using {estimation_params.method.value}...")

    method_class = METHOD_CLASSES[estimation_params.method]
    method = method_class(max_iter=estimation_params.iter)
    S_norm = method.estimate(csd_matrix, transfer_matrix, kx)

    # Step 11: Scale spectrum to preserve energy
    # Use the first sensor's surface-equivalent spectrum (following MATLAB convention)
    # This ensures the total energy is in surface elevation units
    S = np.zeros((n_freqs, n_dirs))
    ddir = 360.0 / n_dirs
    for fi in range(n_freqs):
        S[fi, :] = S_norm[fi, :] * Ss[0, fi] / ddir

    # Step 12: Apply smoothing if requested
    if estimation_params.smooth:
        if verbose >= 2:
            print("Smoothing spectrum...")
        S = _smooth_spectrum(S)

    # Step 13: Create output spectral matrix
    spectrum = SpectralMatrix(
        freqs=freqs,
        dirs=dirs,
        S=S,
        xaxisdir=90.0,  # East
        funit="hz",
        dunit="cart",
    )

    # Step 14: Compute spectral statistics
    info = _compute_spectral_info(spectrum)

    return spectrum, info


def _validate_inputs(
    instrument_data: InstrumentData,
    estimation_params: EstimationParameters,
) -> None:
    """Validate input data and parameters."""
    if instrument_data.n_sensors < 1:
        raise ValueError("At least 1 sensor required")

    if instrument_data.n_samples < 64:
        raise ValueError("At least 64 samples required")

    if estimation_params.nfft is not None:
        if estimation_params.nfft > instrument_data.n_samples:
            raise ValueError(
                f"nfft ({estimation_params.nfft}) cannot exceed "
                f"number of samples ({instrument_data.n_samples})"
            )


def _single_sensor_spectrum(
    instrument_data: InstrumentData,
    estimation_params: EstimationParameters,
    freqs: NDArray[np.floating] | None,
    dirs: NDArray[np.floating] | None,
) -> tuple["SpectralMatrix", "SpectralInfo"]:
    """Compute a non-directional (uniform) spectrum from a single sensor.

    Applies the sensor transfer function to convert raw measurements to
    surface-elevation-equivalent spectra before distributing uniformly
    across directions.
    """
    nfft = estimation_params.nfft
    if nfft is None:
        nfft = min(instrument_data.n_samples, 256)
        nfft = int(2 ** np.floor(np.log2(nfft)))

    data = detrend_data(instrument_data.data)
    csd_freqs, csd_matrix = compute_csd_matrix(data, fs=instrument_data.fs, nfft=nfft)

    if freqs is None:
        min_freq = 0.04
        freq_mask = csd_freqs >= min_freq
        freqs = csd_freqs[freq_mask]
        S_raw = np.real(csd_matrix[freq_mask, 0, 0])
    else:
        freqs = np.asarray(freqs)
        S_raw = np.interp(freqs, csd_freqs, np.real(csd_matrix[:, 0, 0]))

    # Apply transfer function correction: divide by |H|^2 averaged over directions
    # Use a representative direction (0 rad) for the magnitude, then take max over
    # a few directions so the correction is direction-averaged
    sigma = frequency_to_angular(freqs)
    k = wavenumber(sigma, instrument_data.depth)
    sensor_z = instrument_data.layout[2, 0]
    theta_sample = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    from .transfer import get_transfer_function

    tf = get_transfer_function(instrument_data.datatypes[0])
    H_vals = tf(sigma, k, theta_sample, instrument_data.depth, sensor_z)
    # H_vals shape: [n_freqs, n_theta_sample]; take mean |H|^2 over directions
    H2_mean = np.mean(np.abs(H_vals) ** 2, axis=1)
    H2_mean = np.maximum(H2_mean, 1e-6)
    S_1d = S_raw / H2_mean

    if dirs is None:
        dirs = np.linspace(0, 360, estimation_params.dres, endpoint=False)
    else:
        dirs = np.asarray(dirs)

    n_dirs = len(dirs)
    ddir = 360.0 / n_dirs
    S = np.outer(S_1d, np.ones(n_dirs)) / (n_dirs * ddir)

    spectrum = SpectralMatrix(freqs=freqs, dirs=dirs, S=S, xaxisdir=90.0, funit="hz", dunit="cart")
    info = _compute_spectral_info(spectrum)
    return spectrum, info


def _interpolate_csd(
    csd_matrix: NDArray[np.complexfloating],
    freqs_in: NDArray[np.floating],
    freqs_out: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Interpolate CSD matrix to new frequency grid."""
    n_sensors = csd_matrix.shape[1]
    n_freqs_out = len(freqs_out)

    csd_out = np.zeros((n_freqs_out, n_sensors, n_sensors), dtype=np.complex128)

    for i in range(n_sensors):
        for j in range(n_sensors):
            # Interpolate real and imaginary parts separately
            real_interp = np.interp(freqs_out, freqs_in, np.real(csd_matrix[:, i, j]))
            imag_interp = np.interp(freqs_out, freqs_in, np.imag(csd_matrix[:, i, j]))
            csd_out[:, i, j] = real_interp + 1j * imag_interp

    return csd_out


def _smooth_spectrum(
    S: NDArray[np.floating],
    kernel: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Apply 2D smoothing to spectrum.

    Args:
        S: Input spectrum [n_freqs x n_dirs].
        kernel: Smoothing kernel. If None, uses default.

    Returns:
        Smoothed spectrum.
    """
    from scipy import ndimage

    if kernel is None:
        # Default smoothing kernel (from DIWASP)
        kernel = np.array([[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]])
        kernel = kernel / np.sum(kernel)

    # Apply convolution with wrap mode for circular direction dimension
    S_smooth = ndimage.convolve(S, kernel, mode="wrap")

    return S_smooth


def _compute_spectral_info(spectrum: SpectralMatrix) -> SpectralInfo:
    """Compute spectral statistics."""
    return SpectralInfo(
        hsig=hsig(spectrum.S, spectrum.freqs, spectrum.dirs),
        tp=1.0 / peak_frequency(spectrum.S, spectrum.freqs, spectrum.dirs),
        fp=peak_frequency(spectrum.S, spectrum.freqs, spectrum.dirs),
        dp=peak_direction(spectrum.S, spectrum.freqs, spectrum.dirs),
        dm=mean_direction(spectrum.S, spectrum.freqs, spectrum.dirs),
        spread=directional_spread(spectrum.S, spectrum.freqs, spectrum.dirs),
    )
