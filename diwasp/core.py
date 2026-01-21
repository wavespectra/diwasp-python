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

    # Step 1: Detrend data
    if verbose >= 2:
        print("Detrending data...")
    data = detrend_data(instrument_data.data)

    # Step 2: Determine FFT length
    nfft = estimation_params.nfft
    if nfft is None:
        # Auto-calculate: use power of 2 close to data length
        nfft = min(instrument_data.n_samples, 2048)
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

    # Step 9: Apply estimation method
    if verbose >= 1:
        print(f"Estimating spectrum using {estimation_params.method.value}...")

    method_class = METHOD_CLASSES[estimation_params.method]
    method = method_class(max_iter=estimation_params.iter)
    S_norm = method.estimate(csd_matrix, transfer_matrix, kx)

    # Step 10: Scale spectrum to preserve energy
    # Get total energy from auto-spectra
    total_energy = np.zeros(n_freqs)
    for i in range(instrument_data.n_sensors):
        # Only use elevation-equivalent sensors for energy
        total_energy += np.real(csd_matrix[:, i, i])
    total_energy = total_energy / instrument_data.n_sensors

    # Scale directional distribution by frequency energy
    S = np.zeros((n_freqs, n_dirs))
    ddir = 360.0 / n_dirs
    for fi in range(n_freqs):
        S[fi, :] = S_norm[fi, :] * total_energy[fi] / ddir

    # Step 11: Apply smoothing if requested
    if estimation_params.smooth:
        if verbose >= 2:
            print("Smoothing spectrum...")
        S = _smooth_spectrum(S)

    # Step 12: Create output spectral matrix
    spectrum = SpectralMatrix(
        freqs=freqs,
        dirs=dirs,
        S=S,
        xaxisdir=90.0,  # East
        funit="hz",
        dunit="cart",
    )

    # Step 13: Calculate statistics
    if verbose >= 2:
        print("Calculating statistics...")
    info = _compute_spectral_info(spectrum)

    if verbose >= 1:
        print(f"\nResults:")
        print(f"  Significant wave height: {info.hsig:.2f} m")
        print(f"  Peak period: {info.tp:.2f} s")
        print(f"  Peak direction: {info.dp:.1f} deg")
        print(f"  Mean direction: {info.dm:.1f} deg")
        print(f"  Directional spread: {info.spread:.1f} deg")

    return spectrum, info


def dirspec_xarray(
    ds: xr.Dataset,
    depth: float,
    fs: float,
    estimation_params: EstimationParameters | None = None,
    freqs: NDArray[np.floating] | None = None,
    dirs: NDArray[np.floating] | None = None,
    verbose: int = 1,
) -> xr.Dataset:
    """Estimate directional spectrum from xarray Dataset.

    Convenience wrapper around `dirspec` that accepts and returns xarray objects.

    Args:
        ds: Dataset with sensor data variables. Each variable should have
            'sensor_type', 'x', 'y', 'z' attributes.
        depth: Mean water depth in meters.
        fs: Sampling frequency in Hz.
        estimation_params: Analysis parameters.
        freqs: Output frequency grid in Hz.
        dirs: Output direction grid in degrees.
        verbose: Verbosity level.

    Returns:
        xarray Dataset compatible with wavespectra package.
    """
    # Convert to InstrumentData
    instrument_data = InstrumentData.from_xarray(ds, depth, fs)

    # Run analysis
    spectrum, info = dirspec(
        instrument_data,
        estimation_params=estimation_params,
        freqs=freqs,
        dirs=dirs,
        verbose=verbose,
    )

    # Convert to xarray
    output = spectrum.to_xarray()

    # Add statistics as attributes
    output.attrs["hsig"] = info.hsig
    output.attrs["tp"] = info.tp
    output.attrs["fp"] = info.fp
    output.attrs["dp"] = info.dp
    output.attrs["dm"] = info.dm
    output.attrs["spread"] = info.spread

    return output


def _validate_inputs(
    instrument_data: InstrumentData,
    estimation_params: EstimationParameters,
) -> None:
    """Validate input data and parameters."""
    if instrument_data.n_sensors < 2:
        raise ValueError("At least 2 sensors required for directional analysis")

    if instrument_data.n_samples < 64:
        raise ValueError("At least 64 samples required")

    if estimation_params.nfft is not None:
        if estimation_params.nfft > instrument_data.n_samples:
            raise ValueError(
                f"nfft ({estimation_params.nfft}) cannot exceed "
                f"number of samples ({instrument_data.n_samples})"
            )


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
        kernel = np.array(
            [[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]
        )
        kernel = kernel / np.sum(kernel)

    # Apply convolution with wrap mode for circular direction dimension
    S_smooth = ndimage.convolve(S, kernel, mode="wrap")

    return S_smooth


def _compute_spectral_info(spectrum: SpectralMatrix) -> SpectralInfo:
    """Compute spectral statistics."""
    return SpectralInfo(
        hsig=hsig(spectrum.S, spectrum.freqs, spectrum.dirs),
        tp=1.0 / peak_frequency(spectrum.S, spectrum.freqs),
        fp=peak_frequency(spectrum.S, spectrum.freqs),
        dp=peak_direction(spectrum.S, spectrum.freqs, spectrum.dirs),
        dm=mean_direction(spectrum.S, spectrum.dirs),
        spread=directional_spread(spectrum.S, spectrum.dirs),
    )
