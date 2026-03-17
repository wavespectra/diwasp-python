"""High-level wrapper function for directional wave spectrum analysis.

This module provides the main `diwasp` function that accepts xarray Datasets
or pandas DataFrames and returns wavespectra-compatible output with analysis
results over multiple time windows.
"""

from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from .core import dirspec
from .methods import BDM, DFTM, EMLM, EMEP, IMLM
from .types import (
    EstimationMethod,
    EstimationParameters,
    InstrumentData,
    SensorType,
    SpectralInfo,
)


# Default sensor type mapping
DEFAULT_SENSOR_MAP = {
    "pres": SensorType.PRES,
    "velx": SensorType.VELX,
    "vely": SensorType.VELY,
    "velz": SensorType.VELZ,
}

# Method name to enum mapping
METHOD_MAP = {
    "dftm": EstimationMethod.DFTM,
    "emlm": EstimationMethod.EMLM,
    "imlm": EstimationMethod.IMLM,
    "emep": EstimationMethod.EMEP,
    "bdm": EstimationMethod.BDM,
}


def diwasp(
    data: xr.Dataset | pd.DataFrame,
    sensor_mapping: dict[str, str],
    depth: float,
    window_length: float = 1200,
    window_overlap: float = 0,
    method: str = "emlm",
    time_var: str = "time",
    x_var: str = "x",
    y_var: str = "y",
    z_var: str = "z",
    z: float | dict[str, float] | None = None,
    x: float | dict[str, float] | None = None,
    y: float | dict[str, float] | None = None,
    fs: float | None = None,
    freqs: NDArray[np.floating] | None = None,
    dirs: NDArray[np.floating] | None = None,
    dres: int = 180,
    nfft: int | None = None,
    smooth: bool = True,
    window_timestamp: Literal["start", "center", "end"] = "center",
    verbose: int = 1,
) -> xr.Dataset:
    """Estimate directional wave spectra from sensor data over multiple windows.

    This is the main entry point for the DIWASP package. It accepts either an
    xarray Dataset or pandas DataFrame containing sensor measurements and returns
    a wavespectra-compatible xarray Dataset with spectral estimates for each
    analysis window.

    Parameters
    ----------
    data : xr.Dataset or pd.DataFrame
        Input sensor data. For DataFrame, index must be a DatetimeIndex and
        columns are sensor variables. For Dataset, must contain a time dimension
        with datetime values.
    sensor_mapping : dict[str, str]
        Mapping from variable/column names to sensor types.
        Keys are the names in the data, values are sensor type codes:
        'elev', 'pres', 'velx', 'vely', 'velz', 'vels', 'accs',
        'slpx', 'slpy', 'accx', 'accy', 'accz', 'dspx', 'dspy'.
        Example: {'pressure': 'pres', 'u': 'velx', 'v': 'vely'}
    window_length : float, default 1200
        Analysis window length in seconds.
    window_overlap : float, default 0
        Overlap between consecutive windows in seconds.
    depth : float
        Water depth in meters.
    method : str, default 'emlm'
        Estimation method: 'dftm', 'emlm', 'imlm', 'emep', or 'bdm'.
    time_var : str, default 'time'
        Name of the time variable/dimension in xarray Dataset.
    x_var : str, default 'x'
        Name of the x-coordinate variable in Dataset (if present).
    y_var : str, default 'y'
        Name of the y-coordinate variable in Dataset (if present).
    z_var : str, default 'z'
        Name of the z-coordinate variable in Dataset (if present).
    z : float or dict, optional
        Sensor z-positions (height above seabed in meters).
        If float, applies to all sensors. If dict, maps variable names to z values.
        If None, attempts to read from Dataset coordinates or defaults to depth.
    x : float or dict, optional
        Sensor x-positions in meters. If float, all sensors at same x.
        If dict, maps variable names to x values. Default 0.
    y : float or dict, optional
        Sensor y-positions in meters. If float, all sensors at same y.
        If dict, maps variable names to y values. Default 0.
    fs : float, optional
        Sampling frequency in Hz. If None, inferred from time variable.
    freqs : ndarray, optional
        Output frequency grid in Hz. If None, auto-determined.
    dirs : ndarray, optional
        Output direction grid in degrees. If None, uses 0-360 with dres bins.
    dres : int, default 180
        Directional resolution (number of bins for 360 degrees).
    nfft : int, optional
        FFT length. If None, auto-determined from window length.
    smooth : bool, default True
        Apply spectral smoothing.
    window_timestamp: Literal["start", "center", "end"], optional
        Timestamp to use for window. Default is 'start' of window.
    verbose : int, default 1
        Verbosity level (0=silent, 1=normal, 2=detailed).

    Returns
    -------
    xr.Dataset
        wavespectra-compatible Dataset with dimensions (time, freq, dir) containing:
        - efth: Spectral energy density [m^2/Hz/degree]

    Raises
    ------
    ValueError
        If input data format is invalid or required parameters are missing.
    TypeError
        If data is not a DataFrame or Dataset.

    Examples
    --------
    From pandas DataFrame:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from diwasp import diwasp
    >>>
    >>> # Create sample data
    >>> time = pd.date_range('2024-01-01', periods=7200, freq='0.5s')
    >>> df = pd.DataFrame({
    ...     'pressure': np.random.randn(7200),
    ...     'u_vel': np.random.randn(7200),
    ...     'v_vel': np.random.randn(7200),
    ... }, index=time)
    >>>
    >>> # Run analysis
    >>> result = diwasp(
    ...     df,
    ...     sensor_mapping={'pressure': 'pres', 'u_vel': 'velx', 'v_vel': 'vely'},
    ...     window_length=1800,  # 30 minutes
    ...     window_overlap=900,  # 15 minutes
    ...     depth=20.0,
    ...     z=0.5,  # sensors 0.5m above seabed
    ... )

    From xarray Dataset:

    >>> import xarray as xr
    >>> ds = xr.Dataset({
    ...     'pres': (['time'], pressure_data),
    ...     'velx': (['time'], velx_data),
    ...     'vely': (['time'], vely_data),
    ... }, coords={'time': time_values})
    >>>
    >>> result = diwasp(
    ...     ds,
    ...     sensor_mapping={'pres': 'pres', 'velx': 'velx', 'vely': 'vely'},
    ...     window_length=1800,
    ...     window_overlap=900,
    ...     depth=20.0,
    ... )
    """
    # Validate inputs
    if not sensor_mapping:
        raise ValueError("sensor_mapping cannot be empty")

    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")

    if window_overlap >= window_length:
        raise ValueError(
            f"window_overlap ({window_overlap}s) must be less than "
            f"window_length ({window_length}s)"
        )

    # Validate method
    method_lower = method.lower()
    if method_lower not in METHOD_MAP:
        raise ValueError(f"Unknown method '{method}'. Must be one of: {list(METHOD_MAP.keys())}")

    # Convert input to standardized format
    if isinstance(data, pd.DataFrame):
        sensor_data, time_index, inferred_fs, layout = _process_dataframe(
            data, sensor_mapping, z, x, y, depth, fs
        )
    elif isinstance(data, xr.Dataset):
        sensor_data, time_index, inferred_fs, layout = _process_dataset(
            data, sensor_mapping, time_var, x_var, y_var, z_var, z, x, y, depth, fs
        )
    else:
        raise TypeError(f"data must be pandas DataFrame or xarray Dataset, got {type(data)}")

    # Use provided or inferred sampling frequency
    sampling_freq = fs if fs is not None else inferred_fs

    if verbose >= 1:
        print(f"DIWASP Analysis")
        print(f"  Method: {method_lower.upper()}")
        print(f"  Sampling frequency: {sampling_freq:.2f} Hz")
        print(f"  Window length: {window_length:.1f} s")
        print(f"  Window overlap: {window_overlap:.1f} s")
        print(f"  Depth: {depth:.1f} m")
        print(f"  Sensors: {len(sensor_mapping)}")

    # Calculate window parameters
    window_samples = int(window_length * sampling_freq)
    overlap_samples = int(window_overlap * sampling_freq)
    step_samples = window_samples - overlap_samples

    if window_samples > len(sensor_data):
        raise ValueError(
            f"Window length ({window_length}s = {window_samples} samples) "
            f"exceeds data length ({len(sensor_data)} samples)"
        )

    # Calculate number of windows
    n_windows = 1 + (len(sensor_data) - window_samples) // step_samples

    if verbose >= 1:
        print(f"  Number of windows: {n_windows}")

    # Set up estimation parameters
    est_params = EstimationParameters(
        method=METHOD_MAP[method_lower],
        nfft=nfft,
        dres=dres,
        smooth=smooth,
    )

    # Determine output grids
    if dirs is None:
        dirs = np.linspace(0, 360, dres, endpoint=False)

    # Process each window
    spectra_list = []
    window_times = []

    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + window_samples

        # Extract window data
        window_data = sensor_data[start_idx:end_idx, :]

        if window_timestamp == "center":
            # Get window center time
            window_idx = start_idx + window_samples // 2
        elif window_timestamp == "end":
            window_idx = end_idx - 1
        else:
            window_idx = start_idx
        window_time = time_index[window_idx]
        window_times.append(window_time)

        if verbose >= 2:
            print(f"\nProcessing window {i + 1}/{n_windows}: {window_time}")

        # Create InstrumentData for this window
        sensor_types = [SensorType(sensor_mapping[name]) for name in sensor_mapping]

        instrument = InstrumentData(
            data=window_data,
            layout=layout,
            datatypes=sensor_types,
            depth=depth,
            fs=sampling_freq,
        )

        # Estimate spectrum
        spectrum, _info = dirspec(
            instrument,
            estimation_params=est_params,
            freqs=freqs,
            dirs=dirs,
            verbose=max(0, verbose - 1),
        )

        # Store results
        spectra_list.append(spectrum.S)

        # Update freqs from first result if not specified
        if freqs is None:
            freqs = spectrum.freqs

    # Build output Dataset
    output = _build_output_dataset(
        spectra_list, window_times, freqs, dirs, spectrum.xaxisdir, spectrum.funit, spectrum.dunit
    )

    if verbose >= 1:
        print(f"\nAnalysis complete. Output shape: {output['efth'].shape}")

    return output


def _process_dataframe(
    df: pd.DataFrame,
    sensor_mapping: dict[str, str],
    z: float | dict[str, float] | None,
    x: float | dict[str, float] | None,
    y: float | dict[str, float] | None,
    depth: float,
    fs: float | None,
) -> tuple[NDArray, pd.DatetimeIndex, float, NDArray]:
    """Process pandas DataFrame input.

    Returns:
        Tuple of (data array, time index, sampling frequency, layout array)
    """
    # Validate index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame index must be DatetimeIndex, got {type(df.index)}")

    # Validate all sensor columns exist
    missing = [name for name in sensor_mapping if name not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    # Extract data in sensor order
    sensor_names = list(sensor_mapping.keys())

    # Infer sampling frequency from time index
    time_diff = df.index.to_series().diff().median()
    inferred_fs = 1.0 / time_diff.total_seconds()

    # Determine target sampling frequency
    target_fs = fs if fs is not None else inferred_fs

    # Check if resampling is needed
    time_diff_std = df.index.to_series().diff().std().total_seconds()
    needs_resampling = time_diff_std > 1e-6 or (fs is not None and abs(inferred_fs - fs) > 0.01)

    if needs_resampling:
        # Resample to uniform frequency
        period_ms = int(1000.0 / target_fs)
        new_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=f"{period_ms}ms")
        df_resampled = df[sensor_names].reindex(new_index).interpolate(method="linear")
        data = df_resampled.values
        time_index = new_index
        inferred_fs = target_fs
    else:
        data = df[sensor_names].values
        time_index = df.index

    # Build layout array [3 x n_sensors]
    n_sensors = len(sensor_names)
    layout = np.zeros((3, n_sensors))

    for i, name in enumerate(sensor_names):
        # X position
        if x is None:
            layout[0, i] = 0.0
        elif isinstance(x, dict):
            layout[0, i] = x.get(name, 0.0)
        else:
            layout[0, i] = x

        # Y position
        if y is None:
            layout[1, i] = 0.0
        elif isinstance(y, dict):
            layout[1, i] = y.get(name, 0.0)
        else:
            layout[1, i] = y

        # Z position
        if z is None:
            layout[2, i] = depth  # Default to surface
        elif isinstance(z, dict):
            layout[2, i] = z.get(name, depth)
        else:
            layout[2, i] = z

    return data, time_index, inferred_fs, layout


def _process_dataset(
    ds: xr.Dataset,
    sensor_mapping: dict[str, str],
    time_var: str,
    x_var: str,
    y_var: str,
    z_var: str,
    z: float | dict[str, float] | None,
    x: float | dict[str, float] | None,
    y: float | dict[str, float] | None,
    depth: float,
    fs: float | None,
) -> tuple[NDArray, pd.DatetimeIndex, float, NDArray]:
    """Process xarray Dataset input.

    Returns:
        Tuple of (data array, time index, sampling frequency, layout array)
    """
    # Validate time dimension exists
    if time_var not in ds.dims and time_var not in ds.coords:
        raise ValueError(f"Time variable '{time_var}' not found in Dataset")

    # Get time coordinate
    time_coord = ds[time_var]

    # Validate time is datetime-like
    if not np.issubdtype(time_coord.dtype, np.datetime64):
        raise ValueError(
            f"Time variable '{time_var}' must be datetime type, got {time_coord.dtype}"
        )

    # Validate all sensor variables exist
    missing = [name for name in sensor_mapping if name not in ds.data_vars]
    if missing:
        raise ValueError(f"Variables not found in Dataset: {missing}")

    # Extract data in sensor order
    sensor_names = list(sensor_mapping.keys())

    # Convert time to DatetimeIndex
    time_index = pd.DatetimeIndex(time_coord.values)

    # Infer sampling frequency
    time_diff = pd.Series(time_index).diff().median()
    inferred_fs = 1.0 / time_diff.total_seconds()

    # Determine target sampling frequency
    target_fs = fs if fs is not None else inferred_fs

    # Check if resampling is needed
    time_diff_std = pd.Series(time_index).diff().std().total_seconds()
    needs_resampling = time_diff_std > 1e-6 or (fs is not None and abs(inferred_fs - fs) > 0.01)

    if needs_resampling:
        # Resample to uniform frequency
        period_ms = int(1000.0 / target_fs)
        new_index = pd.date_range(start=time_index[0], end=time_index[-1], freq=f"{period_ms}ms")
        # Convert to DataFrame for resampling
        df_temp = pd.DataFrame({name: ds[name].values for name in sensor_names}, index=time_index)
        df_resampled = df_temp.reindex(new_index).interpolate(method="linear")
        data = df_resampled.values
        time_index = new_index
        inferred_fs = target_fs
    else:
        data_arrays = [ds[name].values for name in sensor_names]
        data = np.column_stack(data_arrays)

    # Build layout array [3 x n_sensors]
    n_sensors = len(sensor_names)
    layout = np.zeros((3, n_sensors))

    for i, name in enumerate(sensor_names):
        da = ds[name]

        # Try to get position from Dataset coordinates or attributes
        # X position
        if x is not None:
            if isinstance(x, dict):
                layout[0, i] = x.get(name, 0.0)
            else:
                layout[0, i] = x
        elif x_var in da.coords:
            layout[0, i] = float(da[x_var].values)
        elif x_var in da.attrs:
            layout[0, i] = da.attrs[x_var]
        else:
            layout[0, i] = 0.0

        # Y position
        if y is not None:
            if isinstance(y, dict):
                layout[1, i] = y.get(name, 0.0)
            else:
                layout[1, i] = y
        elif y_var in da.coords:
            layout[1, i] = float(da[y_var].values)
        elif y_var in da.attrs:
            layout[1, i] = da.attrs[y_var]
        else:
            layout[1, i] = 0.0

        # Z position
        if z is not None:
            if isinstance(z, dict):
                layout[2, i] = z.get(name, depth)
            else:
                layout[2, i] = z
        elif z_var in da.coords:
            layout[2, i] = float(da[z_var].values)
        elif z_var in da.attrs:
            layout[2, i] = da.attrs[z_var]
        else:
            layout[2, i] = depth  # Default to surface

    return data, time_index, inferred_fs, layout


def _build_output_dataset(
    spectra_list: list[NDArray],
    window_times: list,
    freqs: NDArray,
    dirs: NDArray,
    xaxisdir: float,
    funit: str = "hz",
    dunit: str = "cart",
) -> xr.Dataset:
    """Build wavespectra-compatible output Dataset.

    Args:
        spectra_list: List of spectral matrices.
        window_times: List of window center times.
        freqs: Frequency array.
        dirs: Direction array.
        xaxisdir: Reference x-axis direction.
        funit: Frequency units ('hz' or 'rad/s').
        dunit: Direction units ('cart' or 'naut').

    Returns:
        xarray Dataset with dimensions (time, freq, dir)
    """
    from .utils import (
        hsig,
        mean_direction,
        one_sided_directional_spread,
        peak_direction,
        peak_frequency,
    )

    # Stack spectra into 3D array [time, freq, dir]
    efth = np.stack(spectra_list, axis=0)
    n_times = len(spectra_list)

    # Convert times to numpy datetime64
    time_values = np.array(window_times, dtype="datetime64[ns]")

    # Compute spectral statistics for each time step
    hsig_arr = np.zeros(n_times)
    tp_arr = np.zeros(n_times)
    fp_arr = np.zeros(n_times)
    dp_arr = np.zeros(n_times)
    dm_arr = np.zeros(n_times)
    spread_arr = np.zeros(n_times)

    for i, S in enumerate(spectra_list):
        hsig_arr[i] = hsig(S, freqs, dirs)
        fp_arr[i] = peak_frequency(S, freqs, dirs)
        tp_arr[i] = 1.0 / fp_arr[i] if fp_arr[i] > 0 else np.nan
        dp_arr[i] = peak_direction(S, freqs, dirs)
        dm_arr[i] = mean_direction(S, freqs, dirs)
        spread_arr[i] = one_sided_directional_spread(S, freqs, dirs)

    # Create Dataset
    ds = xr.Dataset(
        {
            "efth": (["time", "freq", "dir"], efth),
            "hsig": (["time"], hsig_arr),
            "tp": (["time"], tp_arr),
            "fp": (["time"], fp_arr),
            "dp": (["time"], dp_arr),
            "dm": (["time"], dm_arr),
            "spread": (["time"], spread_arr),
        },
        coords={
            "time": time_values,
            "freq": freqs,
            "dir": dirs,
        },
        attrs={
            "xaxisdir": xaxisdir,
            "source": "diwasp",
            "conventions": "wavespectra",
        },
    )

    # Construct units string based on funit and dunit
    freq_unit = "Hz" if funit == "hz" else "rad/s"
    dir_unit = "degree" if dunit == "cart" else "degree"  # Both use degrees

    # Add variable attributes
    ds["efth"].attrs = {
        "units": f"m^2/{freq_unit}/{dir_unit}",
        "long_name": "Spectral energy density",
        "standard_name": "sea_surface_wave_variance_spectral_density",
    }
    ds["hsig"].attrs = {
        "units": "m",
        "long_name": "Significant wave height",
        "standard_name": "sea_surface_wave_significant_height",
    }
    ds["tp"].attrs = {
        "units": "s",
        "long_name": "Peak wave period",
        "standard_name": "sea_surface_wave_period_at_variance_spectral_density_maximum",
    }
    ds["fp"].attrs = {
        "units": "Hz",
        "long_name": "Peak wave frequency",
    }
    ds["dp"].attrs = {
        "units": "degree",
        "long_name": "Peak wave direction",
        "standard_name": "sea_surface_wave_from_direction_at_variance_spectral_density_maximum",
    }
    ds["dm"].attrs = {
        "units": "degree",
        "long_name": "Mean wave direction",
        "standard_name": "sea_surface_wave_from_direction",
    }
    ds["spread"].attrs = {
        "units": "degree",
        "long_name": "Directional spread",
    }

    return ds
