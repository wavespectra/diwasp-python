# High-Level Wrapper Function

The `diwasp` function is the main entry point for analyzing wave data. It provides a simple interface for processing continuous sensor data over multiple time windows and returns wavespectra-compatible output.

## Overview

```python
from diwasp import diwasp

result = diwasp(
    data,                    # DataFrame or Dataset
    sensor_mapping,          # Map variable names to sensor types
    window_length,           # Analysis window in seconds
    window_overlap,          # Overlap in seconds
    depth,                   # Water depth in meters
    method='imlm',           # Estimation method
)
```

## Input Data Formats

### pandas DataFrame

For DataFrame input:

- **Index**: Must be a `DatetimeIndex`
- **Columns**: Sensor variables (pressure, velocity, etc.)

```python
import pandas as pd
from diwasp import diwasp

# Load data with datetime index
df = pd.read_csv('wave_data.csv', index_col='time', parse_dates=True)

# Or create from scratch
time = pd.date_range('2024-01-01', periods=7200, freq='500ms')
df = pd.DataFrame({
    'pressure': pressure_data,
    'u_velocity': u_data,
    'v_velocity': v_data,
}, index=time)

result = diwasp(
    df,
    sensor_mapping={'pressure': 'pres', 'u_velocity': 'velx', 'v_velocity': 'vely'},
    window_length=1800,
    window_overlap=900,
    depth=20.0,
)
```

### xarray Dataset

For Dataset input:

- **Time dimension**: Must contain datetime values
- **Data variables**: Sensor measurements

```python
import xarray as xr
from diwasp import diwasp

ds = xr.Dataset({
    'pres': (['time'], pressure_data),
    'velx': (['time'], u_data),
    'vely': (['time'], v_data),
}, coords={
    'time': time_values,  # datetime64 array
})

result = diwasp(
    ds,
    sensor_mapping={'pres': 'pres', 'velx': 'velx', 'vely': 'vely'},
    window_length=1800,
    window_overlap=900,
    depth=20.0,
)
```

## Parameters

### Required Parameters

| Parameter        | Type                 | Description                         |
| ---------------- | -------------------- | ----------------------------------- |
| `data`           | DataFrame or Dataset | Input sensor data                   |
| `sensor_mapping` | dict                 | Maps variable names to sensor types |
| `window_length`  | float                | Analysis window length in seconds   |
| `window_overlap` | float                | Overlap between windows in seconds  |
| `depth`          | float                | Water depth in meters               |

### Sensor Mapping

The `sensor_mapping` parameter maps your variable/column names to DIWASP sensor types:

```python
sensor_mapping = {
    'my_pressure_col': 'pres',    # Pressure sensor
    'east_velocity': 'velx',       # East velocity component
    'north_velocity': 'vely',      # North velocity component
}
```

Available sensor types:

| Type     | Description                   |
| -------- | ----------------------------- |
| `'elev'` | Surface elevation             |
| `'pres'` | Pressure                      |
| `'velx'` | X-component velocity          |
| `'vely'` | Y-component velocity          |
| `'velz'` | Z-component velocity          |
| `'vels'` | Surface vertical velocity     |
| `'accs'` | Surface vertical acceleration |
| `'slpx'` | X-component surface slope     |
| `'slpy'` | Y-component surface slope     |
| `'accx'` | X-component acceleration      |
| `'accy'` | Y-component acceleration      |
| `'accz'` | Z-component acceleration      |
| `'dspx'` | X displacement                |
| `'dspy'` | Y displacement                |

### Optional Parameters

| Parameter  | Type          | Default  | Description                                                        |
| ---------- | ------------- | -------- | ------------------------------------------------------------------ |
| `method`   | str           | `'imlm'` | Estimation method: `'dftm'`, `'emlm'`, `'imlm'`, `'emep'`, `'bdm'` |
| `time_var` | str           | `'time'` | Name of time dimension (Dataset only)                              |
| `x_var`    | str           | `'x'`    | Name of x-coordinate variable                                      |
| `y_var`    | str           | `'y'`    | Name of y-coordinate variable                                      |
| `z_var`    | str           | `'z'`    | Name of z-coordinate variable                                      |
| `z`        | float or dict | None     | Sensor z-positions (height above seabed)                           |
| `x`        | float or dict | None     | Sensor x-positions                                                 |
| `y`        | float or dict | None     | Sensor y-positions                                                 |
| `fs`       | float         | None     | Sampling frequency (inferred if None)                              |
| `freqs`    | ndarray       | None     | Output frequency grid                                              |
| `dirs`     | ndarray       | None     | Output direction grid                                              |
| `dres`     | int           | 180      | Directional resolution                                             |
| `nfft`     | int           | None     | FFT length                                                         |
| `smooth`   | bool          | True     | Apply spectral smoothing                                           |
| `verbose`  | int           | 1        | Verbosity level (0, 1, 2)                                          |

## Sensor Positions

### Single Position for All Sensors

If all sensors are co-located:

```python
result = diwasp(
    df,
    sensor_mapping={'p': 'pres', 'u': 'velx', 'v': 'vely'},
    window_length=1800,
    window_overlap=900,
    depth=20.0,
    z=0.5,  # All sensors 0.5m above seabed
)
```

### Different Positions per Sensor

For sensors at different locations:

```python
result = diwasp(
    df,
    sensor_mapping={'p': 'pres', 'u': 'velx', 'v': 'vely'},
    window_length=1800,
    window_overlap=900,
    depth=20.0,
    x={'p': 0, 'u': 0, 'v': 0},
    y={'p': 0, 'u': 0, 'v': 0},
    z={'p': 0.5, 'u': 1.0, 'v': 1.0},
)
```

### Positions from Dataset Attributes

For xarray Datasets, positions can be stored as variable attributes:

```python
ds = xr.Dataset({
    'pres': (['time'], data, {'x': 0, 'y': 0, 'z': 0.5}),
    'velx': (['time'], data, {'x': 0, 'y': 0, 'z': 1.0}),
    'vely': (['time'], data, {'x': 0, 'y': 0, 'z': 1.0}),
})

# Positions are read automatically from attributes
result = diwasp(ds, sensor_mapping={...}, ...)
```

## Output Format

The function returns a wavespectra-compatible xarray Dataset:

### Dimensions

- `time`: Center time of each analysis window
- `freq`: Frequency bins (Hz)
- `dir`: Direction bins (degrees)

### Variables

| Variable | Dimensions        | Units        | Description             |
| -------- | ----------------- | ------------ | ----------------------- |
| `efth`   | (time, freq, dir) | m²/Hz/degree | Spectral energy density |
| `hsig`   | (time)            | m            | Significant wave height |
| `tp`     | (time)            | s            | Peak period             |
| `fp`     | (time)            | Hz           | Peak frequency          |
| `dp`     | (time)            | degree       | Peak direction          |
| `dm`     | (time)            | degree       | Mean direction          |
| `spread` | (time)            | degree       | Directional spread      |

### Example Output Usage

```python
result = diwasp(df, ...)

# Access spectral data
spectrum = result['efth']  # 3D array (time, freq, dir)

# Access statistics time series
hsig = result['hsig'].values  # 1D array
tp = result['tp'].values

# Plot Hsig time series
import matplotlib.pyplot as plt
result['hsig'].plot()
plt.ylabel('Hsig (m)')
plt.show()

# Save to NetCDF
result.to_netcdf('wave_spectra.nc')
```

## Window Configuration

### Window Length

The `window_length` parameter sets the duration of each analysis window in seconds. Longer windows provide better frequency resolution but fewer output time points.

```
Frequency resolution ≈ 1 / window_length
```

Typical values:

- **1800 s (30 min)**: Standard for ocean waves
- **600-1200 s**: Higher time resolution
- **3600 s (1 hour)**: Better frequency resolution

### Window Overlap

The `window_overlap` parameter controls how much consecutive windows overlap:

- **0**: No overlap, independent windows
- **window_length/2**: 50% overlap (common choice)
- **window_length \* 0.75**: 75% overlap for smoother time series

```python
# 30-minute windows with 50% overlap
result = diwasp(
    df,
    sensor_mapping=mapping,
    window_length=1800,
    window_overlap=900,
    depth=20.0,
)
```

### Number of Windows

The number of output windows is calculated as:

```
n_windows = 1 + (data_length - window_length) / (window_length - window_overlap)
```

## Sampling Frequency

The sampling frequency is automatically inferred from the time index. The function will automatically resample data if:

1. The time index has non-uniform spacing (irregular sampling)
2. An explicit `fs` parameter is provided that differs from the inferred frequency

### Automatic Resampling

If resampling is needed, the data is interpolated linearly to a uniform grid:

```python
# Data with irregular sampling will be automatically resampled
result = diwasp(
    df_irregular,  # Non-uniform time spacing
    sensor_mapping=mapping,
    window_length=1800,
    window_overlap=900,
    depth=20.0,
)
```

### Override Sampling Frequency

To force resampling to a specific frequency:

```python
result = diwasp(
    df,
    sensor_mapping=mapping,
    window_length=1800,
    window_overlap=900,
    depth=20.0,
    fs=2.0,  # Force 2 Hz sampling frequency (triggers resampling if different)
)
```

## Estimation Methods

See [Estimation Methods](estimation_methods.md) for detailed information on each method.

Quick guide:

- `'dftm'`: Fastest, lowest accuracy
- `'emlm'`: Fast, good for narrow spectra
- `'imlm'`: Good balance (default)
- `'emep'`: Robust, handles noise well
- `'bdm'`: Best accuracy, slowest

```python
# Use EMEP for noisy data
result = diwasp(
    df,
    sensor_mapping=mapping,
    window_length=1800,
    window_overlap=900,
    depth=20.0,
    method='emep',
)
```

## Complete Examples

### PUV Sensor Analysis

```python
import pandas as pd
from diwasp import diwasp

# Load PUV data
df = pd.read_csv('puv_data.csv', index_col='time', parse_dates=True)

# Analyze with IMLM method
result = diwasp(
    df,
    sensor_mapping={
        'pressure': 'pres',
        'u_velocity': 'velx',
        'v_velocity': 'vely',
    },
    window_length=1800,      # 30 minutes
    window_overlap=900,       # 15 minutes overlap
    depth=15.0,              # 15m water depth
    z=0.5,                   # Sensors 0.5m above seabed
    method='imlm',
    dres=180,                # 2-degree directional resolution
    verbose=1,
)

# Print summary statistics
print(f"Number of spectra: {len(result.time)}")
print(f"Hsig range: {result.hsig.min().values:.2f} - {result.hsig.max().values:.2f} m")
print(f"Tp range: {result.tp.min().values:.1f} - {result.tp.max().values:.1f} s")

# Save results
result.to_netcdf('wave_spectra.nc')
```

### Pressure Gauge Array

```python
import pandas as pd
from diwasp import diwasp

# Load data from 3 pressure gauges
df = pd.read_csv('pressure_array.csv', index_col='time', parse_dates=True)

# Define sensor positions (triangular array)
result = diwasp(
    df,
    sensor_mapping={
        'p1': 'pres',
        'p2': 'pres',
        'p3': 'pres',
    },
    window_length=1800,
    window_overlap=900,
    depth=10.0,
    x={'p1': 0, 'p2': 5, 'p3': -5},
    y={'p1': 0, 'p2': 5, 'p3': 5},
    z=0.0,  # All on seabed
    method='emep',
)
```

### Integration with wavespectra

```python
from diwasp import diwasp
import wavespectra

# Run DIWASP analysis
result = diwasp(df, sensor_mapping=mapping, ...)

# Use wavespectra methods
hs = result.spec.hs()  # Significant wave height
tp = result.spec.tp()  # Peak period

# Plot using wavespectra
result.spec.plot()
```

## Input Validation

The function validates inputs and will raise errors for:

### Empty Sensor Mapping

```python
# This will raise ValueError
result = diwasp(df, sensor_mapping={}, ...)
```

**Error**: `ValueError: sensor_mapping cannot be empty`

### Invalid Depth

```python
# This will raise ValueError
result = diwasp(df, sensor_mapping=mapping, depth=-10.0, ...)
```

**Error**: `ValueError: depth must be positive`

### Invalid Window Overlap

```python
# This will raise ValueError
result = diwasp(
    df,
    sensor_mapping=mapping,
    window_length=300,
    window_overlap=300,  # Must be less than window_length
    depth=20.0,
)
```

**Error**: `ValueError: window_overlap must be less than window_length`

## Troubleshooting

### "DatetimeIndex" Error

```
ValueError: DataFrame index must be DatetimeIndex
```

**Solution**: Ensure your DataFrame has a datetime index:

```python
df.index = pd.to_datetime(df.index)
# or
df = pd.read_csv('data.csv', index_col='time', parse_dates=True)
```

### "Window exceeds data length" Error

**Solution**: Reduce window length or get more data:

```python
# Check data length
print(f"Data duration: {(df.index[-1] - df.index[0]).total_seconds()} seconds")

# Use shorter windows
result = diwasp(df, ..., window_length=600, ...)  # 10 minutes
```

### Poor Results

1. Check sensor positions are correct
2. Try a more robust method (`'emep'` or `'bdm'`)
3. Verify data quality (no gaps, correct units)
4. Ensure sufficient sensors for directional resolution

### Slow Performance

1. Use faster method (`'imlm'` or `'emlm'`)
2. Reduce directional resolution (`dres=90`)
3. Limit frequency range with `freqs` parameter
4. Reduce `nfft` for faster FFT computation
