# Data Structures

DIWASP uses three main data structures to manage input and output data:

1. **InstrumentData** - Contains the sensor layout, types, and measured data
2. **SpectralMatrix** - Contains the output directional spectrum
3. **EstimationParameters** - Contains configuration for the estimation method

## InstrumentData

The `InstrumentData` class contains all information about the instrument sensors and the measured data.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `data` | ndarray | Measured wave data matrix [n_samples x n_sensors] |
| `layout` | ndarray | Sensor positions [3 x n_sensors] as (x, y, z) where z is elevation from seabed (m) |
| `datatypes` | list[SensorType] | List of sensor types, one per column of data |
| `depth` | float | Mean overall depth of measurement area (m) |
| `fs` | float | Sampling frequency (Hz) - must be same for all sensors |

### Sensor Types

The following sensor types are supported:

| Type | Description |
|------|-------------|
| `SensorType.ELEV` | Surface elevation |
| `SensorType.PRES` | Pressure |
| `SensorType.VELX` | X component velocity |
| `SensorType.VELY` | Y component velocity |
| `SensorType.VELZ` | Z component velocity |
| `SensorType.VELS` | Vertical velocity of surface |
| `SensorType.ACCS` | Vertical acceleration of surface |
| `SensorType.SLPX` | X component surface slope |
| `SensorType.SLPY` | Y component surface slope |
| `SensorType.ACCX` | X component acceleration |
| `SensorType.ACCY` | Y component acceleration |
| `SensorType.ACCZ` | Z component acceleration |
| `SensorType.DSPX` | X displacement |
| `SensorType.DSPY` | Y displacement |

### Example: Pressure Gauge Array

For three pressure gauges spread in a triangle on the sea floor:

```python
from diwasp import InstrumentData, SensorType
import numpy as np

# Sensor data [n_samples x 3]
data = np.loadtxt('pressure_data.csv')

# Layout: x, y, z coordinates for each sensor
# Positions: [0,0], [5,5], [-5,5] with all sensors on seabed
layout = np.array([
    [0.0, 5.0, -5.0],   # x coordinates
    [0.0, 5.0,  5.0],   # y coordinates
    [0.0, 0.0,  0.0]    # z coordinates (seabed)
])

datatypes = [SensorType.PRES, SensorType.PRES, SensorType.PRES]

instrument = InstrumentData(
    data=data,
    layout=layout,
    datatypes=datatypes,
    depth=10.0,
    fs=2.0
)
```

### Example: PUV (Pressure + Velocity) Sensor

For a directional current meter and pressure sensor mounted 0.5m above the seabed:

```python
# All sensors at same location
layout = np.array([
    [0.0, 0.0, 0.0],   # x coordinates
    [0.0, 0.0, 0.0],   # y coordinates
    [0.5, 0.5, 0.5]    # z coordinates (0.5m above seabed)
])

datatypes = [SensorType.VELX, SensorType.VELY, SensorType.PRES]

instrument = InstrumentData(
    data=data,
    layout=layout,
    datatypes=datatypes,
    depth=10.0,
    fs=2.0
)
```

### Creating from xarray Dataset

```python
import xarray as xr

# Create dataset with sensor data
ds = xr.Dataset({
    'pressure': (['time'], pressure_data, {'sensor_type': 'pres', 'x': 0, 'y': 0, 'z': 0.5}),
    'vel_x': (['time'], velx_data, {'sensor_type': 'velx', 'x': 0, 'y': 0, 'z': 0.5}),
    'vel_y': (['time'], vely_data, {'sensor_type': 'vely', 'x': 0, 'y': 0, 'z': 0.5}),
})

instrument = InstrumentData.from_xarray(ds, depth=10.0, fs=2.0)
```

## SpectralMatrix

The `SpectralMatrix` class contains the directional wave spectrum.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `freqs` | ndarray | Frequency bin centers (Hz) |
| `dirs` | ndarray | Direction bin centers (degrees) |
| `S` | ndarray | Spectral density matrix [n_freqs x n_dirs] in m^2/(Hz*degree) |
| `xaxisdir` | float | Compass direction of x-axis (default 90 = East) |
| `funit` | str | Frequency units: 'hz' or 'rad/s' |
| `dunit` | str | Direction units: 'cart' (Cartesian) or 'naut' (nautical) |

### Direction Convention

Directions are measured anticlockwise from the positive x-axis in Cartesian convention (`dunit='cart'`). The `xaxisdir` field defines the compass direction of the x-axis.

```
        y
    N   ^
    ^   |  Wave component
    |   |  traveling at +30 deg
    |   +--------> x

With xaxisdir=90, the x-axis points East.
```

For nautical convention (`dunit='naut'`), directions are the compass bearing that waves are coming FROM.

### Converting to xarray

The spectrum can be converted to an xarray Dataset compatible with wavespectra:

```python
ds = spectrum.to_xarray()

# Access the spectral density
efth = ds['efth']  # Energy density variable
```

### Spectral Density Units

The spectral density `S` is in units of m^2/(Hz*degree). To convert to component wave amplitudes:

```
a_ij = sqrt(2 * S_ij * df * ddir)
```

where `df` is the frequency bin width and `ddir` is the direction bin width.

## EstimationParameters

The `EstimationParameters` class configures the spectrum estimation.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | EstimationMethod | IMLM | Estimation algorithm |
| `nfft` | int or None | None | FFT length (auto-calculated if None) |
| `dres` | int | 180 | Directional resolution (bins for full circle) |
| `iter` | int | 100 | Number of iterations for iterative methods |
| `smooth` | bool | True | Apply spectral smoothing |

### Example

```python
from diwasp import EstimationParameters, EstimationMethod

# Use BDM method with higher directional resolution
params = EstimationParameters(
    method=EstimationMethod.BDM,
    dres=360,
    iter=150,
    smooth=True
)
```

## SpectralInfo

The `SpectralInfo` class contains statistics computed from a directional spectrum.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `hsig` | float | Significant wave height (m) |
| `tp` | float | Peak period (s) |
| `fp` | float | Peak frequency (Hz) |
| `dp` | float | Peak direction (deg) |
| `dm` | float | Mean direction (deg) |
| `spread` | float | Directional spread (deg) |
