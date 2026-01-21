# DIWASP-Python

**DIrectional WAve SPectrum** analysis - Python port of the DIWASP Matlab toolbox

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

DIWASP is a comprehensive toolbox for estimating directional wave spectra from measurements of water surface elevation, pressure, velocity, or acceleration. This Python implementation provides a modern, user-friendly interface built on top of industry-standard scientific Python libraries.

## Features

- **Multiple estimation methods**: DFTM, EMLM, IMLM, EMEP, and BDM
- **Flexible input formats**: pandas DataFrame or xarray Dataset
- **Seamless integration** with [wavespectra](https://wavespectra.readthedocs.io/) for advanced wave analysis
- **Windowed analysis**: Process continuous time series with configurable window length and overlap
- **Multiple sensor types**: Pressure, velocity, acceleration, surface elevation, and more
- **Array configurations**: Support for single sensors or multi-sensor arrays
- **Modern output**: Returns wavespectra-compatible xarray Datasets with comprehensive metadata

## Quick Start

### Installation

```bash
pip install diwasp
```

### Basic Usage

```python
import pandas as pd
from diwasp import diwasp

# Load your wave data (pressure and velocity measurements)
df = pd.read_csv('wave_data.csv', index_col='time', parse_dates=True)

# Run directional wave analysis
result = diwasp(
    df,
    sensor_mapping={
        'pressure': 'pres',      # Map column names to sensor types
        'u_velocity': 'velx',
        'v_velocity': 'vely',
    },
    window_length=1800,          # 30-minute analysis windows
    window_overlap=900,           # 15-minute overlap
    depth=15.0,                  # Water depth in meters
    z=0.5,                       # Sensor height above seabed
    method='imlm',               # Iterative Maximum Likelihood Method
)

# Access wave statistics
print(f"Significant wave height: {result.hsig.values} m")
print(f"Peak period: {result.tp.values} s")
print(f"Peak direction: {result.dp.values} degrees")

# Save results
result.to_netcdf('wave_spectra.nc')
```

## Estimation Methods

| Method | Description                  | Speed   | Accuracy                |
| ------ | ---------------------------- | ------- | ----------------------- |
| `dftm` | Direct Fourier Transform     | Fastest | Lowest                  |
| `emlm` | Extended Maximum Likelihood  | Fast    | Good for narrow spectra |
| `imlm` | Iterative Maximum Likelihood | Medium  | Good balance (default)  |
| `emep` | Extended Maximum Entropy     | Medium  | Robust to noise         |
| `bdm`  | Bayesian Directional         | Slowest | Highest                 |

## Supported Sensor Types

- **Surface elevation** (`elev`)
- **Pressure** (`pres`)
- **Velocity components** (`velx`, `vely`, `velz`, `vels`)
- **Acceleration components** (`accx`, `accy`, `accz`, `accs`)
- **Surface slopes** (`slpx`, `slpy`)
- **Displacement** (`dspx`, `dspy`)

## Output Format

The function returns a wavespectra-compatible xarray Dataset containing:

**Dimensions:**

- `time`: Center time of each analysis window
- `freq`: Frequency bins (Hz)
- `dir`: Direction bins (degrees, nautical convention)

**Variables:**

- `efth`: Spectral energy density (m²/Hz/degree)
- `hsig`: Significant wave height (m)
- `tp`: Peak period (s)
- `fp`: Peak frequency (Hz)
- `dp`: Peak direction (degrees)
- `dm`: Mean direction (degrees)
- `spread`: Directional spread (degrees)

## Examples

### Pressure-Velocity-Velocity (PUV) Sensor

```python
from diwasp import diwasp
import pandas as pd

# Load data
df = pd.read_csv('puv_data.csv', index_col='time', parse_dates=True)

# Analyze
result = diwasp(
    df,
    sensor_mapping={'p': 'pres', 'u': 'velx', 'v': 'vely'},
    window_length=1800,
    window_overlap=900,
    depth=20.0,
    z=0.5,
    method='imlm',
)

# Plot time series
result['hsig'].plot()
```

### Pressure Gauge Array

```python
# Multiple pressure sensors in triangular configuration
result = diwasp(
    df,
    sensor_mapping={'p1': 'pres', 'p2': 'pres', 'p3': 'pres'},
    window_length=1800,
    window_overlap=900,
    depth=10.0,
    x={'p1': 0, 'p2': 5, 'p3': -5},
    y={'p1': 0, 'p2': 5, 'p3': 5},
    z=0.0,
    method='emep',
)
```

### Using xarray Dataset

```python
import xarray as xr

ds = xr.Dataset({
    'pres': (['time'], pressure_data),
    'velx': (['time'], u_data),
    'vely': (['time'], v_data),
}, coords={
    'time': time_values,
})

result = diwasp(
    ds,
    sensor_mapping={'pres': 'pres', 'velx': 'velx', 'vely': 'vely'},
    window_length=1800,
    window_overlap=900,
    depth=15.0,
)
```

## Documentation

Full documentation is available at [https://diwasp-python.readthedocs.io](https://diwasp-python.readthedocs.io)

- [Installation Guide](docs/installation.md)
- [High-Level Wrapper Function](docs/wrapper.md)
- [Estimation Methods](docs/estimation_methods.md)
- [Data Structures](docs/data_structures.md)
- [API Reference](docs/api_reference.md)

## Requirements

- Python >= 3.9
- numpy >= 1.20
- scipy >= 1.7
- xarray >= 0.19
- wavespectra >= 4.0
- pandas >= 1.3

## Development

### Install for Development

```bash
git clone https://github.com/yourusername/diwasp-python.git
cd diwasp-python
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black diwasp tests
ruff check diwasp tests
```

## Citation

If you use DIWASP in your research, please cite the original DIWASP toolbox:

> Johnson, D. (2002). DIWASP, a directional wave spectra toolbox for MATLAB: User Manual.
> Research Report WP-1601-DJ (V1.1), Centre for Water Research, University of Western Australia.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

- **Original DIWASP Matlab toolbox**: David Johnson, University of Western Australia
- **Python port**: Developed with support from the oceanographic community
- Built on [wavespectra](https://wavespectra.readthedocs.io/), [xarray](https://xarray.pydata.org/), and [pandas](https://pandas.pydata.org/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Related Projects

- [wavespectra](https://github.com/wavespectra/wavespectra) - Wave spectra analysis library
- [Original DIWASP (Matlab)](http://www.metocean.co.nz/diwasp/) - The original Matlab implementation
