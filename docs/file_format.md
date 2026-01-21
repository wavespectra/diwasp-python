# File Format

## DIWASP Spectrum File Format

The original DIWASP toolbox uses its own ASCII format for storing spectrum files. This format is simple and portable.

### Format Structure

The file consists of a single ASCII stream of numbers:

| Position | Type | Description |
|----------|------|-------------|
| 1 | Real | Compass direction of x-axis |
| 2 | Integer | Number of frequency bins (nf) |
| 3 | Integer | Number of directional bins (nd) |
| 4 to nf+3 | Real | List of frequencies (SM.freqs) |
| nf+4 to nf+nd+3 | Real | List of directions (SM.dirs) |
| nf+nd+4 | Integer | 999 (marks end of header) |
| nf+nd+5 to end | Real | Spectral density values (SM.S) |

The spectral density is written with frequency as the outer loop:

```
for i = 1 to nf:
    for j = 1 to nd:
        write S[i, j]
```

### Example File

```
90.0
3
4
0.05
0.10
0.15
0.0
90.0
180.0
270.0
999
0.001
0.002
0.003
0.004
0.010
0.020
0.015
0.008
0.005
0.008
0.006
0.003
```

This represents a 3x4 spectrum with:
- x-axis at 90 degrees (East)
- Frequencies: 0.05, 0.10, 0.15 Hz
- Directions: 0, 90, 180, 270 degrees

## Python File I/O

### Using xarray (Recommended)

The recommended approach is to use xarray/NetCDF for file I/O:

```python
from diwasp import dirspec

# Estimate spectrum
spectrum, info = dirspec(instrument)

# Convert to xarray and save
ds = spectrum.to_xarray()
ds.to_netcdf('spectrum.nc')

# Load back
import xarray as xr
ds_loaded = xr.open_dataset('spectrum.nc')
```

### Using wavespectra

For compatibility with the wavespectra ecosystem:

```python
import wavespectra

# Save using wavespectra format
ds = spectrum.to_xarray()
ds.spec.to_netcdf('spectrum.nc')

# Load with wavespectra
spec = wavespectra.read_netcdf('spectrum.nc')
```

### Legacy DIWASP Format (if needed)

To read/write the original DIWASP format:

```python
def write_diwasp_spec(spectrum, filename):
    """Write spectrum in DIWASP ASCII format."""
    with open(filename, 'w') as f:
        f.write(f"{spectrum.xaxisdir}\n")
        f.write(f"{len(spectrum.freqs)}\n")
        f.write(f"{len(spectrum.dirs)}\n")

        for freq in spectrum.freqs:
            f.write(f"{freq}\n")

        for dir in spectrum.dirs:
            f.write(f"{dir}\n")

        f.write("999\n")

        for i in range(len(spectrum.freqs)):
            for j in range(len(spectrum.dirs)):
                f.write(f"{spectrum.S[i, j]}\n")


def read_diwasp_spec(filename):
    """Read spectrum from DIWASP ASCII format."""
    from diwasp import SpectralMatrix
    import numpy as np

    with open(filename, 'r') as f:
        xaxisdir = float(f.readline())
        nf = int(f.readline())
        nd = int(f.readline())

        freqs = np.array([float(f.readline()) for _ in range(nf)])
        dirs = np.array([float(f.readline()) for _ in range(nd)])

        marker = int(f.readline())  # Should be 999
        assert marker == 999, "Invalid file format"

        S = np.zeros((nf, nd))
        for i in range(nf):
            for j in range(nd):
                S[i, j] = float(f.readline())

    return SpectralMatrix(
        freqs=freqs,
        dirs=dirs,
        S=S,
        xaxisdir=xaxisdir
    )
```

## NetCDF Format Details

When saved via xarray, the NetCDF file contains:

### Dimensions

- `freq`: Frequency dimension
- `dir`: Direction dimension

### Variables

- `efth(freq, dir)`: Spectral energy density (m^2/Hz/degree)

### Coordinates

- `freq`: Frequency values (Hz)
- `dir`: Direction values (degrees)

### Attributes

- `xaxisdir`: Compass direction of x-axis
- `funit`: Frequency units
- `dunit`: Direction units
- `hsig`: Significant wave height (if computed)
- `tp`: Peak period (if computed)
- `dp`: Peak direction (if computed)
