# API Reference

## Core Functions

### dirspec

Main directional spectrum estimation routine. Takes measured data and information about sensors and returns the estimated directional spectrum.

```python
def dirspec(
    instrument_data: InstrumentData,
    estimation_params: EstimationParameters | None = None,
    freqs: NDArray | None = None,
    dirs: NDArray | None = None,
    verbose: int = 1,
) -> tuple[SpectralMatrix, SpectralInfo]:
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `instrument_data` | InstrumentData | Sensor measurements and configuration |
| `estimation_params` | EstimationParameters | Analysis parameters (uses defaults if None) |
| `freqs` | ndarray | Output frequency grid in Hz (auto if None) |
| `dirs` | ndarray | Output direction grid in degrees (auto if None) |
| `verbose` | int | Verbosity level (0=silent, 1=normal, 2=detailed) |

**Returns:**

- `SpectralMatrix`: Estimated directional spectrum
- `SpectralInfo`: Computed spectral statistics

**Example:**

```python
from diwasp import dirspec, InstrumentData, SensorType, EstimationParameters, EstimationMethod
import numpy as np

# Create instrument data
data = np.random.randn(2048, 3)
layout = np.array([[0, 10, 0], [0, 0, 10], [10, 10, 10]]).T
datatypes = [SensorType.PRES, SensorType.VELX, SensorType.VELY]

instrument = InstrumentData(
    data=data, layout=layout, datatypes=datatypes,
    depth=20.0, fs=2.0
)

# Estimate spectrum with custom parameters
params = EstimationParameters(method=EstimationMethod.EMEP, iter=50)
spectrum, info = dirspec(instrument, params, verbose=1)

print(f"Hsig: {info.hsig:.2f} m")
print(f"Peak period: {info.tp:.2f} s")
```

### dirspec_xarray

Convenience wrapper that accepts and returns xarray objects.

```python
def dirspec_xarray(
    ds: xr.Dataset,
    depth: float,
    fs: float,
    estimation_params: EstimationParameters | None = None,
    freqs: NDArray | None = None,
    dirs: NDArray | None = None,
    verbose: int = 1,
) -> xr.Dataset:
```

**Returns:**

xarray Dataset compatible with wavespectra package, containing:
- `efth`: Energy density variable
- Attributes with spectral statistics (hsig, tp, dp, etc.)

## Spectrum Utilities

### makespec

Generate a synthetic directional wave spectrum.

```python
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
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `freq_range` | tuple | (low_freq, peak_freq, high_freq) in Hz |
| `theta` | float or list | Mean wave direction(s) in degrees |
| `spread` | float or list | Spreading parameter(s), 25-100 typical |
| `weights` | float or list | Relative weights for multi-modal spectra |
| `hsig` | float | Target significant wave height (m) |
| `depth` | float | Water depth (m) |
| `n_freqs` | int | Number of frequency bins |
| `n_dirs` | int | Number of direction bins |

**Example:**

```python
from diwasp import makespec

# Unimodal spectrum
spectrum = makespec(
    freq_range=(0.05, 0.1, 0.3),
    theta=45.0,
    spread=50.0,
    hsig=2.0
)

# Bimodal spectrum (wind sea + swell)
spectrum = makespec(
    freq_range=(0.04, 0.08, 0.3),
    theta=[270.0, 180.0],
    spread=[25.0, 75.0],
    weights=[0.3, 0.7],
    hsig=3.0
)
```

### interpspec

Interpolate spectrum to a new frequency/direction grid.

```python
def interpspec(
    spectrum: SpectralMatrix,
    freqs_out: NDArray | None = None,
    dirs_out: NDArray | None = None,
) -> SpectralMatrix:
```

### make_wave_data

Generate synthetic sensor data from a directional spectrum.

```python
def make_wave_data(
    spectrum: SpectralMatrix,
    instrument_data: InstrumentData,
    n_samples: int,
    noise_level: float = 0.0,
    seed: int | None = None,
) -> NDArray:
```

**Example:**

```python
from diwasp import makespec, make_wave_data, InstrumentData, SensorType
import numpy as np

# Create spectrum
spectrum = makespec(
    freq_range=(0.05, 0.1, 0.3),
    theta=45.0,
    spread=50.0,
    hsig=1.5
)

# Define sensor configuration
layout = np.array([[0, 5, 0], [0, 0, 5], [10, 10, 10]]).T
datatypes = [SensorType.PRES, SensorType.VELX, SensorType.VELY]

instrument = InstrumentData(
    data=np.zeros((100, 3)),  # Placeholder
    layout=layout,
    datatypes=datatypes,
    depth=20.0,
    fs=2.0
)

# Generate synthetic data
data = make_wave_data(spectrum, instrument, n_samples=4096, noise_level=0.01, seed=42)
```

## Utility Functions

### wavenumber

Calculate wavenumber from angular frequency using the dispersion relation.

```python
def wavenumber(
    sigma: NDArray | float,
    depth: float,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> NDArray:
```

Solves: `sigma^2 = g * k * tanh(k * d)`

### hsig

Calculate significant wave height from directional spectrum.

```python
def hsig(
    S: NDArray,
    freqs: NDArray,
    dirs: NDArray,
) -> float:
```

### peak_frequency

Calculate peak frequency from directional spectrum.

```python
def peak_frequency(S: NDArray, freqs: NDArray) -> float:
```

### peak_direction

Calculate peak direction from directional spectrum.

```python
def peak_direction(S: NDArray, freqs: NDArray, dirs: NDArray) -> float:
```

### mean_direction

Calculate energy-weighted mean direction (circular mean).

```python
def mean_direction(S: NDArray, dirs: NDArray) -> float:
```

### directional_spread

Calculate directional spread (circular standard deviation).

```python
def directional_spread(S: NDArray, dirs: NDArray) -> float:
```

## Estimation Methods

All estimation methods inherit from `EstimationMethodBase` and implement the `estimate` method:

```python
class EstimationMethodBase(ABC):
    def estimate(
        self,
        csd_matrix: NDArray,
        transfer_matrix: NDArray,
        kx: NDArray,
    ) -> NDArray:
        """Estimate directional spectrum from cross-spectral density."""
        pass
```

### Available Methods

| Class | Description |
|-------|-------------|
| `DFTM` | Direct Fourier Transform Method |
| `EMLM` | Extended Maximum Likelihood Method |
| `IMLM` | Iterated Maximum Likelihood Method |
| `EMEP` | Extended Maximum Entropy Principle |
| `BDM` | Bayesian Direct Method |

### Custom Method Example

```python
from diwasp.methods import EstimationMethodBase
import numpy as np

class CustomMethod(EstimationMethodBase):
    def estimate(self, csd_matrix, transfer_matrix, kx):
        n_freqs, n_dirs, n_sensors = transfer_matrix.shape
        S = np.zeros((n_freqs, n_dirs))
        # Custom estimation logic here
        return S
```
