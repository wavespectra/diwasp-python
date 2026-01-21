# DIWASP - DIrectional WAve SPectra Toolbox

**Python Port - Version 0.1.0**

A Python package for estimating directional wave spectra from multi-sensor measurements. This is a port of the original DIWASP Matlab toolbox developed by David Johnson at the Coastal Oceanography Group, Centre for Water Research, University of Western Australia.

## Overview

DIWASP is a toolbox for the estimation of directional wave spectra. Spectra can be calculated from a variety of data types using a single function `dirspec`. Five different estimation methods are available depending on the quality or speed of estimation required. Utility functions are also included to manage spectra, plot results, and run tests on the estimation methods.

## Features

- **Multiple estimation methods**: DFTM, EMLM, IMLM, EMEP, BDM
- **Flexible sensor support**: Pressure, velocity, elevation, acceleration, slope, and displacement sensors
- **xarray/wavespectra integration**: Output compatible with the wavespectra package
- **Synthetic spectrum generation**: Create test spectra and sensor data

## Supported Data Types

All standard wave recorder data types are supported:

- Surface elevation (`elev`)
- Pressure (`pres`)
- Current velocity components (`velx`, `vely`, `velz`)
- Surface slope components (`slpx`, `slpy`)
- Water surface vertical velocity (`vels`)
- Water surface vertical acceleration (`accs`)
- Current accelerations (`accx`, `accy`, `accz`)
- Horizontal displacement (`dspx`, `dspy`)

## Estimation Methods

Five different estimation methods can be used, each with different levels of performance in terms of accuracy, speed, and suitability for different data types:

| Method | Name                               | Reference                   |
| ------ | ---------------------------------- | --------------------------- |
| DFTM   | Direct Fourier Transform Method    | Barber (1961)               |
| EMLM   | Extended Maximum Likelihood Method | Isobe et al. (1984)         |
| IMLM   | Iterated Maximum Likelihood Method | Pawka (1983)                |
| EMEP   | Extended Maximum Entropy Principle | Hashimoto et al. (1993)     |
| BDM    | Bayesian Direct Method             | Hashimoto and Kobune (1987) |

## Quick Start

```python
from diwasp import dirspec, InstrumentData, SensorType, EstimationParameters
import numpy as np

# Load or create sensor data
data = np.loadtxt('wave_data.csv')  # [n_samples x n_sensors]

# Define sensor configuration
instrument = InstrumentData(
    data=data,
    layout=np.array([[0, 10, 0], [0, 0, 10], [10, 10, 10]]).T,
    datatypes=[SensorType.PRES, SensorType.VELX, SensorType.VELY],
    depth=20.0,
    fs=2.0
)

# Estimate spectrum
spectrum, info = dirspec(instrument)

print(f"Hsig: {info.hsig:.2f} m")
print(f"Peak period: {info.tp:.2f} s")
print(f"Peak direction: {info.dp:.1f} deg")
```

## License

DIWASP is free software distributed under the MIT License. See LICENSE for details.

## Citation

This document should be referenced as:

> Johnson, D. (2002) "DIWASP, a directional wave spectra toolbox for MATLAB: User Manual." Research Report WP-1601-DJ (V1.1), Centre for Water Research, University of Western Australia.

## Documentation Contents

- [Installation](installation.md)
- [Data Structures](data_structures.md)
- [API Reference](api_reference.md)
- [Estimation Methods](estimation_methods.md)
- [File Format](file_format.md)
- [References](references.md)
