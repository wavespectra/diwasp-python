"""DIWASP - DIrectional WAve SPectrum analysis.

A Python package for estimating directional wave spectra from multi-sensor
measurements. This is a port of the original DIWASP Matlab toolbox developed
by David Johnson at the Coastal Oceanography Group, UWA, Perth.

Main Functions
--------------
diwasp : High-level wrapper for analysis of DataFrame/Dataset over multiple windows
dirspec : Estimate directional spectrum from sensor data (single window)

Data Structures
---------------
InstrumentData : Sensor measurements and configuration
SpectralMatrix : 2D directional wave spectrum
EstimationParameters : Analysis configuration
SpectralInfo : Computed spectral statistics

Sensor Types
------------
SensorType : Enum of supported sensor types (ELEV, PRES, VELX, VELY, etc.)

Estimation Methods
------------------
EstimationMethod : Enum of estimation algorithms (DFTM, EMLM, IMLM, EMEP, BDM)

Spectrum Utilities
------------------
makespec : Generate synthetic directional spectra
interpspec : Interpolate spectra to new grids
make_wave_data : Generate synthetic sensor data

Example (High-level API)
------------------------
>>> import pandas as pd
>>> from diwasp import diwasp
>>>
>>> # Load sensor data with datetime index
>>> df = pd.read_csv('wave_data.csv', index_col='time', parse_dates=True)
>>>
>>> # Run analysis over multiple windows
>>> result = diwasp(
...     df,
...     sensor_mapping={'pressure': 'pres', 'u': 'velx', 'v': 'vely'},
...     window_length=1800,  # 30 minutes
...     window_overlap=900,   # 15 minutes
...     depth=20.0,
...     z=0.5,  # sensors 0.5m above seabed
... )
>>>
>>> print(f"Hsig: {result.hsig.values}")

Example (Low-level API)
-----------------------
>>> from diwasp import dirspec, InstrumentData, SensorType
>>> import numpy as np
>>>
>>> # Define sensor configuration manually
>>> instrument = InstrumentData(
...     data=data,
...     layout=np.array([[0, 10, 0], [0, 0, 10], [10, 10, 10]]).T,
...     datatypes=[SensorType.PRES, SensorType.VELX, SensorType.VELY],
...     depth=20.0,
...     fs=2.0
... )
>>>
>>> # Estimate spectrum for single window
>>> spectrum, info = dirspec(instrument)
>>> print(f"Hsig: {info.hsig:.2f} m")

References
----------
Main reference:
    Hashimoto, N. (1997) "Analysis of the directional wave spectrum from
    field data" in Advances in Coastal Engineering Vol. 3, World Scientific.

Original DIWASP:
    Johnson, D. (2002) DIWASP, a directional wave spectra toolbox for MATLAB:
    User Manual. Research Report WP-1601-DJ (V1.1), Centre for Water Research,
    University of Western Australia.
"""

__version__ = "0.1.0"

# High-level wrapper
from .wrapper import diwasp

# Core driver functions
from .core import dirspec

# Data structures
from .types import (
    EstimationMethod,
    EstimationParameters,
    InstrumentData,
    SensorType,
    SpectralInfo,
    SpectralMatrix,
)

# Spectrum utilities
from .spectrum import interpspec, make_wave_data, makespec

# Utility functions
from .utils import (
    directional_spread,
    hsig,
    mean_direction,
    peak_direction,
    peak_frequency,
    wavenumber,
)

__all__ = [
    # High-level wrapper
    "diwasp",
    # Core functions
    "dirspec",
    # Data structures
    "InstrumentData",
    "SpectralMatrix",
    "EstimationParameters",
    "SpectralInfo",
    # Enums
    "SensorType",
    "EstimationMethod",
    # Spectrum utilities
    "makespec",
    "interpspec",
    "make_wave_data",
    # Utility functions
    "wavenumber",
    "hsig",
    "peak_frequency",
    "peak_direction",
    "mean_direction",
    "directional_spread",
]
