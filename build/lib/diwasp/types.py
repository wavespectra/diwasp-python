"""Type definitions and data structures for DIWASP.

This module defines the core data structures used throughout the DIWASP package,
following the original Matlab implementation's three main structures:
- InstrumentData (ID): Sensor measurements and configuration
- SpectralMatrix (SM): 2D directional wave spectrum
- EstimationParameters (EP): Analysis configuration
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray


class SensorType(str, Enum):
    """Sensor measurement types supported by DIWASP.

    Each sensor type has an associated transfer function that converts
    measurements to equivalent surface elevation spectra.
    """

    ELEV = "elev"  # Surface elevation
    PRES = "pres"  # Pressure
    VELX = "velx"  # Horizontal velocity (x-component)
    VELY = "vely"  # Horizontal velocity (y-component)
    VELZ = "velz"  # Vertical velocity
    VELS = "vels"  # Surface velocity
    ACCX = "accx"  # Horizontal acceleration (x-component)
    ACCY = "accy"  # Horizontal acceleration (y-component)
    ACCZ = "accz"  # Vertical acceleration
    ACCS = "accs"  # Surface acceleration
    SLPX = "slpx"  # Surface slope (x-component)
    SLPY = "slpy"  # Surface slope (y-component)
    DSPX = "dspx"  # Horizontal displacement (x-component)
    DSPY = "dspy"  # Horizontal displacement (y-component)


class EstimationMethod(str, Enum):
    """Directional spectrum estimation methods.

    Each method implements a different algorithm for estimating
    the directional wave spectrum from cross-spectral density.
    """

    DFTM = "DFTM"  # Direct Fourier Transform Method
    EMLM = "EMLM"  # Extended Maximum Likelihood Method
    IMLM = "IMLM"  # Iterated Maximum Likelihood Method
    EMEP = "EMEP"  # Extended Maximum Entropy Principle
    BDM = "BDM"  # Bayesian Direct Method


@dataclass
class InstrumentData:
    """Instrument data structure containing measured wave sensor data.

    Attributes:
        data: Matrix of measured wave data [n_samples x n_sensors].
        layout: Sensor positions [3 x n_sensors] as (x, y, z) where z is
            elevation from seabed.
        datatypes: List of sensor types, one per column of data.
        depth: Mean water depth in meters.
        fs: Sampling frequency in Hz.
    """

    data: NDArray[np.floating]
    layout: NDArray[np.floating]
    datatypes: list[SensorType]
    depth: float
    fs: float

    def __post_init__(self) -> None:
        """Validate input data."""
        n_sensors = self.data.shape[1] if self.data.ndim > 1 else 1
        if self.layout.shape[1] != n_sensors:
            raise ValueError(
                f"Layout columns ({self.layout.shape[1]}) must match "
                f"number of sensors ({n_sensors})"
            )
        if len(self.datatypes) != n_sensors:
            raise ValueError(
                f"Number of datatypes ({len(self.datatypes)}) must match "
                f"number of sensors ({n_sensors})"
            )
        if self.depth <= 0:
            raise ValueError(f"Water depth must be positive, got {self.depth}")
        if self.fs <= 0:
            raise ValueError(f"Sampling frequency must be positive, got {self.fs}")

    @property
    def n_samples(self) -> int:
        """Number of time samples."""
        return self.data.shape[0]

    @property
    def n_sensors(self) -> int:
        """Number of sensors."""
        return self.data.shape[1] if self.data.ndim > 1 else 1

    @classmethod
    def from_xarray(cls, ds: xr.Dataset, depth: float, fs: float) -> "InstrumentData":
        """Create InstrumentData from an xarray Dataset.

        The dataset should have data variables with sensor measurements and
        attributes specifying sensor types and positions.

        Args:
            ds: xarray Dataset with sensor data variables.
            depth: Mean water depth in meters.
            fs: Sampling frequency in Hz.

        Returns:
            InstrumentData instance.
        """
        # Extract sensor data and metadata from dataset
        data_vars = list(ds.data_vars)
        data_list = []
        datatypes = []
        positions = []

        for var in data_vars:
            da = ds[var]
            data_list.append(da.values)

            # Get sensor type from attributes
            sensor_type = da.attrs.get("sensor_type", "elev")
            datatypes.append(SensorType(sensor_type))

            # Get position from attributes (default to origin)
            x = da.attrs.get("x", 0.0)
            y = da.attrs.get("y", 0.0)
            z = da.attrs.get("z", depth)  # Default z to water depth (surface)
            positions.append([x, y, z])

        data = np.column_stack(data_list)
        layout = np.array(positions).T

        return cls(
            data=data,
            layout=layout,
            datatypes=datatypes,
            depth=depth,
            fs=fs,
        )


@dataclass
class SpectralMatrix:
    """Spectral matrix structure representing a 2D directional wave spectrum.

    Attributes:
        freqs: Frequency bin centers in Hz.
        dirs: Direction bin centers in degrees.
        S: Spectral density matrix [n_freqs x n_dirs] in m^2/(Hz*degree).
        xaxisdir: Reference x-axis direction in compass degrees (default 90 = East).
        funit: Frequency units ('hz' or 'rad/s').
        dunit: Direction units ('cart' for Cartesian, 'naut' for nautical/compass).
    """

    freqs: NDArray[np.floating]
    dirs: NDArray[np.floating]
    S: NDArray[np.floating]
    xaxisdir: float = 90.0
    funit: Literal["hz", "rad/s"] = "hz"
    dunit: Literal["cart", "naut"] = "naut"

    def __post_init__(self) -> None:
        """Validate spectral matrix dimensions."""
        if self.S.shape != (len(self.freqs), len(self.dirs)):
            raise ValueError(
                f"Spectral matrix shape {self.S.shape} must match "
                f"(n_freqs={len(self.freqs)}, n_dirs={len(self.dirs)})"
            )

    @property
    def n_freqs(self) -> int:
        """Number of frequency bins."""
        return len(self.freqs)

    @property
    def n_dirs(self) -> int:
        """Number of direction bins."""
        return len(self.dirs)

    @property
    def df(self) -> float:
        """Frequency resolution in Hz."""
        if len(self.freqs) < 2:
            return 0.0
        return float(np.mean(np.diff(self.freqs)))

    @property
    def ddir(self) -> float:
        """Direction resolution in degrees."""
        if len(self.dirs) < 2:
            return 0.0
        return float(np.mean(np.diff(self.dirs)))

    def to_xarray(self) -> xr.Dataset:
        """Convert to xarray Dataset compatible with wavespectra.

        Returns:
            xarray Dataset with 'efth' variable (energy density).
        """
        # wavespectra expects dimensions (time, freq, dir) or (freq, dir)
        ds = xr.Dataset(
            {
                "efth": (["freq", "dir"], self.S),
            },
            coords={
                "freq": self.freqs,
                "dir": self.dirs,
            },
            attrs={
                "xaxisdir": self.xaxisdir,
                "funit": self.funit,
                "dunit": self.dunit,
            },
        )
        ds["efth"].attrs["units"] = "m^2/Hz/degree"
        ds["efth"].attrs["long_name"] = "Spectral energy density"
        return ds

    @classmethod
    def from_xarray(cls, ds: xr.Dataset) -> "SpectralMatrix":
        """Create SpectralMatrix from xarray Dataset.

        Args:
            ds: xarray Dataset with 'efth' variable.

        Returns:
            SpectralMatrix instance.
        """
        S = ds["efth"].values
        freqs = ds["freq"].values
        dirs = ds["dir"].values

        xaxisdir = ds.attrs.get("xaxisdir", 90.0)
        funit = ds.attrs.get("funit", "hz")
        dunit = ds.attrs.get("dunit", "cart")

        return cls(
            freqs=freqs,
            dirs=dirs,
            S=S,
            xaxisdir=xaxisdir,
            funit=funit,
            dunit=dunit,
        )


@dataclass
class EstimationParameters:
    """Estimation parameters structure for spectrum analysis configuration.

    Attributes:
        method: Estimation algorithm to use.
        nfft: FFT length for frequency resolution (auto-calculated if None).
        dres: Directional resolution (number of bins for 360 degrees).
        iter: Maximum iteration count for iterative methods.
        smooth: Whether to apply spectral smoothing.
    """

    method: EstimationMethod = EstimationMethod.IMLM
    nfft: int | None = None
    dres: int = 180
    iter: int = 100
    smooth: bool = True

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.dres < 4:
            raise ValueError(f"dres must be at least 4, got {self.dres}")
        if self.iter < 1:
            raise ValueError(f"iter must be at least 1, got {self.iter}")


@dataclass
class SpectralInfo:
    """Statistics computed from a directional spectrum.

    Attributes:
        hsig: Significant wave height (Hs) in meters.
        tp: Peak period in seconds.
        fp: Peak frequency in Hz.
        dp: Peak direction in degrees.
        dm: Mean direction in degrees.
        spread: Directional spread in degrees.
    """

    hsig: float
    tp: float
    fp: float
    dp: float
    dm: float
    spread: float
