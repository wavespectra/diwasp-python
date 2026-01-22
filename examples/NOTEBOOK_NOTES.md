# Demo Notebook API Notes

The `demo_analysis.ipynb` notebook needs updates to match the actual DIWASP API.

## API Corrections Needed

### 1. makespec() Function

**Current (incorrect) usage in notebook:**

```python
makespec(
    freqs=np.linspace(0.05, 0.5, 50),
    dirs=np.linspace(0, 360, 181, endpoint=False),
    spreading=75,
    frequency_hz=0.1,
    direction_deg=45,
    gamma=3.3,
)
```

**Correct API:**

```python
makespec(
    freq_range=(0.05, 0.1, 0.5),  # (low, peak, high) in Hz
    theta=45.0,                    # Peak direction in degrees
    spread=75.0,                   # Directional spreading parameter
    hsig=2.0,                      # Significant wave height
    depth=20.0,                    # Water depth
    n_freqs=50,                    # Number of frequency bins
    n_dirs=180,                    # Number of direction bins
)
```

### 2. make_wave_data() Function

**Current (incorrect) usage in notebook:**

```python
make_wave_data(
    spec=spec,
    layout=layout,
    datatypes=['pres', 'velx', 'vely'],
    depth=20.0,
    fs=fs,
    duration=duration,
)
```

**Correct API:**

```python
from diwasp import InstrumentData, SensorType

# Create InstrumentData object
instrument = InstrumentData(
    data=np.zeros((n_samples, 3)),  # Placeholder
    layout=layout,
    datatypes=[SensorType.PRES, SensorType.VELX, SensorType.VELY],
    depth=20.0,
    fs=fs
)

# Generate wave data
data = make_wave_data(
    spectrum=spec,
    instrument_data=instrument,
    n_samples=n_samples,
)
```

## Cells That Need Updating

The following cells in the notebook need to be corrected:

1. **Cell 2**: First makespec() call - ✅ FIXED
2. **Cell 3**: First make_wave_data() call - ✅ FIXED
3. **Cell 8**: Varying sea state makespec() calls (4 times in loop)
4. **Cell 11**: Method comparison makespec() call
5. **Cell 12**: Method comparison make_wave_data() call
6. **Cell 14**: Array analysis makespec() call
7. **Cell 15**: Array analysis make_wave_data() call

## Status

- Cells 2-3: Fixed
- Remaining cells: Need manual correction or script to update

The test file `tests/test_end_to_end.py` has already been corrected with the proper API usage and can serve as a reference.
