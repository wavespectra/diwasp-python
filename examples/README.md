# DIWASP-Python Examples

This directory contains example notebooks and scripts demonstrating the use of DIWASP-Python.

## Notebooks

### demo_analysis.ipynb

Comprehensive demonstration of end-to-end wave analysis workflow including:

1. **Steady Sea State Analysis** - Creating synthetic data and analyzing with the DIWASP wrapper
2. **Varying Sea State** - Tracking changes in wave parameters over time with slowly varying conditions
3. **Method Comparison** - Comparing different estimation algorithms (DFTM, EMLM, IMLM, EMEP, BDM)
4. **Pressure Array Analysis** - Using multiple sensors in different spatial locations

The notebook demonstrates:

- Creating synthetic wave spectra with `makespec()`
- Generating sensor data with `make_wave_data()`
- Running analysis with the `diwasp()` wrapper function
- Visualizing results (time series, 2D spectra, comparisons)
- Working with both pandas DataFrame and xarray Dataset inputs

## Running the Notebooks

Install Jupyter and visualization dependencies:

```bash
pip install jupyter matplotlib
```

Launch Jupyter:

```bash
jupyter notebook examples/demo_analysis.ipynb
```

**Note**: The notebook currently needs API corrections for the `make_wave_data()` function to match the actual implementation which requires an `InstrumentData` object.

## End-to-End Tests

See `tests/test_end_to_end.py` for comprehensive end-to-end tests covering:

- Steady and varying sea states
- Different sensor configurations (PUV, pressure arrays)
- All estimation methods
- Edge cases (short duration, high frequency, bimodal spectra)

Run the tests with:

```bash
pytest tests/test_end_to_end.py -v
```

**Note**: Tests also need API corrections for `make_wave_data()` function usage.
