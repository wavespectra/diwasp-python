# Installation

## Requirements

- Python >= 3.9
- numpy >= 1.20
- scipy >= 1.7
- xarray >= 0.19
- wavespectra >= 4.0
- pandas >= 1.3

## Install from PyPI

```bash
pip install diwasp
```

## Install from Source

```bash
git clone https://github.com/yourusername/diwasp-python.git
cd diwasp-python
pip install -e .
```

## Development Installation

For development, install with extra dependencies:

```bash
pip install -e ".[dev]"
```

This includes:

- pytest for testing
- black for code formatting
- ruff for linting
- mypy for type checking

## Verify Installation

```python
import diwasp
print(diwasp.__version__)
```

## Dependencies

The package integrates with:

- **[wavespectra](https://wavespectra.readthedocs.io/)**: For wave spectra analysis and manipulation
- **[xarray](https://xarray.pydata.org/)**: For labeled multi-dimensional arrays
- **[pandas](https://pandas.pydata.org/)**: For data manipulation and analysis
