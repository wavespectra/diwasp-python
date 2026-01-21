DIWASP-Python
=============

DIrectional WAve SPectrum analysis for Python
----------------------------------------------

DIWASP-Python is an open source library for estimating directional wave spectra from 
multi-sensor measurements. It is a Python port of the original DIWASP Matlab toolbox 
developed by David Johnson at the University of Western Australia.

The library provides a modern, user-friendly interface built on top of industry-standard 
scientific Python libraries including `xarray`_, `pandas`_, and `wavespectra`_.

.. _xarray: https://xarray.pydata.org/en/stable/
.. _pandas: https://pandas.pydata.org/
.. _wavespectra: https://wavespectra.readthedocs.io/

Key Features
------------

* **Multiple estimation methods**: DFTM, EMLM, IMLM, EMEP, and BDM
* **Flexible input formats**: pandas DataFrame or xarray Dataset
* **Seamless integration** with wavespectra for advanced wave analysis
* **Windowed analysis**: Process continuous time series with configurable window length and overlap
* **Multiple sensor types**: Pressure, velocity, acceleration, surface elevation, and more
* **Array configurations**: Support for single sensors or multi-sensor arrays
* **Modern output**: Returns wavespectra-compatible xarray Datasets

Documentation
-------------

**Getting Started**

* :doc:`installation`
* :doc:`wrapper`
* :doc:`data_structures`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Getting Started:

    installation
    wrapper
    data_structures

**Reference**

* :doc:`api_reference`
* :doc:`estimation_methods`
* :doc:`file_format`
* :doc:`references`

.. toctree::
    :maxdepth: 1
    :hidden:
    :caption: Reference:

    api_reference
    estimation_methods
    file_format
    references

Quick Example
-------------

.. code-block:: python

    import pandas as pd
    from diwasp import diwasp

    # Load wave data with datetime index
    df = pd.read_csv('wave_data.csv', index_col='time', parse_dates=True)

    # Run analysis over multiple windows
    result = diwasp(
        df,
        sensor_mapping={'pressure': 'pres', 'u': 'velx', 'v': 'vely'},
        window_length=1800,  # 30 minutes
        window_overlap=900,   # 15 minutes
        depth=20.0,
        z=0.5,  # sensors 0.5m above seabed
    )

    # Access wave statistics
    print(f"Hsig: {result.hsig.values}")
    print(f"Peak period: {result.tp.values}")
    print(f"Peak direction: {result.dp.values}")

History
-------

DIWASP was originally developed as a Matlab toolbox by David Johnson at the 
Centre for Water Research, University of Western Australia. This Python port 
maintains the core functionality while providing a modern interface compatible 
with the Python scientific ecosystem.

Citation
--------

If you use DIWASP in your research, please cite the original toolbox:

    Johnson, D. (2002). DIWASP, a directional wave spectra toolbox for MATLAB: User Manual. 
    Research Report WP-1601-DJ (V1.1), Centre for Water Research, University of Western Australia.

License
-------

DIWASP-Python is licensed under the MIT License.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
