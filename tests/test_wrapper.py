"""Tests for the high-level diwasp wrapper function."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from diwasp import diwasp


class TestDiwaspDataFrame:
    """Tests for diwasp with pandas DataFrame input."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        np.random.seed(42)

        # Create 1 hour of data at 2 Hz
        n_samples = 7200
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")

        # Create simple sinusoidal wave signal
        t = np.arange(n_samples) / 2.0
        f = 0.1  # 10 second waves

        df = pd.DataFrame(
            {
                "pressure": np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "u_vel": np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "v_vel": 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
            },
            index=time,
        )
        return df

    def test_basic_analysis(self, sample_dataframe):
        """Test basic analysis runs without error."""
        result = diwasp(
            sample_dataframe,
            sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
            window_length=600,  # 10 minutes
            window_overlap=300,  # 5 minutes
            depth=20.0,
            z=0.5,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars
        assert "hsig" in result.data_vars
        assert "tp" in result.data_vars
        assert "dp" in result.data_vars
        assert "time" in result.dims
        assert "freq" in result.dims
        assert "dir" in result.dims

    def test_multiple_windows(self, sample_dataframe):
        """Test that multiple windows are generated."""
        result = diwasp(
            sample_dataframe,
            sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            verbose=0,
        )

        # With 1 hour of data, 10 min windows, 5 min overlap:
        # (3600 - 600) / 300 + 1 = 11 windows
        assert len(result.time) > 1

    def test_different_methods(self, sample_dataframe):
        """Test different estimation methods."""
        for method in ["dftm", "emlm", "imlm"]:
            result = diwasp(
                sample_dataframe,
                sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
                window_length=600,
                window_overlap=300,
                depth=20.0,
                method=method,
                verbose=0,
            )
            assert isinstance(result, xr.Dataset)

    def test_custom_z_positions(self, sample_dataframe):
        """Test with different z positions per sensor."""
        result = diwasp(
            sample_dataframe,
            sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            z={"pressure": 0.5, "u_vel": 1.0, "v_vel": 1.0},
            verbose=0,
        )
        assert isinstance(result, xr.Dataset)

    def test_custom_xy_positions(self, sample_dataframe):
        """Test with custom x, y positions."""
        result = diwasp(
            sample_dataframe,
            sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            x={"pressure": 0, "u_vel": 5, "v_vel": 0},
            y={"pressure": 0, "u_vel": 0, "v_vel": 5},
            z=1.0,
            verbose=0,
        )
        assert isinstance(result, xr.Dataset)

    def test_custom_frequency_grid(self, sample_dataframe):
        """Test with custom frequency grid."""
        freqs = np.linspace(0.05, 0.3, 20)

        result = diwasp(
            sample_dataframe,
            sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            freqs=freqs,
            verbose=0,
        )

        assert len(result.freq) == 20

    def test_custom_direction_grid(self, sample_dataframe):
        """Test with custom direction resolution."""
        result = diwasp(
            sample_dataframe,
            sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            dres=90,  # 4 degree bins
            verbose=0,
        )

        assert len(result.dir) == 90

    def test_invalid_index_raises(self):
        """Test that non-datetime index raises error."""
        df = pd.DataFrame({"pressure": [1, 2, 3], "u_vel": [1, 2, 3], "v_vel": [1, 2, 3]})

        with pytest.raises(ValueError, match="DatetimeIndex"):
            diwasp(
                df,
                sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
                window_length=1,
                window_overlap=0,
                depth=10.0,
                verbose=0,
            )

    def test_missing_column_raises(self, sample_dataframe):
        """Test that missing columns raise error."""
        with pytest.raises(ValueError, match="not found"):
            diwasp(
                sample_dataframe,
                sensor_mapping={"nonexistent": "pres"},
                window_length=600,
                window_overlap=300,
                depth=20.0,
                verbose=0,
            )

    def test_invalid_method_raises(self, sample_dataframe):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown method"):
            diwasp(
                sample_dataframe,
                sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
                window_length=600,
                window_overlap=300,
                depth=20.0,
                method="invalid",
                verbose=0,
            )

    def test_window_too_long_raises(self, sample_dataframe):
        """Test that window longer than data raises error."""
        with pytest.raises(ValueError, match="exceeds data length"):
            diwasp(
                sample_dataframe,
                sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
                window_length=10000,  # Longer than data
                window_overlap=0,
                depth=20.0,
                verbose=0,
            )


class TestDiwaspDataset:
    """Tests for diwasp with xarray Dataset input."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample Dataset for testing."""
        np.random.seed(42)

        n_samples = 7200
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")

        t = np.arange(n_samples) / 2.0
        f = 0.1

        ds = xr.Dataset(
            {
                "pres": (["time"], np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples)),
                "velx": (["time"], np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples)),
                "vely": (
                    ["time"],
                    0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                ),
            },
            coords={"time": time.values},
        )
        return ds

    def test_basic_analysis(self, sample_dataset):
        """Test basic analysis with Dataset."""
        result = diwasp(
            sample_dataset,
            sensor_mapping={"pres": "pres", "velx": "velx", "vely": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars
        assert "hsig" in result.data_vars

    def test_dataset_with_z_coords(self):
        """Test Dataset with z coordinates as variable attributes."""
        np.random.seed(42)

        n_samples = 3600
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0
        f = 0.1

        ds = xr.Dataset(
            {
                "pres": (["time"], np.sin(2 * np.pi * f * t), {"z": 0.5}),
                "velx": (["time"], np.cos(2 * np.pi * f * t), {"z": 1.0}),
                "vely": (["time"], 0.5 * np.sin(2 * np.pi * f * t), {"z": 1.0}),
            },
            coords={"time": time.values},
        )

        result = diwasp(
            ds,
            sensor_mapping={"pres": "pres", "velx": "velx", "vely": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)

    def test_missing_time_var_raises(self, sample_dataset):
        """Test that missing time variable raises error."""
        with pytest.raises(ValueError, match="not found"):
            diwasp(
                sample_dataset,
                sensor_mapping={"pres": "pres", "velx": "velx", "vely": "vely"},
                window_length=600,
                window_overlap=300,
                depth=20.0,
                time_var="nonexistent",
                verbose=0,
            )

    def test_missing_variable_raises(self, sample_dataset):
        """Test that missing variables raise error."""
        with pytest.raises(ValueError, match="not found"):
            diwasp(
                sample_dataset,
                sensor_mapping={"nonexistent": "pres"},
                window_length=600,
                window_overlap=300,
                depth=20.0,
                verbose=0,
            )


class TestDiwaspOutput:
    """Tests for output format and wavespectra compatibility."""

    @pytest.fixture
    def result_dataset(self):
        """Create a result from diwasp for testing."""
        np.random.seed(42)

        n_samples = 3600
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0
        f = 0.1

        df = pd.DataFrame(
            {
                "p": np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "u": np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
            },
            index=time,
        )

        return diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            verbose=0,
        )

    def test_output_dimensions(self, result_dataset):
        """Test output has correct dimensions."""
        assert set(result_dataset.dims) == {"time", "freq", "dir"}

    def test_output_variables(self, result_dataset):
        """Test output has required variables."""
        required_vars = ["efth", "hsig", "tp", "fp", "dp", "dm", "spread"]
        for var in required_vars:
            assert var in result_dataset.data_vars

    def test_efth_shape(self, result_dataset):
        """Test efth has correct shape."""
        efth = result_dataset["efth"]
        assert efth.dims == ("time", "freq", "dir")

    def test_statistics_shape(self, result_dataset):
        """Test statistics have correct shape."""
        for var in ["hsig", "tp", "fp", "dp", "dm", "spread"]:
            assert result_dataset[var].dims == ("time",)

    def test_positive_hsig(self, result_dataset):
        """Test that hsig is non-negative."""
        assert np.all(result_dataset["hsig"].values >= 0)

    def test_positive_tp(self, result_dataset):
        """Test that tp is positive."""
        assert np.all(result_dataset["tp"].values > 0)

    def test_dir_range(self, result_dataset):
        """Test direction coordinates are in valid range."""
        dirs = result_dataset["dir"].values
        assert np.all(dirs >= 0)
        assert np.all(dirs < 360)

    def test_freq_positive(self, result_dataset):
        """Test frequency coordinates are positive."""
        freqs = result_dataset["freq"].values
        assert np.all(freqs > 0)

    def test_output_attributes(self, result_dataset):
        """Test output has required attributes."""
        assert "source" in result_dataset.attrs
        assert result_dataset.attrs["source"] == "diwasp"

    def test_variable_attributes(self, result_dataset):
        """Test variables have units and long_name."""
        for var in ["efth", "hsig", "tp"]:
            assert "units" in result_dataset[var].attrs
            assert "long_name" in result_dataset[var].attrs


class TestDiwaspInvalidInput:
    """Tests for invalid input handling."""

    def test_invalid_data_type_raises(self):
        """Test that invalid data type raises error."""
        with pytest.raises(TypeError, match="must be pandas DataFrame or xarray Dataset"):
            diwasp(
                [1, 2, 3],  # Invalid type
                sensor_mapping={"a": "pres"},
                window_length=1,
                window_overlap=0,
                depth=10.0,
            )

    def test_non_datetime_time_raises(self):
        """Test that non-datetime time variable raises error."""
        ds = xr.Dataset(
            {"pres": (["time"], [1, 2, 3])},
            coords={"time": [0, 1, 2]},  # Integer time, not datetime
        )

        with pytest.raises(ValueError, match="datetime type"):
            diwasp(
                ds,
                sensor_mapping={"pres": "pres"},
                window_length=1,
                window_overlap=0,
                depth=10.0,
                verbose=0,
            )


class TestDiwaspEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def short_dataframe(self):
        """Create short DataFrame for edge case testing."""
        np.random.seed(42)
        n_samples = 1200  # 10 minutes at 2 Hz
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0
        f = 0.1

        return pd.DataFrame(
            {
                "p": np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "u": np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
            },
            index=time,
        )

    def test_zero_overlap(self, short_dataframe):
        """Test with zero window overlap."""
        result = diwasp(
            short_dataframe,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=300,  # 5 minutes
            window_overlap=0,  # No overlap
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        # 600s data / 300s window = 2 windows
        assert len(result.time) == 2

    def test_single_window(self, short_dataframe):
        """Test when window length equals data length (single window)."""
        result = diwasp(
            short_dataframe,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,  # Exactly matches data length
            window_overlap=0,
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert len(result.time) == 1

    def test_high_overlap(self, short_dataframe):
        """Test with high overlap (75%)."""
        result = diwasp(
            short_dataframe,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=300,
            window_overlap=225,  # 75% overlap
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert len(result.time) > 1

    def test_overlap_greater_than_window_raises(self, short_dataframe):
        """Test that overlap >= window length raises error."""
        with pytest.raises(ValueError, match="[Oo]verlap"):
            diwasp(
                short_dataframe,
                sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
                window_length=300,
                window_overlap=300,  # Equal to window length
                depth=20.0,
                verbose=0,
            )

    def test_negative_depth_raises(self, short_dataframe):
        """Test that negative depth raises error."""
        with pytest.raises(ValueError, match="[Dd]epth"):
            diwasp(
                short_dataframe,
                sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
                window_length=300,
                window_overlap=0,
                depth=-10.0,
                verbose=0,
            )

    def test_empty_sensor_mapping_raises(self, short_dataframe):
        """Test that empty sensor mapping raises error."""
        with pytest.raises(ValueError, match="[Ss]ensor"):
            diwasp(
                short_dataframe,
                sensor_mapping={},
                window_length=300,
                window_overlap=0,
                depth=20.0,
                verbose=0,
            )


class TestDiwaspAllMethods:
    """Tests for all estimation methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for method testing."""
        np.random.seed(42)
        n_samples = 2400  # 20 minutes at 2 Hz
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0
        f = 0.1

        return pd.DataFrame(
            {
                "p": np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "u": np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
            },
            index=time,
        )

    @pytest.mark.parametrize("method", ["dftm", "emlm", "imlm", "emep", "bdm"])
    def test_all_methods(self, sample_data, method):
        """Test that all estimation methods run successfully."""
        result = diwasp(
            sample_data,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            method=method,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars
        assert "hsig" in result.data_vars
        assert not np.any(np.isnan(result["hsig"].values))

    @pytest.mark.parametrize("method", ["DFTM", "Emlm", "IMLM"])
    def test_case_insensitive_method(self, sample_data, method):
        """Test that method names are case-insensitive."""
        result = diwasp(
            sample_data,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            method=method,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)


class TestDiwaspSensorConfigurations:
    """Tests for different sensor configurations."""

    @pytest.fixture
    def multi_sensor_data(self):
        """Create data with multiple sensor types."""
        np.random.seed(42)
        n_samples = 2400
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0
        f = 0.1

        return pd.DataFrame(
            {
                "p1": np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "p2": np.sin(2 * np.pi * f * t + 0.1) + 0.1 * np.random.randn(n_samples),
                "p3": np.sin(2 * np.pi * f * t + 0.2) + 0.1 * np.random.randn(n_samples),
                "u": np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "w": 0.2 * np.cos(2 * np.pi * f * t) + 0.05 * np.random.randn(n_samples),
            },
            index=time,
        )

    def test_pressure_only(self, multi_sensor_data):
        """Test with pressure sensors only (array)."""
        result = diwasp(
            multi_sensor_data,
            sensor_mapping={"p1": "pres", "p2": "pres", "p3": "pres"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            x={"p1": 0, "p2": 5, "p3": -5},
            y={"p1": 0, "p2": 5, "p3": 5},
            z=0.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars

    def test_velocity_only(self, multi_sensor_data):
        """Test with velocity sensors only."""
        result = diwasp(
            multi_sensor_data,
            sensor_mapping={"u": "velx", "v": "vely", "w": "velz"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            z=1.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars

    def test_single_pressure_sensor(self, multi_sensor_data):
        """Test with single pressure sensor."""
        result = diwasp(
            multi_sensor_data,
            sensor_mapping={"p1": "pres"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            z=0.5,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        # Single sensor produces non-directional spectrum (uniform direction)

    def test_puv_configuration(self, multi_sensor_data):
        """Test standard PUV configuration."""
        result = diwasp(
            multi_sensor_data,
            sensor_mapping={"p1": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            z={"p1": 0.5, "u": 1.0, "v": 1.0},
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert len(result.dir) > 1  # Should have directional resolution


class TestDiwaspWindowCalculation:
    """Tests for window calculation and timing."""

    def test_window_count(self):
        """Test that correct number of windows is generated."""
        np.random.seed(42)
        # Create exactly 1 hour of data at 2 Hz
        n_samples = 7200
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        df = pd.DataFrame(
            {
                "p": np.sin(0.1 * t),
                "u": np.cos(0.1 * t),
                "v": np.sin(0.1 * t),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,  # 10 minutes
            window_overlap=300,  # 5 minutes
            depth=20.0,
            verbose=0,
        )

        # Expected windows: (3600 - 600) / 300 + 1 = 11
        assert len(result.time) == 11

    def test_window_times_are_centered(self):
        """Test that output times are at window centers."""
        np.random.seed(42)
        n_samples = 3600  # 30 minutes at 2 Hz
        start_time = pd.Timestamp("2024-01-01 00:00:00")
        time = pd.date_range(start_time, periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        df = pd.DataFrame(
            {
                "p": np.sin(0.1 * t),
                "u": np.cos(0.1 * t),
                "v": np.sin(0.1 * t),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,  # 10 minutes
            window_overlap=0,
            depth=20.0,
            verbose=0,
        )

        # With 30 min data, 10 min windows, 0 overlap: 3 windows
        assert len(result.time) == 3

        # First window: 0-10 min, center at 5 min
        expected_first = start_time + pd.Timedelta(minutes=5)
        actual_first = pd.Timestamp(result.time.values[0])
        assert abs((actual_first - expected_first).total_seconds()) < 1


class TestDiwaspSamplingFrequency:
    """Tests for sampling frequency handling."""

    def test_inferred_frequency(self):
        """Test that sampling frequency is correctly inferred."""
        np.random.seed(42)
        n_samples = 1200
        # 4 Hz data
        time = pd.date_range("2024-01-01", periods=n_samples, freq="250ms")
        t = np.arange(n_samples) / 4.0

        df = pd.DataFrame(
            {
                "p": np.sin(0.1 * t),
                "u": np.cos(0.1 * t),
                "v": np.sin(0.1 * t),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=150,
            window_overlap=0,
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        # Nyquist frequency should be 2 Hz
        assert result.freq.values.max() <= 2.0

    def test_explicit_frequency(self):
        """Test with explicitly specified sampling frequency."""
        np.random.seed(42)
        n_samples = 1200
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        df = pd.DataFrame(
            {
                "p": np.sin(0.1 * t),
                "u": np.cos(0.1 * t),
                "v": np.sin(0.1 * t),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=300,
            window_overlap=0,
            depth=20.0,
            fs=2.0,  # Explicit 2 Hz
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)

    def test_frequency_mismatch_warning(self):
        """Test that frequency mismatch produces warning or uses explicit fs."""
        np.random.seed(42)
        n_samples = 1200
        # Create data with 2 Hz sampling
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        df = pd.DataFrame(
            {
                "p": np.sin(0.1 * t),
                "u": np.cos(0.1 * t),
                "v": np.sin(0.1 * t),
            },
            index=time,
        )

        # Specify different fs - should override inferred
        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=300,
            window_overlap=0,
            depth=20.0,
            fs=4.0,  # Different from actual 2 Hz
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)


class TestDiwaspSpectralOptions:
    """Tests for spectral analysis options."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for spectral option testing."""
        np.random.seed(42)
        n_samples = 2400
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        return pd.DataFrame(
            {
                "p": np.sin(0.1 * t) + 0.1 * np.random.randn(n_samples),
                "u": np.cos(0.1 * t) + 0.1 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(0.1 * t) + 0.1 * np.random.randn(n_samples),
            },
            index=time,
        )

    def test_custom_nfft(self, sample_data):
        """Test with custom NFFT length."""
        result = diwasp(
            sample_data,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            nfft=512,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)

    def test_smoothing_disabled(self, sample_data):
        """Test with smoothing disabled."""
        result = diwasp(
            sample_data,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            smooth=False,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)

    def test_custom_direction_array(self, sample_data):
        """Test with custom direction array."""
        dirs = np.linspace(0, 350, 36)  # 10 degree resolution

        result = diwasp(
            sample_data,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            dirs=dirs,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert len(result.dir) == 36


class TestDiwaspResampling:
    """Tests for automatic resampling functionality."""

    def test_non_uniform_sampling_resampled(self):
        """Test that non-uniform sampling is automatically resampled."""
        np.random.seed(42)
        n_samples = 1200

        # Create non-uniform time index (irregular gaps)
        base_time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        time_jitter = pd.to_timedelta(np.random.randint(-50, 50, n_samples), unit="ms")
        time = base_time + time_jitter

        t = np.arange(n_samples) / 2.0
        f = 0.1

        df = pd.DataFrame(
            {
                "p": np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "u": np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=300,
            window_overlap=0,
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars

    def test_explicit_fs_triggers_resampling(self):
        """Test that explicit fs different from data triggers resampling."""
        np.random.seed(42)
        n_samples = 1200

        # Create data at 2 Hz
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0
        f = 0.1

        df = pd.DataFrame(
            {
                "p": np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "u": np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
            },
            index=time,
        )

        # Request resampling to 1 Hz
        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=300,
            window_overlap=0,
            depth=20.0,
            fs=1.0,  # Resample to 1 Hz
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars

    def test_dataset_resampling(self):
        """Test resampling with xarray Dataset input."""
        np.random.seed(42)
        n_samples = 1200

        # Create non-uniform time index
        base_time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        time_jitter = pd.to_timedelta(np.random.randint(-30, 30, n_samples), unit="ms")
        time = base_time + time_jitter

        t = np.arange(n_samples) / 2.0
        f = 0.1

        ds = xr.Dataset(
            {
                "pres": (["time"], np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples)),
                "velx": (["time"], np.cos(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples)),
                "vely": (
                    ["time"],
                    0.5 * np.sin(2 * np.pi * f * t) + 0.1 * np.random.randn(n_samples),
                ),
            },
            coords={"time": time.values},
        )

        result = diwasp(
            ds,
            sensor_mapping={"pres": "pres", "velx": "velx", "vely": "vely"},
            window_length=300,
            window_overlap=0,
            depth=20.0,
            verbose=0,
        )

        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars


class TestDiwaspDepthAttenuation:
    """Tests for correct depth attenuation handling."""

    def test_pressure_sensor_depth_correction(self):
        """Test that pressure sensor depth attenuation is correctly compensated.

        A pressure sensor at depth sees attenuated wave signal. The transfer
        function correction should recover the surface wave height.
        """
        np.random.seed(42)

        # Parameters
        depth = 20.0  # Water depth
        z_sensor = 1.0  # Sensor 1m above seabed (19m below surface)
        f_wave = 0.1  # Wave frequency (10s period)
        amplitude = 1.0  # Surface wave amplitude

        # Calculate expected pressure attenuation
        # k from dispersion: omega^2 = g*k*tanh(k*d)
        from diwasp.utils import wavenumber

        omega = 2 * np.pi * f_wave
        k = wavenumber(np.array([omega]), depth)[0]

        # Pressure transfer function: cosh(k*z) / cosh(k*d)
        # where z is height above seabed
        pressure_tf = np.cosh(k * z_sensor) / np.cosh(k * depth)

        # Create synthetic pressure data (attenuated surface signal)
        n_samples = 3600  # 30 minutes at 2 Hz
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        # Pressure signal = attenuated surface elevation
        # (In real units, p = rho*g*eta*H(k,z), but we're working in normalized units)
        pressure_signal = amplitude * pressure_tf * np.sin(2 * np.pi * f_wave * t)
        pressure_signal += 0.02 * np.random.randn(n_samples)

        # Also add velocity sensors to help with directional estimation
        u_signal = amplitude * np.cos(2 * np.pi * f_wave * t) + 0.02 * np.random.randn(n_samples)
        v_signal = 0.5 * amplitude * np.sin(2 * np.pi * f_wave * t) + 0.02 * np.random.randn(n_samples)

        df = pd.DataFrame(
            {"p": pressure_signal, "u": u_signal, "v": v_signal},
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=0,
            depth=depth,
            z=z_sensor,  # Must specify correct sensor depth!
            verbose=0,
        )

        # The Hsig should be approximately 4*amplitude/sqrt(2) ≈ 2.83
        # for a sinusoidal wave (Hsig = 4*std = 4*amplitude/sqrt(2))
        # But our synthetic signal has amplitude 1.0, so expected Hsig ≈ 1.4
        # The key point is that depth attenuation is CORRECTED for

        mean_hsig = result["hsig"].mean().values
        # Should be order of unity, not severely underestimated due to depth
        assert mean_hsig > 0.5, f"Hsig {mean_hsig} too low - depth correction may not be working"
        assert mean_hsig < 5.0, f"Hsig {mean_hsig} unreasonably high"

    def test_shallow_vs_deep_sensor(self):
        """Test that shallow and deep sensors give consistent results.

        When both sensors are properly corrected for depth, they should
        estimate similar wave heights.
        """
        np.random.seed(42)

        depth = 20.0
        f_wave = 0.1
        amplitude = 1.0

        n_samples = 3600
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        # Create surface elevation signal
        eta = amplitude * np.sin(2 * np.pi * f_wave * t) + 0.02 * np.random.randn(n_samples)

        df = pd.DataFrame({"eta": eta}, index=time)

        # Analyze as surface elevation (z = depth, i.e., at surface)
        result_surface = diwasp(
            df,
            sensor_mapping={"eta": "elev"},
            window_length=600,
            window_overlap=0,
            depth=depth,
            z=depth,  # At surface
            verbose=0,
        )

        # The Hsig from surface elevation
        hsig_surface = result_surface["hsig"].mean().values

        # For comparison, analyze the same signal as pressure at mid-depth
        # (This is an artificial test - same signal, different assumed sensor type)
        result_pressure = diwasp(
            df,
            sensor_mapping={"eta": "pres"},
            window_length=600,
            window_overlap=0,
            depth=depth,
            z=10.0,  # Mid-depth
            verbose=0,
        )

        hsig_pressure = result_pressure["hsig"].mean().values

        # The pressure result should be different due to transfer function
        # This tests that the transfer function IS being applied
        # (If no transfer function was applied, both would be the same)
        assert hsig_pressure != hsig_surface, "Transfer function correction may not be applied"


class TestDiwaspIntegration:
    """Integration tests with realistic wave signals."""

    def test_known_wave_detection(self):
        """Test that known wave signal is detected at correct frequency."""
        np.random.seed(42)
        n_samples = 3600  # 30 minutes at 2 Hz
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        # Create wave at 0.1 Hz (10 second period)
        f_wave = 0.1
        amplitude = 1.0

        df = pd.DataFrame(
            {
                "p": amplitude * np.sin(2 * np.pi * f_wave * t) + 0.05 * np.random.randn(n_samples),
                "u": amplitude * np.cos(2 * np.pi * f_wave * t) + 0.05 * np.random.randn(n_samples),
                "v": 0.5 * amplitude * np.sin(2 * np.pi * f_wave * t)
                + 0.05 * np.random.randn(n_samples),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=0,
            depth=20.0,
            z=0.5,
            verbose=0,
        )

        # Peak frequency should be close to 0.1 Hz
        fp_mean = result["fp"].mean().values
        assert 0.08 < fp_mean < 0.12, f"Peak frequency {fp_mean} not near expected 0.1 Hz"

    def test_bimodal_sea_state(self):
        """Test detection of bimodal sea state."""
        np.random.seed(42)
        n_samples = 3600
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        # Swell at 0.08 Hz (12.5s) + wind sea at 0.15 Hz (6.7s)
        f_swell = 0.08
        f_wind = 0.15

        df = pd.DataFrame(
            {
                "p": (
                    np.sin(2 * np.pi * f_swell * t)
                    + 0.5 * np.sin(2 * np.pi * f_wind * t)
                    + 0.1 * np.random.randn(n_samples)
                ),
                "u": (
                    np.cos(2 * np.pi * f_swell * t)
                    + 0.5 * np.cos(2 * np.pi * f_wind * t)
                    + 0.1 * np.random.randn(n_samples)
                ),
                "v": 0.3 * np.random.randn(n_samples),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=0,
            depth=20.0,
            verbose=0,
        )

        # Should have positive Hsig
        assert np.all(result["hsig"].values > 0)

    def test_consistent_results_across_windows(self):
        """Test that stationary signal gives consistent results across windows."""
        np.random.seed(42)
        n_samples = 7200  # 1 hour at 2 Hz
        time = pd.date_range("2024-01-01", periods=n_samples, freq="500ms")
        t = np.arange(n_samples) / 2.0

        # Stationary wave signal
        f_wave = 0.1

        df = pd.DataFrame(
            {
                "p": np.sin(2 * np.pi * f_wave * t) + 0.05 * np.random.randn(n_samples),
                "u": np.cos(2 * np.pi * f_wave * t) + 0.05 * np.random.randn(n_samples),
                "v": 0.5 * np.sin(2 * np.pi * f_wave * t) + 0.05 * np.random.randn(n_samples),
            },
            index=time,
        )

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=20.0,
            verbose=0,
        )

        # Hsig should be fairly consistent across windows
        hsig_std = result["hsig"].std().values
        hsig_mean = result["hsig"].mean().values
        coefficient_of_variation = hsig_std / hsig_mean

        # CV should be < 0.3 for stationary signal
        assert coefficient_of_variation < 0.3, f"Hsig too variable: CV = {coefficient_of_variation}"
