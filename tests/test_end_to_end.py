"""End-to-end tests for the diwasp wrapper function.

These tests use synthetic data generated with makespec and make_wave_data
to verify the complete analysis pipeline works correctly with varying
wave parameters across multiple analysis windows.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from diwasp import diwasp, make_wave_data, makespec


class TestEndToEndDataFrame:
    """End-to-end tests using pandas DataFrame input."""

    def test_steady_sea_state_puv(self):
        """Test analysis of steady sea state with PUV sensors."""
        # Create steady sea state
        spec = makespec(
            freq_range=(0.05, 0.1, 0.5),  # (low, peak, high) in Hz
            theta=45.0,  # Peak direction (from NE)
            spread=75.0,  # Directional spreading
            hsig=2.0,  # Significant wave height
            depth=20.0,
            n_freqs=50,
            n_dirs=180,
        )

        # Generate 1 hour of data at 2 Hz
        fs = 2.0
        duration = 3600  # seconds
        n_samples = int(duration * fs)

        # Sensor layout: PUV at same location
        layout = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).T  # x, y, z
        layout[2, :] = [0.5, 1.0, 1.0]  # z positions

        data = make_wave_data(
            spec=spec,
            layout=layout,
            datatypes=["pres", "velx", "vely"],
            depth=20.0,
            fs=fs,
            duration=duration,
        )

        # Create DataFrame
        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        df = pd.DataFrame(
            {"pressure": data[:, 0], "u_vel": data[:, 1], "v_vel": data[:, 2]},
            index=time,
        )

        # Run analysis with 30-minute windows, 15-minute overlap
        result = diwasp(
            df,
            sensor_mapping={"pressure": "pres", "u_vel": "velx", "v_vel": "vely"},
            window_length=1800,
            window_overlap=900,
            depth=20.0,
            z={"pressure": 0.5, "u_vel": 1.0, "v_vel": 1.0},
            method="imlm",
            verbose=0,
        )

        # Verify output structure
        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars
        assert "hsig" in result.data_vars
        assert "tp" in result.data_vars
        assert "dp" in result.data_vars

        # Verify dimensions
        assert "time" in result.dims
        assert "freq" in result.dims
        assert "dir" in result.dims

        # Should have multiple time windows
        assert len(result.time) > 1

        # Check wave parameters are reasonable
        hsig_mean = result.hsig.mean().values
        tp_mean = result.tp.mean().values
        dp_mean = result.dp.mean().values

        # Hsig should be positive
        assert hsig_mean > 0
        # Peak period should be close to 10s
        assert 8 < tp_mean < 12
        # Peak direction should be close to 45 degrees
        assert 30 < dp_mean < 60

    def test_varying_sea_state(self):
        """Test analysis with slowly varying wave parameters."""
        # Generate data with varying parameters
        fs = 2.0
        duration = 7200  # 2 hours
        n_samples = int(duration * fs)

        # Create time-varying spectra (simulate changing conditions)
        # We'll create 4 different sea states and concatenate the data
        all_data = []
        segment_duration = duration / 4

        for i, (hsig_target, tp_target, dir_target) in enumerate(
            [(1.5, 8, 30), (2.0, 10, 45), (2.5, 12, 60), (2.0, 10, 45)]
        ):
            fp_target = 1.0 / tp_target
            spec = makespec(
                freq_range=(0.05, fp_target, 0.5),
                theta=dir_target,
                spread=75.0,
                hsig=hsig_target,
                depth=20.0,
                n_freqs=50,
                n_dirs=180,
            )

            layout = np.array([[0, 0, 0.5], [0, 0, 1.0], [0, 0, 1.0]]).T

            segment_data = make_wave_data(
                spec=spec,
                layout=layout,
                datatypes=["pres", "velx", "vely"],
                depth=20.0,
                fs=fs,
                duration=segment_duration,
            )
            all_data.append(segment_data)

        # Concatenate all segments
        data = np.vstack(all_data)

        # Create DataFrame
        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        df = pd.DataFrame({"p": data[:, 0], "u": data[:, 1], "v": data[:, 2]}, index=time)

        # Run analysis
        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=1800,
            window_overlap=900,
            depth=20.0,
            z={"p": 0.5, "u": 1.0, "v": 1.0},
            method="imlm",
            verbose=0,
        )

        # Verify we captured the variation
        assert len(result.time) >= 4

        # Check that Hsig varies
        hsig_std = result.hsig.std().values
        assert hsig_std > 0.1  # Should see variation

        # Check that peak period varies
        tp_std = result.tp.std().values
        assert tp_std > 0.5  # Should see variation

    def test_pressure_array(self):
        """Test analysis with pressure gauge array."""
        # Create spectrum
        spec = makespec(
            freq_range=(0.05, 0.1, 0.5),
            theta=90.0,  # From East
            spread=75.0,
            hsig=2.0,
            depth=15.0,
            n_freqs=50,
            n_dirs=180,
        )

        # Triangular array of pressure sensors
        layout = np.array([[0, 5, -5], [0, 5, 5], [0, 0, 0]]).T  # x, y, z

        fs = 2.0
        duration = 3600

        data = make_wave_data(
            spec=spec,
            layout=layout,
            datatypes=["pres", "pres", "pres"],
            depth=15.0,
            fs=fs,
            duration=duration,
        )

        # Create DataFrame
        n_samples = int(duration * fs)
        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        df = pd.DataFrame({"p1": data[:, 0], "p2": data[:, 1], "p3": data[:, 2]}, index=time)

        # Run analysis
        result = diwasp(
            df,
            sensor_mapping={"p1": "pres", "p2": "pres", "p3": "pres"},
            window_length=1800,
            window_overlap=900,
            depth=15.0,
            x={"p1": 0, "p2": 5, "p3": -5},
            y={"p1": 0, "p2": 5, "p3": 5},
            z=0.0,
            method="emep",
            verbose=0,
        )

        # Verify output
        assert isinstance(result, xr.Dataset)
        assert len(result.time) > 1

        # Peak direction should be close to 90 degrees (East)
        dp_mean = result.dp.mean().values
        assert 70 < dp_mean < 110


class TestEndToEndDataset:
    """End-to-end tests using xarray Dataset input."""

    def test_dataset_with_coordinates(self):
        """Test analysis with Dataset containing sensor position coordinates."""
        # Create spectrum
        spec = makespec(
            freq_range=(0.05, 0.125, 0.5),  # 8 second waves
            theta=180.0,  # From South
            spread=75.0,
            hsig=2.0,
            depth=20.0,
            n_freqs=50,
            n_dirs=180,
        )

        # Generate data
        layout = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).T
        layout[2, :] = [0.5, 1.0, 1.0]

        fs = 2.0
        duration = 3600
        n_samples = int(duration * fs)

        data = make_wave_data(
            spec=spec,
            layout=layout,
            datatypes=["pres", "velx", "vely"],
            depth=20.0,
            fs=fs,
            duration=duration,
        )

        # Create Dataset with position attributes
        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        ds = xr.Dataset(
            {
                "pres": (["time"], data[:, 0], {"z": 0.5}),
                "velx": (["time"], data[:, 1], {"z": 1.0}),
                "vely": (["time"], data[:, 2], {"z": 1.0}),
            },
            coords={"time": time.values},
        )

        # Run analysis (positions read from attributes)
        result = diwasp(
            ds,
            sensor_mapping={"pres": "pres", "velx": "velx", "vely": "vely"},
            window_length=1800,
            window_overlap=900,
            depth=20.0,
            method="imlm",
            verbose=0,
        )

        # Verify output
        assert isinstance(result, xr.Dataset)
        assert len(result.time) > 1

        # Peak period should be close to 8s
        tp_mean = result.tp.mean().values
        assert 7 < tp_mean < 9

        # Peak direction should be close to 180 degrees (South)
        dp_mean = result.dp.mean().values
        assert 160 < dp_mean < 200


class TestEndToEndMethods:
    """Test all estimation methods with end-to-end workflow."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic wave data for testing."""
        spec = makespec(
            freqs=np.linspace(0.05, 0.5, 50),
            dirs=np.linspace(0, 360, 181, endpoint=False),
            spreading=75,
            frequency_hz=0.1,
            direction_deg=45,
            gamma=3.3,
        )

        layout = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]).T
        layout[2, :] = [0.5, 1.0, 1.0]

        fs = 2.0
        duration = 1800  # 30 minutes
        n_samples = int(duration * fs)

        data = make_wave_data(
            spec=spec,
            layout=layout,
            datatypes=["pres", "velx", "vely"],
            depth=20.0,
            fs=fs,
            duration=duration,
        )

        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        df = pd.DataFrame({"p": data[:, 0], "u": data[:, 1], "v": data[:, 2]}, index=time)

        return df

    @pytest.mark.parametrize("method", ["dftm", "emlm", "imlm", "emep", "bdm"])
    def test_all_methods(self, synthetic_data, method):
        """Test that all estimation methods work end-to-end."""
        result = diwasp(
            synthetic_data,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=900,  # 15 minutes
            window_overlap=0,
            depth=20.0,
            z={"p": 0.5, "u": 1.0, "v": 1.0},
            method=method,
            verbose=0,
        )

        # Verify output structure
        assert isinstance(result, xr.Dataset)
        assert "efth" in result.data_vars
        assert "hsig" in result.data_vars

        # Verify reasonable wave parameters
        hsig = result.hsig.values[0]
        tp = result.tp.values[0]
        dp = result.dp.values[0]

        assert hsig > 0
        assert 5 < tp < 15
        assert 0 <= dp < 360

        # Verify no NaN values in key outputs
        assert not np.any(np.isnan(result.hsig.values))
        assert not np.any(np.isnan(result.tp.values))
        assert not np.any(np.isnan(result.dp.values))


class TestEndToEndEdgeCases:
    """Test edge cases in end-to-end workflow."""

    def test_short_duration_single_window(self):
        """Test with data length equal to window length (single window)."""
        spec = makespec(
            freqs=np.linspace(0.05, 0.5, 50),
            dirs=np.linspace(0, 360, 181, endpoint=False),
            spreading=75,
            frequency_hz=0.1,
            direction_deg=45,
            gamma=3.3,
        )

        layout = np.array([[0, 0, 0.5], [0, 0, 1.0], [0, 0, 1.0]]).T

        fs = 2.0
        duration = 600  # 10 minutes
        n_samples = int(duration * fs)

        data = make_wave_data(
            spec=spec,
            layout=layout,
            datatypes=["pres", "velx", "vely"],
            depth=20.0,
            fs=fs,
            duration=duration,
        )

        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        df = pd.DataFrame({"p": data[:, 0], "u": data[:, 1], "v": data[:, 2]}, index=time)

        # Single window analysis
        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=0,
            depth=20.0,
            z=1.0,
            verbose=0,
        )

        # Should have exactly 1 window
        assert len(result.time) == 1
        assert isinstance(result, xr.Dataset)

    def test_high_frequency_waves(self):
        """Test with high frequency (short period) waves."""
        spec = makespec(
            freq_range=(0.1, 0.5, 1.0),  # 2 second waves
            theta=270.0,  # From West
            spread=75.0,
            hsig=1.5,
            depth=10.0,
            n_freqs=50,
            n_dirs=180,
        )

        layout = np.array([[0, 0, 0.5], [0, 0, 1.0], [0, 0, 1.0]]).T

        fs = 4.0  # Higher sampling rate for short waves
        duration = 1200
        n_samples = int(duration * fs)

        data = make_wave_data(
            spec=spec,
            layout=layout,
            datatypes=["pres", "velx", "vely"],
            depth=10.0,
            fs=fs,
            duration=duration,
        )

        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        df = pd.DataFrame({"p": data[:, 0], "u": data[:, 1], "v": data[:, 2]}, index=time)

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=600,
            window_overlap=300,
            depth=10.0,
            z=1.0,
            verbose=0,
        )

        # Peak period should be close to 2s
        tp_mean = result.tp.mean().values
        assert 1.5 < tp_mean < 2.5

    def test_bimodal_spectrum(self):
        """Test with bimodal spectrum (swell + wind sea)."""
        # Create bimodal spectrum directly
        combined_spec = makespec(
            freq_range=(0.05, 0.08, 0.5),
            theta=[180.0, 270.0],  # Swell from South, wind sea from West
            spread=[50.0, 100.0],  # Narrower swell, broader wind sea
            weights=[0.7, 0.3],  # Dominant swell
            hsig=2.5,
            depth=20.0,
            n_freqs=50,
            n_dirs=180,
        )

        layout = np.array([[0, 0, 0.5], [0, 0, 1.0], [0, 0, 1.0]]).T

        fs = 2.0
        duration = 1800
        n_samples = int(duration * fs)

        data = make_wave_data(
            spec=combined_spec,
            layout=layout,
            datatypes=["pres", "velx", "vely"],
            depth=20.0,
            fs=fs,
            duration=duration,
        )

        time = pd.date_range("2024-01-01", periods=n_samples, freq=f"{int(1000/fs)}ms")
        df = pd.DataFrame({"p": data[:, 0], "u": data[:, 1], "v": data[:, 2]}, index=time)

        result = diwasp(
            df,
            sensor_mapping={"p": "pres", "u": "velx", "v": "vely"},
            window_length=900,
            window_overlap=0,
            depth=20.0,
            z=1.0,
            method="emep",  # EMEP is good for complex spectra
            verbose=0,
        )

        # Should detect the dominant (swell) peak
        tp_mean = result.tp.mean().values
        assert 10 < tp_mean < 15  # Should be closer to swell period

        # Verify output is valid
        assert not np.any(np.isnan(result.hsig.values))
        assert not np.any(np.isnan(result.efth.values))
