"""Tests for the main dirspec function and core functionality."""

import numpy as np
import pytest

from diwasp import (
    EstimationMethod,
    EstimationParameters,
    InstrumentData,
    SensorType,
    SpectralInfo,
    SpectralMatrix,
    dirspec,
    make_wave_data,
    makespec,
)


class TestInstrumentData:
    """Tests for InstrumentData structure."""

    def test_creation(self):
        """Test basic InstrumentData creation."""
        data = np.random.randn(1024, 3)
        layout = np.array([[0, 10, 0], [0, 0, 10], [10, 10, 10]]).T
        datatypes = [SensorType.PRES, SensorType.VELX, SensorType.VELY]

        id = InstrumentData(
            data=data,
            layout=layout,
            datatypes=datatypes,
            depth=20.0,
            fs=2.0,
        )

        assert id.n_samples == 1024
        assert id.n_sensors == 3
        assert id.depth == 20.0
        assert id.fs == 2.0

    def test_validation_layout_mismatch(self):
        """Should raise error if layout columns don't match sensors."""
        data = np.random.randn(100, 3)
        layout = np.array([[0, 0], [0, 0], [0, 0]])  # Only 2 columns

        with pytest.raises(ValueError, match="Layout columns"):
            InstrumentData(
                data=data,
                layout=layout,
                datatypes=[SensorType.PRES, SensorType.VELX, SensorType.VELY],
                depth=10.0,
                fs=1.0,
            )

    def test_validation_datatypes_mismatch(self):
        """Should raise error if datatypes don't match sensors."""
        data = np.random.randn(100, 3)
        layout = np.array([[0, 10, 20], [0, 0, 0], [5, 5, 5]])

        with pytest.raises(ValueError, match="datatypes"):
            InstrumentData(
                data=data,
                layout=layout,
                datatypes=[SensorType.PRES, SensorType.VELX],  # Only 2
                depth=10.0,
                fs=1.0,
            )

    def test_validation_negative_depth(self):
        """Should raise error for negative depth."""
        data = np.random.randn(100, 3)
        layout = np.array([[0, 10, 20], [0, 0, 0], [5, 5, 5]])

        with pytest.raises(ValueError, match="depth"):
            InstrumentData(
                data=data,
                layout=layout,
                datatypes=[SensorType.PRES, SensorType.VELX, SensorType.VELY],
                depth=-10.0,
                fs=1.0,
            )


class TestSpectralMatrix:
    """Tests for SpectralMatrix structure."""

    def test_creation(self):
        """Test basic SpectralMatrix creation."""
        freqs = np.linspace(0.05, 0.5, 50)
        dirs = np.linspace(0, 360, 180, endpoint=False)
        S = np.random.rand(50, 180)

        sm = SpectralMatrix(freqs=freqs, dirs=dirs, S=S)

        assert sm.n_freqs == 50
        assert sm.n_dirs == 180
        assert sm.funit == "hz"
        assert sm.dunit == "cart"

    def test_validation_shape_mismatch(self):
        """Should raise error if S shape doesn't match grids."""
        freqs = np.linspace(0.05, 0.5, 50)
        dirs = np.linspace(0, 360, 180, endpoint=False)
        S = np.random.rand(40, 180)  # Wrong freq dimension

        with pytest.raises(ValueError, match="shape"):
            SpectralMatrix(freqs=freqs, dirs=dirs, S=S)

    def test_to_xarray(self):
        """Test conversion to xarray Dataset."""
        freqs = np.linspace(0.05, 0.5, 10)
        dirs = np.linspace(0, 360, 36, endpoint=False)
        S = np.random.rand(10, 36)

        sm = SpectralMatrix(freqs=freqs, dirs=dirs, S=S)
        ds = sm.to_xarray()

        assert "efth" in ds.data_vars
        assert "freq" in ds.coords
        assert "dir" in ds.coords
        np.testing.assert_array_equal(ds["efth"].values, S)


class TestEstimationParameters:
    """Tests for EstimationParameters structure."""

    def test_defaults(self):
        """Test default parameters."""
        ep = EstimationParameters()

        assert ep.method == EstimationMethod.EMLM
        assert ep.dres == 180
        assert ep.iter == 100
        assert ep.smooth is True

    def test_validation_dres(self):
        """Should raise error for invalid dres."""
        with pytest.raises(ValueError, match="dres"):
            EstimationParameters(dres=2)

    def test_validation_iter(self):
        """Should raise error for invalid iter."""
        with pytest.raises(ValueError, match="iter"):
            EstimationParameters(iter=0)


class TestMakespec:
    """Tests for synthetic spectrum generation."""

    def test_unimodal_spectrum(self):
        """Test generation of unimodal spectrum."""
        spectrum = makespec(
            freq_range=(0.05, 0.1, 0.3),
            theta=45.0,
            spread=50.0,
            hsig=2.0,
        )

        assert isinstance(spectrum, SpectralMatrix)
        assert spectrum.n_freqs > 0
        assert spectrum.n_dirs > 0

        # Check Hsig is approximately correct
        from diwasp import hsig

        hs = hsig(spectrum.S, spectrum.freqs, spectrum.dirs)
        np.testing.assert_allclose(hs, 2.0, rtol=0.1)

    def test_bimodal_spectrum(self):
        """Test generation of bimodal spectrum."""
        spectrum = makespec(
            freq_range=(0.04, 0.08, 0.3),
            theta=[270.0, 180.0],
            spread=[25.0, 75.0],
            weights=[0.3, 0.7],
            hsig=3.0,
        )

        assert isinstance(spectrum, SpectralMatrix)

        # Should have energy in both directions
        dir_spectrum = np.sum(spectrum.S, axis=0)
        assert np.max(dir_spectrum) > 0

    def test_custom_resolution(self):
        """Test with custom frequency and direction resolution."""
        spectrum = makespec(
            freq_range=(0.05, 0.1, 0.3),
            theta=90.0,
            spread=50.0,
            n_freqs=100,
            n_dirs=360,
        )

        assert spectrum.n_freqs == 100
        assert spectrum.n_dirs == 360


class TestMakeWaveData:
    """Tests for synthetic wave data generation."""

    def test_generates_data(self):
        """Test that make_wave_data generates data with correct shape."""
        spectrum = makespec(
            freq_range=(0.05, 0.1, 0.3),
            theta=45.0,
            spread=50.0,
            hsig=1.0,
        )

        # Create instrument configuration
        layout = np.array([[0, 10, 0], [0, 0, 10], [10, 10, 10]]).T
        datatypes = [SensorType.PRES, SensorType.VELX, SensorType.VELY]

        id = InstrumentData(
            data=np.zeros((100, 3)),  # Placeholder
            layout=layout,
            datatypes=datatypes,
            depth=20.0,
            fs=2.0,
        )

        data = make_wave_data(spectrum, id, n_samples=1024, seed=42)

        assert data.shape == (1024, 3)
        assert not np.allclose(data, 0)  # Should have signal

    def test_reproducible_with_seed(self):
        """Test that results are reproducible with same seed."""
        spectrum = makespec(
            freq_range=(0.05, 0.1, 0.3),
            theta=45.0,
            spread=50.0,
        )

        layout = np.array([[0, 10], [0, 0], [10, 10]]).T
        datatypes = [SensorType.PRES, SensorType.VELX]

        id = InstrumentData(
            data=np.zeros((100, 2)),
            layout=layout,
            datatypes=datatypes,
            depth=20.0,
            fs=2.0,
        )

        data1 = make_wave_data(spectrum, id, n_samples=256, seed=123)
        data2 = make_wave_data(spectrum, id, n_samples=256, seed=123)

        np.testing.assert_array_equal(data1, data2)


class TestDirspec:
    """Tests for the main dirspec function."""

    @pytest.fixture
    def simple_instrument_data(self):
        """Create simple instrument data for testing."""
        np.random.seed(42)

        # Generate synthetic data
        n_samples = 2048
        fs = 2.0
        t = np.arange(n_samples) / fs

        # Create simple wave signal
        f = 0.1  # Hz
        data = np.zeros((n_samples, 3))
        data[:, 0] = np.sin(2 * np.pi * f * t)  # Pressure-like
        data[:, 1] = np.cos(2 * np.pi * f * t)  # VelX-like
        data[:, 2] = 0.5 * np.sin(2 * np.pi * f * t)  # VelY-like

        # Add noise
        data += 0.1 * np.random.randn(*data.shape)

        layout = np.array([[0, 10, 0], [0, 0, 10], [10, 10, 10]]).T
        datatypes = [SensorType.PRES, SensorType.VELX, SensorType.VELY]

        return InstrumentData(
            data=data,
            layout=layout,
            datatypes=datatypes,
            depth=20.0,
            fs=fs,
        )

    def test_basic_analysis(self, simple_instrument_data):
        """Test basic dirspec analysis runs without error."""
        spectrum, info = dirspec(simple_instrument_data, verbose=0)

        assert isinstance(spectrum, SpectralMatrix)
        assert isinstance(info, SpectralInfo)
        assert spectrum.S.shape[0] > 0
        assert spectrum.S.shape[1] > 0

    def test_all_methods(self, simple_instrument_data):
        """Test all estimation methods run without error."""
        for method in EstimationMethod:
            params = EstimationParameters(method=method, iter=10)
            spectrum, info = dirspec(simple_instrument_data, params, verbose=0)

            assert isinstance(spectrum, SpectralMatrix)
            assert info.hsig >= 0
            assert info.tp > 0

    def test_custom_grids(self, simple_instrument_data):
        """Test with custom frequency and direction grids."""
        freqs = np.linspace(0.05, 0.4, 30)
        dirs = np.linspace(0, 360, 72, endpoint=False)

        spectrum, info = dirspec(
            simple_instrument_data,
            freqs=freqs,
            dirs=dirs,
            verbose=0,
        )

        assert spectrum.n_freqs == 30
        assert spectrum.n_dirs == 72

    def test_no_smoothing(self, simple_instrument_data):
        """Test with smoothing disabled."""
        params = EstimationParameters(smooth=False)
        spectrum, info = dirspec(simple_instrument_data, params, verbose=0)

        assert isinstance(spectrum, SpectralMatrix)

    def test_validation_too_few_sensors(self):
        """Should raise error with only 1 sensor."""
        data = np.random.randn(1024, 1)
        layout = np.array([[0], [0], [10]])

        id = InstrumentData(
            data=data,
            layout=layout,
            datatypes=[SensorType.PRES],
            depth=20.0,
            fs=2.0,
        )

        with pytest.raises(ValueError, match="2 sensors"):
            dirspec(id, verbose=0)

    def test_validation_too_few_samples(self):
        """Should raise error with too few samples."""
        data = np.random.randn(32, 3)
        layout = np.array([[0, 10, 20], [0, 0, 0], [10, 10, 10]])

        id = InstrumentData(
            data=data,
            layout=layout,
            datatypes=[SensorType.PRES, SensorType.VELX, SensorType.VELY],
            depth=20.0,
            fs=2.0,
        )

        with pytest.raises(ValueError, match="64 samples"):
            dirspec(id, verbose=0)


class TestIntegration:
    """Integration tests using synthetic data."""

    def test_roundtrip_synthetic(self):
        """Test that we can recover spectrum from synthetic data."""
        # Create known spectrum
        target_hsig = 1.5
        target_dir = 45.0

        spectrum_in = makespec(
            freq_range=(0.05, 0.1, 0.25),
            theta=target_dir,
            spread=50.0,
            hsig=target_hsig,
            depth=20.0,
        )

        # Generate synthetic sensor data
        layout = np.array([[0, 5, 0], [0, 0, 5], [15, 15, 15]]).T
        datatypes = [SensorType.PRES, SensorType.VELX, SensorType.VELY]

        id_template = InstrumentData(
            data=np.zeros((100, 3)),
            layout=layout,
            datatypes=datatypes,
            depth=20.0,
            fs=2.0,
        )

        data = make_wave_data(
            spectrum_in,
            id_template,
            n_samples=4096,
            noise_level=0.01,
            seed=42,
        )

        # Create instrument data with generated signals
        id = InstrumentData(
            data=data,
            layout=layout,
            datatypes=datatypes,
            depth=20.0,
            fs=2.0,
        )

        # Estimate spectrum
        params = EstimationParameters(method=EstimationMethod.IMLM, iter=50)
        spectrum_out, info = dirspec(id, params, verbose=0)

        # Check that estimated Hsig is reasonable
        # (may not be exact due to estimation uncertainty)
        assert 0.5 * target_hsig < info.hsig < 2.0 * target_hsig

        # Check that peak direction is in correct quadrant
        dir_diff = abs(info.dp - target_dir)
        dir_diff = min(dir_diff, 360 - dir_diff)  # Handle wrap
        assert dir_diff < 90  # Within 90 degrees
