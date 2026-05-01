"""Tests unitarios para los módulos de features."""
import sys, numpy as np, pytest
sys.path.insert(0, "src")
from posture_risk.features.temporal import extract_temporal_features
from posture_risk.features.spectral import extract_spectral_features

@pytest.fixture
def random_window():
    np.random.seed(42)
    return np.random.randn(20, 9).astype(np.float32)  # 20 muestras, 9 canales

def test_temporal_features_shape(random_window):
    feats = extract_temporal_features(random_window)
    assert feats.shape == (6 * 9,), f"Esperado (54,), obtenido {feats.shape}"

def test_spectral_features_shape(random_window):
    feats = extract_spectral_features(random_window, fs=100.0)
    assert feats.shape == (5 * 9,), f"Esperado (45,), obtenido {feats.shape}"

def test_no_nan_in_features(random_window):
    t = extract_temporal_features(random_window)
    s = extract_spectral_features(random_window, fs=100.0)
    assert not np.isnan(t).any(), "NaN en features temporales"
    assert not np.isnan(s).any(), "NaN en features espectrales"
