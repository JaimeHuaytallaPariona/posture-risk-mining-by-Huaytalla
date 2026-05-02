"""
spectral.py
-----------
Features en el dominio frecuencial para señales IMU (y EMG).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch
from scipy.integrate import trapezoid


def _psd(window: NDArray, fs: float):
    n_samples, n_channels = window.shape
    nperseg = min(n_samples, 16)
    freqs, psd = welch(window.T, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd.T


def median_frequency(window: NDArray, fs: float) -> NDArray:
    freqs, psd = _psd(window, fs)
    cumulative = np.cumsum(psd, axis=0)
    total = cumulative[-1]
    mdf = np.zeros(window.shape[1])
    for ch in range(window.shape[1]):
        idx = np.searchsorted(cumulative[:, ch], total[ch] / 2.0)
        mdf[ch] = freqs[min(idx, len(freqs) - 1)]
    return mdf


def mean_frequency(window: NDArray, fs: float) -> NDArray:
    freqs, psd = _psd(window, fs)
    total_power = np.sum(psd, axis=0)
    total_power = np.where(total_power == 0, 1e-10, total_power)
    return np.sum(freqs[:, None] * psd, axis=0) / total_power


def band_power(window: NDArray, fs: float, fmin: float, fmax: float) -> NDArray:
    freqs, psd = _psd(window, fs)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not band_mask.any():
        return np.zeros(window.shape[1])
    df = freqs[1] - freqs[0]
    # scipy.integrate.trapezoid funciona en todas las versiones de NumPy
    return trapezoid(psd[band_mask], dx=df, axis=0)


def extract_spectral_features(
    window: NDArray,
    fs: float,
    bands: dict = None,
) -> NDArray:
    if bands is None:
        bands = {"low": (0.5, 5.0), "mid": (5.0, 15.0), "high": (15.0, 40.0)}

    return np.concatenate([
        median_frequency(window, fs),
        mean_frequency(window, fs),
        band_power(window, fs, *bands["low"]),
        band_power(window, fs, *bands["mid"]),
        band_power(window, fs, *bands["high"]),
    ])
