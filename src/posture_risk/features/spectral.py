"""
spectral.py
-----------
Features en el dominio frecuencial para señales IMU (y EMG).

Features implementados:
  MDF      — Median Frequency (indicador de fatiga muscular)
  MNF      — Mean Frequency (centroide espectral)
  BAND_LOW — Potencia en banda baja  (0.5–5 Hz para IMU)
  BAND_MID — Potencia en banda media (5–15 Hz para IMU)
  BAND_HIG — Potencia en banda alta  (15–40 Hz para IMU)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import welch


def _psd(window: NDArray, fs: float):
    """
    Calcula la PSD (Power Spectral Density) con el método de Welch.
    Retorna (freqs, psd) donde psd.shape = (n_freqs, n_channels).
    """
    n_samples, n_channels = window.shape
    nperseg = min(n_samples, 16)  # segmento para Welch, adaptado a ventanas cortas
    freqs, psd = welch(window.T, fs=fs, nperseg=nperseg, axis=-1)
    return freqs, psd.T  # shape: (n_freqs, n_channels)


def median_frequency(window: NDArray, fs: float) -> NDArray:
    """
    Frecuencia mediana: la frecuencia que divide el espectro en dos mitades de igual potencia.
    Cae con la fatiga muscular (reclutamiento de fibras lentas).
    """
    freqs, psd = _psd(window, fs)
    cumulative = np.cumsum(psd, axis=0)
    total = cumulative[-1]
    mdf = np.zeros(window.shape[1])
    for ch in range(window.shape[1]):
        idx = np.searchsorted(cumulative[:, ch], total[ch] / 2.0)
        mdf[ch] = freqs[min(idx, len(freqs) - 1)]
    return mdf


def mean_frequency(window: NDArray, fs: float) -> NDArray:
    """
    Frecuencia media (centroide espectral): suma(f × PSD) / suma(PSD).
    """
    freqs, psd = _psd(window, fs)
    total_power = np.sum(psd, axis=0)
    total_power = np.where(total_power == 0, 1e-10, total_power)
    return np.sum(freqs[:, None] * psd, axis=0) / total_power


def band_power(window: NDArray, fs: float, fmin: float, fmax: float) -> NDArray:
    """
    Potencia total en una banda de frecuencia [fmin, fmax] Hz.
    """
    freqs, psd = _psd(window, fs)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    if not band_mask.any():
        return np.zeros(window.shape[1])
    df = freqs[1] - freqs[0]  # resolución frecuencial
    return np.trapz(psd[band_mask], dx=df, axis=0)


def extract_spectral_features(
    window: NDArray,
    fs: float,
    bands: dict = None,
) -> NDArray:
    """
    Extrae todos los features espectrales de una ventana multicanal.

    Parámetros
    ----------
    window : NDArray, shape (n_samples, n_channels)
    fs     : float — frecuencia de muestreo en Hz
    bands  : dict  — {"low": (0.5, 5), "mid": (5, 15), "high": (15, 40)}

    Retorna
    -------
    NDArray, shape (n_features,) donde n_features = 5 × n_channels
    Orden: [mdf, mnf, band_low, band_mid, band_high] por canal
    """
    if bands is None:
        bands = {"low": (0.5, 5.0), "mid": (5.0, 15.0), "high": (15.0, 40.0)}

    return np.concatenate([
        median_frequency(window, fs),
        mean_frequency(window, fs),
        band_power(window, fs, *bands["low"]),
        band_power(window, fs, *bands["mid"]),
        band_power(window, fs, *bands["high"]),
    ])
