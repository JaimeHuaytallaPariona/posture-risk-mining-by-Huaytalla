"""
temporal.py
-----------
Extracción de features en el dominio temporal sobre ventanas de señal IMU/EMG.

Features implementados:
  RMS  — root mean square (intensidad de activación)
  MAV  — mean absolute value (amplitud media)
  WL   — waveform length (complejidad/actividad de la señal)
  ZC   — zero crossings (frecuencia de oscilación, barato computacionalmente)
  SSC  — slope sign changes (cambios de pendiente)
  VAR  — varianza (dispersión de la señal)
"""

import numpy as np
from numpy.typing import NDArray


def rms(window: NDArray) -> NDArray:
    """Root Mean Square por canal. Shape: (n_samples, n_channels) → (n_channels,)"""
    return np.sqrt(np.mean(window ** 2, axis=0))


def mav(window: NDArray) -> NDArray:
    """Mean Absolute Value por canal."""
    return np.mean(np.abs(window), axis=0)


def wl(window: NDArray) -> NDArray:
    """Waveform Length: suma de diferencias absolutas consecutivas."""
    return np.sum(np.abs(np.diff(window, axis=0)), axis=0)


def zero_crossings(window: NDArray, threshold: float = 1e-6) -> NDArray:
    """
    Zero Crossings por canal.
    threshold: valor mínimo para considerar un cruce real (evita ruido).
    """
    signs = np.sign(window)
    # Reemplaza ceros con el signo anterior para evitar falsos cruces
    signs[signs == 0] = 1
    crossings = np.diff(signs, axis=0)
    return np.sum(np.abs(crossings) > 0, axis=0).astype(float)


def slope_sign_changes(window: NDArray, threshold: float = 1e-6) -> NDArray:
    """Slope Sign Changes: cambios de signo en la primera diferencia."""
    diff1 = np.diff(window, axis=0)
    diff2 = np.diff(diff1, axis=0)
    signs = np.sign(diff1[:-1]) != np.sign(diff1[1:])
    magnitude = np.abs(diff2) > threshold
    return np.sum(signs & magnitude, axis=0).astype(float)


def variance(window: NDArray) -> NDArray:
    """Varianza muestral por canal."""
    return np.var(window, axis=0, ddof=1)


def extract_temporal_features(window: NDArray) -> NDArray:
    """
    Extrae todos los features temporales de una ventana multicanal.

    Parámetros
    ----------
    window : NDArray, shape (n_samples, n_channels)

    Retorna
    -------
    NDArray, shape (n_features,) donde n_features = 6 × n_channels
    Orden: [rms_ch0, ..., rms_chN, mav_ch0, ..., var_chN]
    """
    return np.concatenate([
        rms(window),
        mav(window),
        wl(window),
        zero_crossings(window),
        slope_sign_changes(window),
        variance(window),
    ])
