"""
biomechanical.py
----------------
Features de dominio biomecánico para análisis postural.

Estas features se calculan SOLO a partir de la señal cruda dentro de cada ventana.
Son funciones puras: misma entrada → misma salida, sin estado oculto, sin riesgo
de leakage. Cualquier transformación que dependa de estadísticas del dataset
(escalado, winsorización) se aplica DESPUÉS dentro de sklearn.Pipeline.

Features implementadas:
  - Magnitud del vector aceleración (intensidad de movimiento)
  - Magnitud del vector angular (velocidad de rotación)
  - Ángulo de inclinación (tilt angle, proxy de ángulos articulares)
  - Jerk (derivada de aceleración: suavidad del movimiento)
  - Signal Magnitude Area (SMA, indicador estándar de actividad)
  - Ratios inter-segmento (coordinación entre extremidades)

Referencias:
  Karantonis et al. (2006) "Implementation of a Real-Time Human Movement
    Classifier Using a Triaxial Accelerometer"
  Hogan & Sternad (2009) "Sensitivity of Smoothness Measures to Movement
    Duration, Amplitude, and Arrests"
"""

import numpy as np
from numpy.typing import NDArray


# ─── Funciones de transformación de señal ─────────────────────────────────────

def vector_magnitude(window: NDArray, axes: list) -> NDArray:
    """
    Magnitud euclidiana del vector 3D en cada muestra de la ventana.

    Parámetros
    ----------
    window : NDArray, shape (n_samples, n_channels)
    axes   : list de 3 índices de columnas correspondientes a x, y, z

    Retorna
    -------
    NDArray, shape (n_samples,) — serie temporal de magnitud
    """
    return np.sqrt(np.sum(window[:, axes] ** 2, axis=1))


def tilt_angle(window: NDArray, axes: list) -> NDArray:
    """
    Ángulo de inclinación del segmento respecto a la vertical.

    Calcula el ángulo entre el eje principal del acelerómetro y la dirección
    de la gravedad. Es un proxy directo de los ángulos articulares usados en RULA.

    theta(t) = atan2(|a_horiz(t)|, a_vert(t))

    donde a_vert = primer eje (asumimos eje principal del sensor)
    y |a_horiz| = magnitud de los otros dos ejes

    Parámetros
    ----------
    window : NDArray, shape (n_samples, n_channels)
    axes   : list de 3 índices, el primero es el eje "vertical" del sensor

    Retorna
    -------
    NDArray, shape (n_samples,) — ángulo en radianes
    """
    a_vert  = window[:, axes[0]]
    a_horiz = np.sqrt(window[:, axes[1]] ** 2 + window[:, axes[2]] ** 2)
    # atan2(horizontal, vertical) da el ángulo respecto a la vertical
    return np.arctan2(a_horiz, a_vert)


def jerk(window: NDArray, axes: list, fs: float) -> NDArray:
    """
    Magnitud del jerk: derivada temporal del vector aceleración.

    Movimientos suaves tienen jerk bajo. Posturas forzadas, sobresaltos
    o golpes producen picos de jerk altos. Es uno de los indicadores
    más sensibles de calidad biomecánica del movimiento.

    Parámetros
    ----------
    window : NDArray, shape (n_samples, n_channels)
    axes   : list de 3 índices del acelerómetro
    fs     : float — frecuencia de muestreo en Hz

    Retorna
    -------
    NDArray, shape (n_samples-1,) — serie de magnitud de jerk
    """
    acc = window[:, axes]
    # diferencia centrada aproximada
    dacc = np.diff(acc, axis=0) * fs   # unidades de aceleración / segundo
    return np.sqrt(np.sum(dacc ** 2, axis=1))


def signal_magnitude_area(window: NDArray, axes: list, fs: float) -> float:
    """
    Signal Magnitude Area (SMA): integral normalizada de la magnitud absoluta.

    SMA = (1/T) * ∫₀ᵀ (|ax(t)| + |ay(t)| + |az(t)|) dt

    Es el descriptor estándar en literatura de clasificación de actividad
    para distinguir reposo de actividad dinámica.

    Retorna
    -------
    float — valor único por ventana
    """
    sum_abs = np.sum(np.abs(window[:, axes]), axis=1)
    duration_s = window.shape[0] / fs
    return float(np.trapezoid(sum_abs, dx=1.0 / fs) / duration_s)


# ─── Agregadores estadísticos ─────────────────────────────────────────────────

def _summary_stats(series: NDArray) -> NDArray:
    """5 estadísticas: mean, std, RMS, max, min. Robustas a NaN."""
    series = series[np.isfinite(series)]
    if len(series) == 0:
        return np.zeros(5, dtype=np.float32)
    return np.array([
        np.mean(series),
        np.std(series),
        np.sqrt(np.mean(series ** 2)),
        np.max(series),
        np.min(series),
    ], dtype=np.float32)


# ─── Extractor principal ──────────────────────────────────────────────────────

# Layout de PAMAP2: cada segmento tiene 9 canales en este orden dentro del
# tensor concatenado [acc16_x, acc16_y, acc16_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
# El tensor completo es la concatenación: [hand(9) | chest(9) | ankle(9)] → 27 canales
SEGMENT_LAYOUT = {
    "hand":  {"acc": [0, 1, 2],   "gyro": [3, 4, 5]},
    "chest": {"acc": [9, 10, 11], "gyro": [12, 13, 14]},
    "ankle": {"acc": [18, 19, 20],"gyro": [21, 22, 23]},
}


def extract_biomechanical_features(window: NDArray, fs: float = 100.0) -> NDArray:
    """
    Extrae el set completo de features biomecánicas de una ventana multicanal.

    Parámetros
    ----------
    window : NDArray, shape (n_samples, 27)
        Ventana de 200 ms con 27 canales (9 por segmento × 3 segmentos)
    fs     : float — frecuencia de muestreo en Hz

    Retorna
    -------
    NDArray, shape (73,) — vector de features biomecánicas

    Composición del vector retornado:
    --------------------------------
    Por cada uno de los 3 segmentos (hand, chest, ankle):
      • Magnitud accel: 5 stats         →  5 features
      • Magnitud gyro: 5 stats          →  5 features
      • Tilt angle: 5 stats             →  5 features
      • Jerk magnitude: 5 stats         →  5 features
      • SMA: 1 valor                    →  1 feature
                                       ──────────────
                                          21 features por segmento
                                          × 3 segmentos = 63 features

    Ratios inter-segmento (coordinación):
      • hand_acc_mag / chest_acc_mag: 5 stats →  5 features
      • ankle_acc_mag / chest_acc_mag: 5 stats →  5 features
                                                ────────────
                                                10 features

    Total: 63 + 10 = 73 features biomecánicas
    """
    features = []
    seg_magnitudes = {}  # cache de magnitudes para ratios posteriores

    for seg_name, idx in SEGMENT_LAYOUT.items():
        # 1. Magnitud aceleración
        acc_mag = vector_magnitude(window, idx["acc"])
        seg_magnitudes[seg_name] = acc_mag
        features.append(_summary_stats(acc_mag))

        # 2. Magnitud giroscopio
        gyro_mag = vector_magnitude(window, idx["gyro"])
        features.append(_summary_stats(gyro_mag))

        # 3. Tilt angle (sobre acelerómetro)
        tilt = tilt_angle(window, idx["acc"])
        features.append(_summary_stats(tilt))

        # 4. Jerk magnitude
        j = jerk(window, idx["acc"], fs)
        features.append(_summary_stats(j))

        # 5. SMA (valor único)
        sma_val = signal_magnitude_area(window, idx["acc"], fs)
        features.append(np.array([sma_val], dtype=np.float32))

    # 6. Ratios inter-segmento (proxy de coordinación entre extremidades)
    eps = 1e-8
    ratio_hand_chest  = seg_magnitudes["hand"]  / (seg_magnitudes["chest"] + eps)
    ratio_ankle_chest = seg_magnitudes["ankle"] / (seg_magnitudes["chest"] + eps)

    features.append(_summary_stats(ratio_hand_chest))
    features.append(_summary_stats(ratio_ankle_chest))

    return np.concatenate(features).astype(np.float32)


# Nombres legibles de cada feature en el orden de extracción.
# Útiles para análisis de importancia y debugging.
BIOMECH_FEATURE_NAMES = []
for _seg in ["hand", "chest", "ankle"]:
    for _signal in ["accMag", "gyroMag", "tilt", "jerk"]:
        for _stat in ["mean", "std", "rms", "max", "min"]:
            BIOMECH_FEATURE_NAMES.append(f"{_seg}_{_signal}_{_stat}")
    BIOMECH_FEATURE_NAMES.append(f"{_seg}_sma")
for _pair in ["handChest", "ankleChest"]:
    for _stat in ["mean", "std", "rms", "max", "min"]:
        BIOMECH_FEATURE_NAMES.append(f"ratio_{_pair}_{_stat}")
