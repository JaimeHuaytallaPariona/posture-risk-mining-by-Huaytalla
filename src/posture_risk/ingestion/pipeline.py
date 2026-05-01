"""
pipeline.py
-----------
Script de ingesta reproducible para el dataset PAMAP2.

Convierte archivos .dat crudos en un tensor de features HDF5 listo
para entrenamiento. Todas las rutas y parámetros vienen de default.yaml.

Uso:
    python -m posture_risk.ingestion.pipeline
    python -m posture_risk.ingestion.pipeline --config configs/default.yaml
    python -m posture_risk.ingestion.pipeline --config configs/default.yaml --subjects 1 2 3

Fases del pipeline:
    1. Carga y validación de datos crudos por sujeto
    2. Filtrado Butterworth pasa-banda por segmento IMU
    3. Segmentación en ventanas deslizantes con solapamiento
    4. Extracción de features temporales y espectrales por ventana
    5. Mapeo de etiquetas de actividad a niveles de riesgo proxy
    6. Escritura en HDF5 con metadatos completos
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
from loguru import logger
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from posture_risk.ingestion.loaders import load_pamap2_dataset, IMU_SIGNAL_COLS
from posture_risk.features.temporal import extract_temporal_features
from posture_risk.features.spectral import extract_spectral_features


# ── Utilidades internas ───────────────────────────────────────────────────────

def _butter_bandpass(signal: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Aplica filtro Butterworth pasa-banda a una señal multicanal."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def _make_windows(signal: np.ndarray, labels: np.ndarray, window_samples: int, step_samples: int):
    """
    Segmenta señal + etiquetas en ventanas deslizantes.

    Estrategia de etiqueta por ventana: moda de las etiquetas de la ventana.
    Si la ventana contiene más de una actividad (transición), se descarta.
    """
    X_windows, y_windows = [], []
    n = len(signal)
    for start in range(0, n - window_samples + 1, step_samples):
        end = start + window_samples
        window_labels = labels[start:end]
        # Descartar ventanas con múltiples actividades
        unique = np.unique(window_labels)
        if len(unique) > 1:
            continue
        X_windows.append(signal[start:end])
        y_windows.append(unique[0])
    return np.array(X_windows), np.array(y_windows)


def _build_activity_label_map(cfg: dict) -> dict:
    """Construye mapeo activity_id → risk_level (0=bajo, 1=medio, 2=alto)."""
    label_map = {}
    for activity_id in cfg["dataset"]["activity_map"]["low_risk"]:
        label_map[activity_id] = 0
    for activity_id in cfg["dataset"]["activity_map"]["medium_risk"]:
        label_map[activity_id] = 1
    for activity_id in cfg["dataset"]["activity_map"]["high_risk"]:
        label_map[activity_id] = 2
    return label_map


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_pipeline(cfg: dict) -> Path:
    """
    Ejecuta el pipeline completo de ingesta.

    Parámetros
    ----------
    cfg : dict — configuración cargada desde default.yaml

    Retorna
    -------
    Path al archivo HDF5 generado
    """
    t0 = time.time()
    seed = cfg["project"]["seed"]
    np.random.seed(seed)

    fs            = cfg["acquisition"]["sampling_rate_hz"]
    window_ms     = cfg["windowing"]["window_ms"]
    overlap       = cfg["windowing"]["overlap_ratio"]
    window_samp   = int((window_ms / 1000.0) * fs)   # muestras por ventana
    step_samp     = int(window_samp * (1 - overlap))  # paso en muestras

    filt_cfg      = cfg["filtering"]["imu"]
    subject_ids   = cfg["dataset"]["subjects"]
    data_dir      = Path(cfg["paths"]["raw_public"])
    out_dir       = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map     = _build_activity_label_map(cfg)
    segments      = list(IMU_SIGNAL_COLS.keys())  # ["hand", "chest", "ankle"]
    all_features  = cfg["features"]["temporal"] + cfg["features"]["spectral"]

    logger.info("=" * 60)
    logger.info("PIPELINE DE INGESTA — PAMAP2")
    logger.info(f"  Ventana: {window_ms} ms ({window_samp} muestras)")
    logger.info(f"  Solapamiento: {overlap*100:.0f}% ({step_samp} muestras de paso)")
    logger.info(f"  Sujetos: {subject_ids}")
    logger.info("=" * 60)

    # ── Fase 1: Carga de datos ────────────────────────────────────────────────
    logger.info("Fase 1/5: Cargando datos crudos...")
    subjects_data = load_pamap2_dataset(data_dir, subject_ids=subject_ids)

    # ── Fase 2–5: Procesamiento por sujeto ───────────────────────────────────
    all_X, all_y, all_subject_ids = [], [], []

    for sid, df in tqdm(subjects_data.items(), desc="Sujetos"):

        # ── Fase 2: Filtrado Butterworth ──────────────────────────────────
        filtered_segments = {}
        for seg in segments:
            cols      = IMU_SIGNAL_COLS[seg]
            raw_sig   = df[cols].values.astype(np.float32)
            filtered_segments[seg] = _butter_bandpass(
                raw_sig, fs,
                filt_cfg["lowcut_hz"],
                filt_cfg["highcut_hz"],
                filt_cfg["order"],
            )

        # Concatenar segmentos en señal multicanal unificada
        signal_concat = np.concatenate(
            [filtered_segments[seg] for seg in segments], axis=1
        )  # shape: (n_samples, 27)  — 9 canales × 3 segmentos

        # Mapeo de etiquetas PAMAP2 → risk_level (ignorar actividades no mapeadas)
        raw_labels  = df["activity_id"].values
        mapped_mask = np.isin(raw_labels, list(label_map.keys()))
        signal_filtered = signal_concat[mapped_mask]
        labels_mapped   = np.array([label_map[a] for a in raw_labels[mapped_mask]])

        # ── Fase 3: Segmentación en ventanas ─────────────────────────────
        X_windows, y_windows = _make_windows(
            signal_filtered, labels_mapped, window_samp, step_samp
        )

        if len(X_windows) == 0:
            logger.warning(f"  Sujeto {sid}: 0 ventanas válidas, omitiendo")
            continue

        # ── Fase 4: Extracción de features ───────────────────────────────
        X_features = np.zeros((len(X_windows), 11 * signal_concat.shape[1]), dtype=np.float32)
        # 11 features = 6 temporal + 5 spectral por canal

        for i, win in enumerate(X_windows):
            t_feats = extract_temporal_features(win)
            s_feats = extract_spectral_features(win, fs)
            X_features[i] = np.concatenate([t_feats, s_feats])

        all_X.append(X_features)
        all_y.append(y_windows.astype(np.int32))
        all_subject_ids.extend([sid] * len(X_features))
        logger.debug(f"  Sujeto {sid}: {len(X_features)} ventanas extraídas")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids_arr = np.array(all_subject_ids, dtype=np.int32)

    # ── Fase 5: Escritura HDF5 ────────────────────────────────────────────────
    logger.info("Fase 5/5: Escribiendo HDF5...")
    out_path = out_dir / "pamap2_features.h5"

    with h5py.File(out_path, "w") as f:
        # Datos
        f.create_dataset("X",           data=X,               compression="gzip")
        f.create_dataset("y",           data=y,               compression="gzip")
        f.create_dataset("subject_ids", data=subject_ids_arr, compression="gzip")

        # Metadatos completos — aseguran la reproducibilidad
        f.attrs["window_ms"]        = window_ms
        f.attrs["overlap_ratio"]    = overlap
        f.attrs["sampling_rate_hz"] = fs
        f.attrs["n_subjects"]       = len(subjects_data)
        f.attrs["n_windows"]        = X.shape[0]
        f.attrs["n_features"]       = X.shape[1]
        f.attrs["label_map"]        = str(label_map)
        f.attrs["segments"]         = str(segments)
        f.attrs["feature_names"]    = str(all_features)
        f.attrs["filter_lowcut"]    = filt_cfg["lowcut_hz"]
        f.attrs["filter_highcut"]   = filt_cfg["highcut_hz"]
        f.attrs["created_at"]       = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["seed"]             = seed

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"INGESTA COMPLETA en {elapsed:.1f}s")
    logger.info(f"  Shape X : {X.shape}  (ventanas × features)")
    logger.info(f"  Shape y : {y.shape}  — clases: {np.unique(y, return_counts=True)}")
    logger.info(f"  Salida  : {out_path}")
    logger.info("=" * 60)

    return out_path


# ── Punto de entrada CLI ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de ingesta PAMAP2 → HDF5"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Ruta al archivo de configuración YAML"
    )
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=None,
        help="IDs de sujetos a procesar (por defecto: todos)"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.subjects:
        cfg["dataset"]["subjects"] = args.subjects

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
