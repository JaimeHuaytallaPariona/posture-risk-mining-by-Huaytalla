"""
prepare_biomech_dataset.py
--------------------------
Genera un HDF5 con features biomecánicas alineadas 1:1 con el HDF5 base
del baseline (pamap2_features.h5).

Por qué se necesita este script:
  El pipeline original solo guarda las 297 features estadísticas. Para la
  Variante 1 necesitamos AÑADIR las 73 features biomecánicas, garantizando
  que cada ventana biomecánica corresponda a la MISMA ventana cruda que
  produjo la fila de X_base.

Solución: reproducir EXACTAMENTE el mismo loop de ventaneo del pipeline
original, con los mismos parámetros (200 ms, 50% solapamiento, mismo
criterio de descarte de transiciones), pero calculando features
biomecánicas en lugar de estadísticas.

Output: data/processed/pamap2_biomech_features.h5
  - X_biomech    : shape (N, 73)
  - y            : shape (N,)        ← idéntico al baseline
  - subject_ids  : shape (N,)        ← idéntico al baseline

Verificación de alineamiento al final del script:
  Las primeras N etiquetas deben coincidir con las del HDF5 base.
"""

import argparse
from pathlib import Path
import time

import h5py
import numpy as np
import yaml
from loguru import logger
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from posture_risk.ingestion.loaders import IMU_SIGNAL_COLS, load_pamap2_dataset
from posture_risk.features.biomechanical import extract_biomechanical_features


def _butter_bandpass(signal, fs, low, high, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=0)


def _build_activity_label_map(cfg):
    label_map = {}
    for aid in cfg["dataset"]["activity_map"]["low_risk"]:    label_map[aid] = 0
    for aid in cfg["dataset"]["activity_map"]["medium_risk"]: label_map[aid] = 1
    for aid in cfg["dataset"]["activity_map"]["high_risk"]:   label_map[aid] = 2
    return label_map


def run(cfg: dict) -> Path:
    """Genera el HDF5 biomecánico aplicando el mismo ventaneo del baseline."""
    fs           = cfg["acquisition"]["sampling_rate_hz"]
    window_ms    = cfg["windowing"]["window_ms"]
    overlap      = cfg["windowing"]["overlap_ratio"]
    window_samp  = int((window_ms / 1000.0) * fs)
    step_samp    = int(window_samp * (1 - overlap))
    filt_cfg     = cfg["filtering"]["imu"]
    data_dir     = Path(cfg["paths"]["raw_public"])
    out_dir      = Path(cfg["paths"]["processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_ids  = cfg["dataset"]["subjects"]
    label_map    = _build_activity_label_map(cfg)
    segments     = list(IMU_SIGNAL_COLS.keys())

    logger.info("=" * 60)
    logger.info("PREPARACIÓN DE DATASET BIOMECÁNICO")
    logger.info(f"  Ventana: {window_ms} ms | Solapamiento: {overlap*100:.0f}%")
    logger.info("=" * 60)

    subjects_data = load_pamap2_dataset(data_dir, subject_ids=subject_ids)

    all_X_biomech, all_y, all_subj = [], [], []
    t0 = time.time()

    for sid, df in tqdm(subjects_data.items(), desc="Sujetos"):
        # Filtrado Butterworth idéntico al baseline
        filtered = {}
        for seg in segments:
            raw = df[IMU_SIGNAL_COLS[seg]].values.astype(np.float32)
            filtered[seg] = _butter_bandpass(
                raw, fs, filt_cfg["lowcut_hz"], filt_cfg["highcut_hz"], filt_cfg["order"]
            )
        signal = np.concatenate([filtered[s] for s in segments], axis=1)

        # Filtrado de actividades mapeadas
        raw_labels  = df["activity_id"].values
        mask        = np.isin(raw_labels, list(label_map.keys()))
        signal      = signal[mask]
        labels      = np.array([label_map[a] for a in raw_labels[mask]])

        # Ventaneo IDÉNTICO al pipeline original
        n = len(signal)
        for start in range(0, n - window_samp + 1, step_samp):
            end = start + window_samp
            win_labels = labels[start:end]
            unique = np.unique(win_labels)
            if len(unique) > 1:
                continue  # mismo criterio de descarte de transiciones
            win = signal[start:end]
            feats = extract_biomechanical_features(win, fs=fs)
            all_X_biomech.append(feats)
            all_y.append(int(unique[0]))
            all_subj.append(sid)

    X_biomech = np.array(all_X_biomech, dtype=np.float32)
    y         = np.array(all_y, dtype=np.int32)
    subj      = np.array(all_subj, dtype=np.int32)

    out_path = out_dir / "pamap2_biomech_features.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("X_biomech",   data=X_biomech, compression="gzip")
        f.create_dataset("y",           data=y,         compression="gzip")
        f.create_dataset("subject_ids", data=subj,      compression="gzip")
        f.attrs["n_features"]  = X_biomech.shape[1]
        f.attrs["window_ms"]   = window_ms
        f.attrs["overlap"]     = overlap
        f.attrs["fs_hz"]       = fs
        f.attrs["created_at"]  = time.strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs["seed"]        = cfg["project"]["seed"]
        f.attrs["description"] = "Features biomecánicas alineadas 1:1 con pamap2_features.h5"

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(f"GENERACIÓN COMPLETA en {elapsed:.1f}s")
    logger.info(f"  X_biomech: {X_biomech.shape}")
    logger.info(f"  Salida   : {out_path}")
    logger.info("=" * 60)

    # ── Verificación de alineamiento con HDF5 base ───────────────────────────
    base_path = out_dir / "pamap2_features.h5"
    if base_path.exists():
        with h5py.File(base_path, "r") as f:
            y_base    = f["y"][:]
            subj_base = f["subject_ids"][:]
        if len(y_base) != len(y):
            logger.error(f"DESALINEAMIENTO: base={len(y_base)} vs biomech={len(y)}")
            raise RuntimeError("El HDF5 biomecánico NO está alineado con el base")
        elif not np.array_equal(y_base, y) or not np.array_equal(subj_base, subj):
            logger.error("Las etiquetas o subject_ids no coinciden con el HDF5 base")
            raise RuntimeError("Desalineamiento detectado")
        else:
            logger.info("✓ Alineamiento 1:1 verificado contra pamap2_features.h5")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generar dataset biomecánico")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()
