"""
extract_activity_ids.py  (versión corregida)
--------------------------------------------
Reconstruye los activity_ids de PAMAP2 alineados 1:1 con las ventanas
del HDF5 base (data/processed/pamap2_features.h5).

CORRECCIÓN vs versión anterior: el pipeline original descarta ventanas solo
cuando mezclan NIVELES DE RIESGO (0/1/2), no cuando mezclan actividades
individuales de un mismo nivel. Esto explica por qué la versión anterior
producía 118 ventanas menos que el HDF5 base.

Para cada ventana, se registra como activity_id la actividad MAYORITARIA
(la más frecuente dentro de la ventana). Como el criterio garantiza que
todas las muestras de la ventana pertenecen al mismo nivel de riesgo,
la actividad mayoritaria es representativa.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from loguru import logger
from tqdm import tqdm

from posture_risk.ingestion.loaders import load_pamap2_dataset


def _build_activity_label_map(cfg):
    label_map = {}
    for aid in cfg["dataset"]["activity_map"]["low_risk"]:    label_map[aid] = 0
    for aid in cfg["dataset"]["activity_map"]["medium_risk"]: label_map[aid] = 1
    for aid in cfg["dataset"]["activity_map"]["high_risk"]:   label_map[aid] = 2
    return label_map


def run(cfg: dict) -> Path:
    """Extrae activity_ids alineados con las ventanas del HDF5 base."""
    fs           = cfg["acquisition"]["sampling_rate_hz"]
    window_ms    = cfg["windowing"]["window_ms"]
    overlap      = cfg["windowing"]["overlap_ratio"]
    window_samp  = int((window_ms / 1000.0) * fs)
    step_samp    = int(window_samp * (1 - overlap))
    data_dir     = Path(cfg["paths"]["raw_public"])
    out_dir      = Path(cfg["paths"]["processed"])
    subject_ids_cfg = cfg["dataset"]["subjects"]
    label_map    = _build_activity_label_map(cfg)

    logger.info("=" * 60)
    logger.info("EXTRACCIÓN DE activity_ids ALINEADOS")
    logger.info(f"  Ventana: {window_ms} ms | Solapamiento: {overlap*100:.0f}%")
    logger.info("=" * 60)

    subjects_data = load_pamap2_dataset(data_dir, subject_ids=subject_ids_cfg)

    all_activity_ids = []
    all_subject_ids  = []

    for sid, df in tqdm(subjects_data.items(), desc="Sujetos"):
        raw_labels = df["activity_id"].values
        mask       = np.isin(raw_labels, list(label_map.keys()))
        raw_acts   = raw_labels[mask]
        # Nivel de riesgo por muestra (0, 1, 2)
        risk_labels = np.array([label_map[a] for a in raw_acts], dtype=np.int32)

        n = len(risk_labels)
        for start in range(0, n - window_samp + 1, step_samp):
            end = start + window_samp
            win_risk = risk_labels[start:end]

            # Descartar solo si mezcla NIVELES DE RIESGO (mismo criterio que el pipeline base)
            if len(np.unique(win_risk)) > 1:
                continue

            # Actividad mayoritaria dentro de la ventana
            win_acts = raw_acts[start:end]
            # np.bincount requiere enteros no negativos; los IDs de PAMAP2 son enteros pequeños
            activity_id_win = int(np.bincount(win_acts.astype(np.int64)).argmax())

            all_activity_ids.append(activity_id_win)
            all_subject_ids.append(sid)

    activity_ids_arr = np.array(all_activity_ids, dtype=np.int32)
    subject_ids_arr  = np.array(all_subject_ids,  dtype=np.int32)

    # ── Verificar alineamiento con HDF5 base ─────────────────────────────────
    base_path = out_dir / "pamap2_features.h5"
    if not base_path.exists():
        raise FileNotFoundError(f"No existe {base_path}. Ejecuta primero el pipeline base.")

    with h5py.File(base_path, "r") as f:
        n_windows_base   = f["y"].shape[0]
        subj_base        = f["subject_ids"][:]

    if len(activity_ids_arr) != n_windows_base:
        raise RuntimeError(
            f"Desalineamiento: HDF5 base tiene {n_windows_base} ventanas, "
            f"pero la reconstrucción produjo {len(activity_ids_arr)}."
        )

    if not np.array_equal(subject_ids_arr, subj_base):
        raise RuntimeError("Los subject_ids no coinciden con el HDF5 base")

    logger.info(f"✓ Alineamiento verificado: {n_windows_base} ventanas")

    # ── Añadir activity_ids al HDF5 base ─────────────────────────────────────
    with h5py.File(base_path, "a") as f:
        if "activity_ids" in f:
            del f["activity_ids"]  # sobrescribir si existía
            logger.info("  activity_ids existente sobrescrito")
        f.create_dataset("activity_ids", data=activity_ids_arr, compression="gzip")
        logger.info(f"  activity_ids añadido al HDF5 base ({activity_ids_arr.shape})")

    # Reporte de actividades encontradas
    unique_acts, counts = np.unique(activity_ids_arr, return_counts=True)
    logger.info(f"\nDistribución de actividades:")
    for a, c in zip(unique_acts, counts):
        logger.info(f"  Actividad {a}: {c} ventanas")

    return base_path


def main():
    parser = argparse.ArgumentParser(description="Extraer activity_ids alineados")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run(cfg)


if __name__ == "__main__":
    main()
