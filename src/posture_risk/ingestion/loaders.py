"""
loaders.py
----------
Funciones para leer y parsear archivos del dataset PAMAP2.

Formato PAMAP2:
  Cada archivo subject{N}.dat contiene 54 columnas separadas por espacios:
  col 0:    timestamp (segundos)
  col 1:    activityID (etiqueta de actividad, 0 = transitional)
  col 2:    heart rate (BPM)
  cols 3-20:  IMU muñeca  (hand)   — temperatura + 4×accel + 4×gyro + 4×mag + 4×orient
  cols 21-38: IMU pecho   (chest)  — misma estructura
  cols 38-55: IMU tobillo (ankle)  — misma estructura

Referencia: https://archive.ics.uci.edu/dataset/231
"""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Dict, Optional


# Nombres de columnas completos del archivo PAMAP2
_PAMAP2_COLS = (
    ["timestamp", "activity_id", "heart_rate"]
    + [f"hand_{c}" for c in [
        "temp",
        "acc16_x", "acc16_y", "acc16_z",
        "acc6_x",  "acc6_y",  "acc6_z",
        "gyro_x",  "gyro_y",  "gyro_z",
        "mag_x",   "mag_y",   "mag_z",
        "orient_1","orient_2","orient_3","orient_4",
    ]]
    + [f"chest_{c}" for c in [
        "temp",
        "acc16_x", "acc16_y", "acc16_z",
        "acc6_x",  "acc6_y",  "acc6_z",
        "gyro_x",  "gyro_y",  "gyro_z",
        "mag_x",   "mag_y",   "mag_z",
        "orient_1","orient_2","orient_3","orient_4",
    ]]
    + [f"ankle_{c}" for c in [
        "temp",
        "acc16_x", "acc16_y", "acc16_z",
        "acc6_x",  "acc6_y",  "acc6_z",
        "gyro_x",  "gyro_y",  "gyro_z",
        "mag_x",   "mag_y",   "mag_z",
        "orient_1","orient_2","orient_3","orient_4",
    ]]
)

# Canales IMU relevantes por segmento (excluye orientación y temperatura)
IMU_SIGNAL_COLS = {
    "hand":  [f"hand_{c}"  for c in ["acc16_x","acc16_y","acc16_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]],
    "chest": [f"chest_{c}" for c in ["acc16_x","acc16_y","acc16_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]],
    "ankle": [f"ankle_{c}" for c in ["acc16_x","acc16_y","acc16_z","gyro_x","gyro_y","gyro_z","mag_x","mag_y","mag_z"]],
}


def load_pamap2_subject(
    path: Path,
    drop_transitional: bool = True,
    interpolate_nans: bool = True,
) -> pd.DataFrame:
    """
    Carga un archivo .dat de PAMAP2 y retorna un DataFrame limpio.

    Parámetros
    ----------
    path : Path
        Ruta al archivo, ej. data/raw/public/PAMAP2/Protocol/subject101.dat
    drop_transitional : bool
        Si True, elimina filas con activity_id == 0 (transiciones no etiquetadas).
    interpolate_nans : bool
        Si True, interpola linealmente los NaN en las señales de IMU.

    Retorna
    -------
    pd.DataFrame con columnas nombradas y señales limpias.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Archivo no encontrado: {path}\n"
            f"Descarga PAMAP2 desde: https://archive.ics.uci.edu/dataset/231"
        )

    logger.info(f"Cargando: {path.name}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=_PAMAP2_COLS,
        na_values="NaN",
    )

    if drop_transitional:
        n_before = len(df)
        df = df[df["activity_id"] != 0].reset_index(drop=True)
        logger.debug(f"  Transitional removidos: {n_before - len(df)} filas")

    if interpolate_nans:
        signal_cols = (
            IMU_SIGNAL_COLS["hand"]
            + IMU_SIGNAL_COLS["chest"]
            + IMU_SIGNAL_COLS["ankle"]
        )
        nan_count = df[signal_cols].isna().sum().sum()
        if nan_count > 0:
            logger.debug(f"  Interpolando {nan_count} NaN en señales IMU")
            df[signal_cols] = df[signal_cols].interpolate(
                method="linear", limit_direction="both"
            )

    logger.info(f"  Filas: {len(df)} | Actividades: {df['activity_id'].unique().tolist()}")
    return df


def load_pamap2_dataset(
    data_dir: Path,
    subject_ids: Optional[list] = None,
) -> Dict[int, pd.DataFrame]:
    """
    Carga todos los sujetos del dataset PAMAP2.

    Parámetros
    ----------
    data_dir : Path
        Directorio que contiene los archivos subject1XX.dat
    subject_ids : list, opcional
        Lista de IDs a cargar. Si None, carga todos los disponibles.

    Retorna
    -------
    Dict[int, pd.DataFrame] — clave: subject_id, valor: DataFrame del sujeto
    """
    data_dir = Path(data_dir)
    available = sorted(data_dir.glob("subject*.dat"))

    if not available:
        raise FileNotFoundError(
            f"No se encontraron archivos .dat en {data_dir}\n"
            "Verifica que hayas descomprimido PAMAP2 correctamente."
        )

    subjects = {}
    for filepath in available:
        # Extrae el ID numérico del nombre (subject101.dat → 1)
        sid = int("".join(filter(str.isdigit, filepath.stem))) - 100
        if subject_ids is None or sid in subject_ids:
            subjects[sid] = load_pamap2_subject(filepath)

    logger.info(f"Dataset cargado: {len(subjects)} sujetos")
    return subjects
