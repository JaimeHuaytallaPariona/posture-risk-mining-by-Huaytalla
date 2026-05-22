"""
logger.py
---------
Logger persistente de experimentos en formato CSV.

Cada fila representa un experimento completo con su configuración,
métricas y costo. El archivo es append-only: nunca se sobrescriben
filas anteriores, garantizando trazabilidad histórica.

Esquema del CSV (logs/metrics_experimentos.csv):
  - exp_id           : identificador único del experimento (EXP001, EXP002, ...)
  - exp_name         : nombre técnico (RF_n200_baseFeats, etc.)
  - timestamp        : fecha/hora ISO de ejecución
  - model            : tipo de modelo
  - feature_set      : conjunto de features usado
  - n_features       : número total de features
  - n_samples_train  : tamaño promedio del train por fold
  - n_samples_test   : tamaño promedio del test por fold
  - cv_strategy      : esquema de validación
  - n_folds          : número de folds
  - seed             : semilla aleatoria
  - f1_macro_mean    : F1-Macro promedio (métrica principal)
  - f1_macro_std     : desviación estándar
  - pr_auc_mean      : PR-AUC One-vs-Rest promedio
  - pr_auc_std       : desviación estándar
  - accuracy_mean    : Accuracy promedio
  - accuracy_std     : desviación estándar
  - train_time_s     : tiempo total de entrenamiento en segundos
  - notes            : breve descripción del cambio respecto al baseline
  - log_path         : ruta al log detallado del experimento
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict


CSV_HEADER = [
    "exp_id", "exp_name", "timestamp",
    "model", "feature_set", "n_features",
    "n_samples_train", "n_samples_test",
    "cv_strategy", "n_folds", "seed",
    "f1_macro_mean",  "f1_macro_std",
    "pr_auc_mean",    "pr_auc_std",
    "accuracy_mean",  "accuracy_std",
    "train_time_s",
    "notes", "log_path",
]


def append_experiment(record: Dict, csv_path: Path) -> None:
    """
    Añade un registro de experimento al CSV de logs.

    Si el archivo no existe, lo crea con el header. Si existe, solo
    appendea la nueva fila preservando todo el historial.

    Parámetros
    ----------
    record   : dict — debe contener TODOS los campos de CSV_HEADER
    csv_path : Path — ruta al archivo CSV
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        # Asegurar todas las columnas presentes
        row = {col: record.get(col, "") for col in CSV_HEADER}
        writer.writerow(row)


def next_exp_id(csv_path: Path) -> str:
    """
    Calcula el siguiente ID de experimento basado en filas existentes.

    Si el CSV está vacío o no existe, retorna EXP001.
    Si ya hay N experimentos registrados, retorna EXP{N+1:03d}.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return "EXP001"

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    n_records = max(0, len(rows) - 1)  # restar header
    return f"EXP{n_records + 1:03d}"


def make_record(
    exp_name: str,
    model: str,
    feature_set: str,
    metrics: dict,
    config: dict,
    notes: str = "",
    log_path: str = "",
    csv_path: Path = Path("logs/metrics_experimentos.csv"),
) -> Dict:
    """
    Construye un registro de experimento listo para serializar al CSV.

    Parámetros
    ----------
    exp_name    : str — nombre técnico (ej. "RF_n200_baseFeats")
    model       : str — tipo de modelo
    feature_set : str — identificador del feature set
    metrics     : dict — debe incluir f1_macro_mean/std, pr_auc_mean/std,
                  accuracy_mean/std, train_time_s, n_samples_train,
                  n_samples_test, n_features
    config      : dict — debe incluir cv_strategy, n_folds, seed
    notes       : str — descripción de qué cambió respecto al baseline
    log_path    : str — ruta al log detallado en disco
    csv_path    : Path — ruta del CSV (para generar exp_id consistente)
    """
    return {
        "exp_id":          next_exp_id(csv_path),
        "exp_name":        exp_name,
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "model":           model,
        "feature_set":     feature_set,
        "n_features":      metrics.get("n_features", ""),
        "n_samples_train": metrics.get("n_samples_train", ""),
        "n_samples_test":  metrics.get("n_samples_test", ""),
        "cv_strategy":     config.get("cv_strategy", "GroupKFold-LOSO"),
        "n_folds":         config.get("n_folds", ""),
        "seed":            config.get("seed", ""),
        "f1_macro_mean":   f"{metrics['f1_macro_mean']:.4f}",
        "f1_macro_std":    f"{metrics['f1_macro_std']:.4f}",
        "pr_auc_mean":     f"{metrics['pr_auc_mean']:.4f}",
        "pr_auc_std":      f"{metrics['pr_auc_std']:.4f}",
        "accuracy_mean":   f"{metrics['accuracy_mean']:.4f}",
        "accuracy_std":    f"{metrics['accuracy_std']:.4f}",
        "train_time_s":    f"{metrics['train_time_s']:.2f}",
        "notes":           notes,
        "log_path":        log_path,
    }
