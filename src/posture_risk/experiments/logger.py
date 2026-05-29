"""
logger.py  (Sprint 2 — actualizado)
-------------------------------------
Cambios respecto a Sprint 1:
  - Añade campo 'data_hash'   : hash MD5 del HDF5 base (ítem 4 del checklist)
  - Añade campo 'git_commit'  : hash corto del commit activo en git
  - Estos dos campos hacen el log completamente auditable (ítem 5)
"""

import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


CSV_HEADER = [
    "exp_id", "exp_name", "timestamp",
    "model", "feature_set", "n_features",
    "n_samples_train", "n_samples_test",
    "cv_strategy", "n_folds", "seed",
    "f1_macro_mean",  "f1_macro_std",
    "pr_auc_mean",    "pr_auc_std",
    "accuracy_mean",  "accuracy_std",
    "train_time_s",
    "data_hash",      # NUEVO Sprint 2 — hash MD5 del HDF5 base
    "git_commit",     # NUEVO Sprint 2 — hash corto del commit activo
    "notes", "log_path",
]


def get_git_commit() -> str:
    """Retorna el hash corto del commit activo en git, o 'N/A' si no está disponible."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


def append_experiment(record: Dict, csv_path: Path) -> None:
    """Añade un registro al CSV de logs (append-only, nunca sobrescribe)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        row = {col: record.get(col, "") for col in CSV_HEADER}
        writer.writerow(row)


def next_exp_id(csv_path: Path) -> str:
    """Calcula el siguiente ID de experimento."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return "EXP001"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    n_records = max(0, len(rows) - 1)
    return f"EXP{n_records + 1:03d}"


def make_record(
    exp_name:    str,
    model:       str,
    feature_set: str,
    metrics:     dict,
    config:      dict,
    notes:       str = "",
    log_path:    str = "",
    data_hash:   str = "",        # NUEVO Sprint 2
    csv_path:    Path = Path("logs/metrics_experimentos.csv"),
) -> Dict:
    """
    Construye un registro de experimento completo para serializar al CSV.

    Parámetros nuevos en Sprint 2:
      data_hash : hash MD5 del archivo HDF5 base (para ítem 4 del checklist)
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
        "data_hash":       data_hash,              # NUEVO Sprint 2
        "git_commit":      get_git_commit(),       # NUEVO Sprint 2
        "notes":           notes,
        "log_path":        log_path,
    }
