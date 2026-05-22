"""
ab_runner.py
------------
Orquestador del experimento A/B con 3 variantes.

Implementa el protocolo del Ing. Glen Rodríguez:
  - Mismo split / seed / datos para las 3 variantes (aislar efecto causal)
  - 1 cambio único por variante (no mezclar)
  - Logging completo con trazabilidad

Variantes implementadas:
  A. RF_n200_baseFeats     — Baseline: Random Forest + 297 features estadísticas
  B. RF_n200_biomechFeats  — Var1: mismo RF + 297 features + 73 biomecánicas (370)
  C. XGB_n200_baseFeats    — Var2: XGBoost + 297 features base (mismo set que A)

Cómo se garantiza "cero leakage":
  1. Los splits se generan UNA VEZ al inicio con seed fijo y se reutilizan
  2. El StandardScaler se entrena SOLO con datos de entrenamiento por fold
  3. Las features biomecánicas son funciones puras de la ventana cruda
     (no dependen de estadísticas del dataset)
  4. El cálculo se hace dentro del bucle de CV, no antes
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from loguru import logger as log
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_recall_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from tqdm import tqdm


# ── Carga del modelo Gradient Boosting con fallback ──────────────────────────
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    XGB_BACKEND = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGB_AVAILABLE = False
    XGB_BACKEND = "sklearn-HistGB"


# ─── Carga y preparación de datos ─────────────────────────────────────────────

def load_processed_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga el HDF5 generado por el pipeline de ingesta del baseline."""
    with h5py.File(h5_path, "r") as f:
        X_base      = f["X"][:]
        y           = f["y"][:]
        subject_ids = f["subject_ids"][:]
    X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)
    log.info(f"Dataset base: X={X_base.shape}, y={y.shape}, sujetos={np.unique(subject_ids)}")
    return X_base, y, subject_ids


def compute_biomechanical_features(
    raw_h5_path: Path,
    fs: float = 100.0,
    window_ms: float = 200.0,
    overlap_ratio: float = 0.5,
) -> np.ndarray:
    """
    Re-procesa las señales crudas para extraer features biomecánicas con
    el MISMO esquema de ventaneo del pipeline original.

    Esta función reconstruye las ventanas usando el mismo split temporal del
    pipeline para garantizar que cada ventana biomecánica corresponda 1:1 con
    su contraparte estadística en X_base.

    NOTA: para mantener compatibilidad con el flujo actual, esta función
    NO se ejecuta aquí — las features biomecánicas se calculan directamente
    desde PAMAP2 en el script `prepare_biomech_dataset.py` que produce un
    HDF5 adicional. ab_runner.py simplemente carga ambos HDF5 y los concatena.
    """
    raise NotImplementedError(
        "Las features biomecánicas se generan vía prepare_biomech_dataset.py. "
        "Ver notebook 03_ab_experiments para el flujo completo."
    )


# ─── Construcción de modelos ──────────────────────────────────────────────────

def build_random_forest(seed: int, n_estimators: int = 200) -> Pipeline:
    """
    Pipeline reproducible: StandardScaler + RandomForest.

    El StandardScaler se ajustará SOLO con los datos de train de cada fold,
    por construcción de sklearn.Pipeline. Esto cumple "divide primero,
    transforma después" del protocolo del asesor.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators      = n_estimators,
            max_depth         = 20,
            min_samples_split = 5,
            min_samples_leaf  = 2,
            class_weight      = "balanced",
            n_jobs            = -1,
            random_state      = seed,
        )),
    ])


def build_gradient_boosting(seed: int, n_estimators: int = 200) -> Pipeline:
    """
    Pipeline reproducible: StandardScaler + Gradient Boosting.

    Usa XGBoost si está disponible; cae a sklearn HistGradientBoosting si no.
    Ambos son boosting basado en histogramas — comparables en rendimiento.
    """
    if XGB_AVAILABLE:
        clf = XGBClassifier(
            n_estimators     = n_estimators,
            max_depth        = 6,
            learning_rate    = 0.1,
            tree_method      = "hist",
            random_state     = seed,
            n_jobs           = -1,
            eval_metric      = "mlogloss",
            verbosity        = 0,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter           = n_estimators,
            max_depth          = 6,
            learning_rate      = 0.1,
            class_weight       = "balanced",
            random_state       = seed,
        )

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ─── Loop de CV reutilizable ──────────────────────────────────────────────────

def cross_validate_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    exp_name: str,
) -> Dict:
    """
    Ejecuta validación cruzada con splits pre-definidos.

    Acepta `splits` como argumento (no los genera internamente) para garantizar
    que TODAS las variantes vean exactamente los mismos folds. Esta es la clave
    de la comparación A/B justa según el protocolo del asesor.
    """
    n_classes  = len(np.unique(y))
    fold_results = []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    t_start = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(splits, desc=exp_name, leave=False)
    ):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # fit() entrena StandardScaler + modelo SOLO con train
        pipeline.fit(X_tr, y_tr)
        y_pred  = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)

        # Métricas del fold
        acc    = accuracy_score(y_te, y_pred)
        f1_mac = f1_score(y_te, y_pred, average="macro", zero_division=0)

        # PR-AUC One-vs-Rest (métrica recomendada por el asesor para clasificación)
        y_te_bin = label_binarize(y_te, classes=list(range(n_classes)))
        try:
            pr_auc = average_precision_score(y_te_bin, y_proba, average="macro")
        except Exception:
            pr_auc = float("nan")

        fold_results.append({
            "fold": fold_idx,
            "test_subject": int(np.unique(groups[test_idx])[0]),
            "accuracy":  acc,
            "f1_macro":  f1_mac,
            "pr_auc":    pr_auc,
            "n_train":   len(train_idx),
            "n_test":    len(test_idx),
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        })
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

    train_time = time.time() - t_start

    # Agregados
    accs   = [r["accuracy"]  for r in fold_results]
    f1s    = [r["f1_macro"]  for r in fold_results]
    prs    = [r["pr_auc"]    for r in fold_results if not np.isnan(r["pr_auc"])]

    summary = {
        "exp_name":         exp_name,
        "n_features":       X.shape[1],
        "n_samples_train":  int(np.mean([r["n_train"] for r in fold_results])),
        "n_samples_test":   int(np.mean([r["n_test"]  for r in fold_results])),
        "accuracy_mean":    float(np.mean(accs)),
        "accuracy_std":     float(np.std(accs)),
        "f1_macro_mean":    float(np.mean(f1s)),
        "f1_macro_std":     float(np.std(f1s)),
        "pr_auc_mean":      float(np.mean(prs)) if prs else float("nan"),
        "pr_auc_std":       float(np.std(prs))  if prs else float("nan"),
        "train_time_s":     train_time,
        "fold_results":     fold_results,
        "y_true_all":       np.array(all_y_true),
        "y_pred_all":       np.array(all_y_pred),
        "y_proba_all":      np.array(all_y_proba),
    }

    log.info(
        f"{exp_name:<28} | F1={summary['f1_macro_mean']:.4f}±{summary['f1_macro_std']:.4f} | "
        f"PR-AUC={summary['pr_auc_mean']:.4f}±{summary['pr_auc_std']:.4f} | "
        f"Time={train_time:.1f}s"
    )
    return summary


# ─── Generador de splits ──────────────────────────────────────────────────────

def make_shared_splits(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera los splits LOSO/GroupKFold una sola vez para reutilizar.

    Esto es lo que el asesor llama "mismo split entre baseline y variantes".
    Cada variante recibirá esta misma lista, garantizando comparación justa.
    """
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=n_groups)
    splits = list(splitter.split(np.zeros_like(y), y, groups=groups))
    log.info(f"Splits generados: {len(splits)} folds, seed={seed}")
    return splits
