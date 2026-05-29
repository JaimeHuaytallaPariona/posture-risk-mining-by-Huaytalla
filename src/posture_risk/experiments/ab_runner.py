"""
ab_runner.py  (Sprint 2 — actualizado)
---------------------------------------
Cambios respecto a la versión del Sprint 1:
  - cross_validate_pipeline ahora retiene las importancias de features
    de cada fold (necesario para análisis de estabilidad de la sesión 6).
  - Se añade el campo 'fold_importances' al dict de resultados.
  - Se añade helper get_feature_importances() para extraer importancias
    del pipeline de forma agnóstica al tipo de modelo.
  - Sin cambios en la lógica de split/seed/métricas (ítem 3 del checklist).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from loguru import logger as log
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    XGB_BACKEND = "xgboost"
except ImportError:
    XGB_AVAILABLE = False
    XGB_BACKEND = "sklearn-HistGB"


# ─── Carga de datos ───────────────────────────────────────────────────────────

def load_processed_h5(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        X           = f["X"][:]
        y           = f["y"][:]
        subject_ids = f["subject_ids"][:]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log.info(f"Dataset base: X={X.shape}, y={y.shape}, sujetos={np.unique(subject_ids)}")
    return X, y, subject_ids


# ─── Construcción de modelos ──────────────────────────────────────────────────

def build_random_forest(seed: int, n_estimators: int = 200) -> Pipeline:
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
    if XGB_AVAILABLE:
        clf = XGBClassifier(
            n_estimators  = n_estimators,
            max_depth     = 6,
            learning_rate = 0.1,
            tree_method   = "hist",
            random_state  = seed,
            n_jobs        = -1,
            eval_metric   = "mlogloss",
            verbosity     = 0,
        )
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_iter      = n_estimators,
            max_depth     = 6,
            learning_rate = 0.1,
            class_weight  = "balanced",
            random_state  = seed,
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


# ─── Extracción de importancias ───────────────────────────────────────────────

def get_feature_importances(pipeline: Pipeline) -> Optional[np.ndarray]:
    """
    Extrae el vector de importancias del clasificador dentro del pipeline.

    Soporta:
      - RandomForest / ExtraTreesClassifier → .feature_importances_
      - XGBoost XGBClassifier               → .feature_importances_ (gain)
      - HistGradientBoostingClassifier       → .feature_importances_
      - Cualquier estimador sin este atributo → retorna None

    El vector resultante tiene longitud == n_features y está normalizado
    de forma que suma 1.0 (comportamiento estándar de scikit-learn).
    """
    clf = pipeline.named_steps.get("clf")
    if clf is None:
        return None
    importances = getattr(clf, "feature_importances_", None)
    if importances is None:
        return None
    return np.array(importances, dtype=np.float32)


# ─── Loop de CV ───────────────────────────────────────────────────────────────

def cross_validate_pipeline(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    exp_name: str,
) -> Dict:
    """
    Validación cruzada con splits pre-definidos.

    Novedad Sprint 2: retiene `fold_importances` (lista de vectores de
    importancia, uno por fold) para análisis de estabilidad posterior.
    Si el modelo no soporta importancias, fold_importances contiene Nones.
    """
    n_classes    = len(np.unique(y))
    fold_results = []
    fold_importances: List[Optional[np.ndarray]] = []
    all_y_true, all_y_pred, all_y_proba = [], [], []

    t_start = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(
        tqdm(splits, desc=exp_name, leave=False)
    ):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # fit() entrena StandardScaler + modelo SOLO con train (ítem 2 del checklist)
        pipeline.fit(X_tr, y_tr)
        y_pred  = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)

        # ── Métricas del fold ────────────────────────────────────────────────
        acc    = accuracy_score(y_te, y_pred)
        f1_mac = f1_score(y_te, y_pred, average="macro", zero_division=0)

        y_te_bin = label_binarize(y_te, classes=list(range(n_classes)))
        try:
            pr_auc = average_precision_score(y_te_bin, y_proba, average="macro")
        except Exception:
            pr_auc = float("nan")

        fold_results.append({
            "fold":         fold_idx,
            "test_subject": int(np.unique(groups[test_idx])[0]),
            "accuracy":     acc,
            "f1_macro":     f1_mac,
            "pr_auc":       pr_auc,
            "n_train":      len(train_idx),
            "n_test":       len(test_idx),
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        })
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

        # ── Importancias del fold (NUEVO en Sprint 2) ────────────────────────
        fold_importances.append(get_feature_importances(pipeline))

    train_time = time.time() - t_start

    accs = [r["accuracy"] for r in fold_results]
    f1s  = [r["f1_macro"] for r in fold_results]
    prs  = [r["pr_auc"]   for r in fold_results if not np.isnan(r["pr_auc"])]

    summary = {
        "exp_name":           exp_name,
        "n_features":         X.shape[1],
        "n_samples_train":    int(np.mean([r["n_train"] for r in fold_results])),
        "n_samples_test":     int(np.mean([r["n_test"]  for r in fold_results])),
        "accuracy_mean":      float(np.mean(accs)),
        "accuracy_std":       float(np.std(accs)),
        "f1_macro_mean":      float(np.mean(f1s)),
        "f1_macro_std":       float(np.std(f1s)),
        "pr_auc_mean":        float(np.mean(prs)) if prs else float("nan"),
        "pr_auc_std":         float(np.std(prs))  if prs else float("nan"),
        "train_time_s":       train_time,
        "fold_results":       fold_results,
        "fold_importances":   fold_importances,   # NUEVO Sprint 2
        "y_true_all":         np.array(all_y_true),
        "y_pred_all":         np.array(all_y_pred),
        "y_proba_all":        np.array(all_y_proba),
    }

    log.info(
        f"{exp_name:<28} | F1={summary['f1_macro_mean']:.4f}±{summary['f1_macro_std']:.4f} | "
        f"PR-AUC={summary['pr_auc_mean']:.4f} | Time={train_time:.1f}s"
    )
    return summary


# ─── Generador de splits compartidos ─────────────────────────────────────────

def make_shared_splits(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera los splits LOSO una sola vez para reutilizar en todas las variantes.
    Mismo split entre baseline y variantes = comparación A/B justa.
    """
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=n_groups)
    splits   = list(splitter.split(np.zeros_like(y), y, groups=groups))
    log.info(f"Splits generados: {len(splits)} folds, seed={seed}")
    return splits
