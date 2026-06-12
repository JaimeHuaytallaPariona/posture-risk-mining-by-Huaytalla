"""
ablations.py  (Fase 1 — Pre-presentación parcial)
--------------------------------------------------
Implementa las 3 ablaciones adicionales para enriquecer la sección
de resultados del informe parcial (slide 19 de la sesión 8):

  1. Ablación de validación   : GroupKFold-LOSO vs StratifiedKFold
     → demuestra empíricamente el leakage por sujeto que evitamos
  2. Ablación de escalado     : XGBoost con vs sin StandardScaler
     → demuestra si el escalado aporta a un modelo basado en árboles
  3. Ablación por segmento    : muñeca / pecho / tobillo / completo
     → identifica qué segmento corporal aporta más a la clasificación

Definición del asesor (slide 19):
  "Una ablación es un experimento controlado donde cambias solo un
   componente del pipeline (o lo apagás) y mantienes todo lo demás
   idéntico (mismo split/seed/métrica)."

Modelo usado en todas las ablaciones: configuración ganadora del Sprint 3 HPO
(bayesian/trial 23) para que las comparaciones sean consistentes con el
modelo adoptado.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ─── Configuración ganadora del HPO (Sprint 3) ────────────────────────────────
# Trial bayesian_023, F1-Macro LOSO=0.8171 ± 0.2047
HPO_WINNER_PARAMS = {
    "learning_rate":    0.12265,
    "max_depth":        6,
    "min_child_weight": 0.04027,
    "subsample":        0.72192,
    "colsample_bytree": 0.89066,
}

HPO_FIXED_PARAMS = {
    "n_estimators":   200,           # sin early stopping en LOSO final
    "tree_method":    "hist",
    "eval_metric":    "mlogloss",
    "random_state":   42,
    "n_jobs":         -1,
    "verbosity":      0,
}


# ─── Constructores de pipeline ────────────────────────────────────────────────

def build_xgb_with_scaler(params: Optional[dict] = None, seed: int = 42) -> Pipeline:
    """Pipeline XGBoost CON StandardScaler — modelo de referencia (adoptado)."""
    if params is None:
        params = HPO_WINNER_PARAMS
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(**HPO_FIXED_PARAMS, **params)),
    ])


def build_xgb_no_scaler(params: Optional[dict] = None, seed: int = 42) -> Pipeline:
    """Pipeline XGBoost SIN StandardScaler — para ablación de escalado."""
    if params is None:
        params = HPO_WINNER_PARAMS
    return Pipeline([
        ("clf", XGBClassifier(**HPO_FIXED_PARAMS, **params)),
    ])


# ─── Generadores de splits ────────────────────────────────────────────────────

def make_loso_splits(y: np.ndarray, groups: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """GroupKFold por sujeto = Leave-One-Subject-Out."""
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=n_groups)
    return list(splitter.split(np.zeros_like(y), y, groups=groups))


def make_stratified_splits(
    y: np.ndarray,
    n_splits: int = 9,
    seed: int   = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    StratifiedKFold con shuffle aleatorio.

    Ignora deliberadamente el grupo (subject_id) → permite que ventanas
    del mismo sujeto aparezcan en train y test. Esto es leakage por
    identidad biométrica, y la ablación lo demuestra cuantitativamente.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros_like(y), y))


# ─── Subset de columnas por segmento corporal ─────────────────────────────────

# Layout del tensor de features (297 columnas):
#   Por cada feature_block (11 en total: 6 temporal + 5 spectral):
#     - cols [block*27 + 0 : block*27 + 9]  → 9 canales de "hand"
#     - cols [block*27 + 9 : block*27 + 18] → 9 canales de "chest"
#     - cols [block*27 + 18: block*27 + 27] → 9 canales de "ankle"
#
# Verificación: 11 feature_blocks × 9 canales = 99 features por segmento
#               99 × 3 segmentos = 297 features totales ✓

SEGMENT_OFFSETS = {
    "hand":  (0,  9),
    "chest": (9,  18),
    "ankle": (18, 27),
}

N_FEATURE_BLOCKS  = 11   # 6 temporal + 5 spectral
N_CHANNELS_TOTAL  = 27   # 9 × 3 segmentos


def get_segment_columns(segment_name: str) -> List[int]:
    """
    Retorna los índices de columnas correspondientes a un segmento corporal.

    Total: 11 feature_blocks × 9 canales = 99 columnas por segmento.
    """
    if segment_name not in SEGMENT_OFFSETS:
        raise ValueError(f"Segmento desconocido: {segment_name}. "
                         f"Opciones: {list(SEGMENT_OFFSETS)}")
    start, end = SEGMENT_OFFSETS[segment_name]
    cols = []
    for block in range(N_FEATURE_BLOCKS):
        block_start = block * N_CHANNELS_TOTAL
        cols.extend(range(block_start + start, block_start + end))
    return cols


def filter_X_by_segment(X: np.ndarray, segment_name: str) -> np.ndarray:
    """Devuelve X con solo las columnas del segmento indicado."""
    cols = get_segment_columns(segment_name)
    return X[:, cols]


# ─── Verificación del leakage por sujeto entre folds ──────────────────────────

def count_subject_leakage(
    splits: List[Tuple[np.ndarray, np.ndarray]],
    subject_ids: np.ndarray,
) -> Dict[str, int]:
    """
    Cuantifica el leakage por sujeto en un esquema de splits.

    Retorna:
      - n_folds_with_leakage : folds donde un sujeto aparece en train Y test
      - total_leak_instances : suma total de sujetos compartidos en todos los folds
    """
    n_folds_with_leak = 0
    total_leak        = 0
    for train_idx, test_idx in splits:
        train_subj = set(subject_ids[train_idx])
        test_subj  = set(subject_ids[test_idx])
        overlap    = train_subj & test_subj
        if overlap:
            n_folds_with_leak += 1
            total_leak        += len(overlap)
    return {
        "n_folds_with_leakage":  n_folds_with_leak,
        "total_leak_instances":  total_leak,
        "n_folds":               len(splits),
    }
