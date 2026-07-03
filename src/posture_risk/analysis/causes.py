"""
causes.py  (Sprint 5)
----------------------
Análisis de causas probables de errores por slice.

Sigue el protocolo del asesor (sesión 11, slide 7):
  "Permutation importance dentro del slice problemático vs el resto de los casos.
   Busca features gatillo que disparan errores FP, FN.
   Muestra 2-3 ejemplos representativos (inputs → predicción → error → explicación)."

Este módulo provee:
  - Permutation importance por slice
  - Comparación de importancia dentro vs fuera del slice
  - Extracción de ejemplos representativos con errores
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


# ─── Permutation importance rápida por slice ──────────────────────────────────

def permutation_importance_slice(
    pipeline:      Pipeline,
    X:             np.ndarray,
    y:             np.ndarray,
    slice_mask:    np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_repeats:     int = 5,
    top_k:         int = 20,
    seed:          int = 42,
) -> Dict:
    """
    Calcula permutation importance restringida al slice indicado.

    Método: para cada feature, se permuta su columna dentro de las muestras
    del slice y se mide cuánto cae el F1-Macro. Features cuya permutación
    degrada más el rendimiento son las más importantes para ese slice.

    Retorna diccionario con top_k features ordenadas.
    """
    X_slice = X[slice_mask]
    y_slice = y[slice_mask]
    n = len(X_slice)

    if n < 20:
        log.warning(f"Slice muy pequeño ({n}) para permutation importance confiable")
        return {"top_features": [], "n_used": n}

    # Baseline: predicción normal sobre el slice
    y_pred_base = pipeline.predict(X_slice)
    f1_base = f1_score(y_slice, y_pred_base, average="macro", zero_division=0)

    rng = np.random.RandomState(seed)
    n_features = X.shape[1]

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    importances = np.zeros(n_features, dtype=np.float32)

    for j in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_slice.copy()
            perm_idx = rng.permutation(n)
            X_perm[:, j] = X_slice[perm_idx, j]
            y_pred_perm = pipeline.predict(X_perm)
            f1_perm = f1_score(y_slice, y_pred_perm, average="macro", zero_division=0)
            drops.append(f1_base - f1_perm)
        importances[j] = float(np.mean(drops))

    # Ordenar y devolver top_k
    top_idx = np.argsort(importances)[::-1][:top_k]
    top_features = [
        {
            "feature":   feature_names[i] if i < len(feature_names) else f"feat_{i}",
            "importance": float(importances[i]),
            "col_idx":   int(i),
        }
        for i in top_idx
    ]

    return {
        "top_features": top_features,
        "n_used":       n,
        "f1_baseline":  f1_base,
    }


# ─── Comparación de importancias: dentro vs fuera del slice ──────────────────

def compare_importances(
    imp_slice: Dict,
    imp_rest:  Dict,
    top_k:     int = 10,
) -> List[Dict]:
    """
    Compara las importancias del slice problemático con las del resto de casos.
    Identifica features cuyo peso CAMBIA significativamente entre ambos grupos.

    Un cambio grande sugiere una "feature gatillo": algo que en el slice
    tiene un patrón distinto al del resto y por eso el modelo falla ahí.

    Retorna las top_k features con mayor cambio de importancia.
    """
    imp_slice_map = {f["feature"]: f["importance"] for f in imp_slice["top_features"]}
    imp_rest_map  = {f["feature"]: f["importance"] for f in imp_rest["top_features"]}

    all_features = set(imp_slice_map) | set(imp_rest_map)
    diffs = []
    for f in all_features:
        val_slice = imp_slice_map.get(f, 0.0)
        val_rest  = imp_rest_map.get(f, 0.0)
        delta     = val_slice - val_rest
        diffs.append({
            "feature":     f,
            "imp_slice":   val_slice,
            "imp_rest":    val_rest,
            "delta":       delta,
            "abs_delta":   abs(delta),
        })

    diffs.sort(key=lambda x: x["abs_delta"], reverse=True)
    return diffs[:top_k]


# ─── Ejemplos representativos ─────────────────────────────────────────────────

def extract_error_examples(
    X:            np.ndarray,
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    y_proba:      np.ndarray,
    slice_mask:   np.ndarray,
    n_examples:   int = 3,
    error_type:   str = "confident_wrong",
) -> List[Dict]:
    """
    Extrae ejemplos representativos de errores dentro del slice.

    error_type:
      - 'confident_wrong': predicciones equivocadas con alta confianza (peores)
      - 'confused'       : predicciones con probabilidad ambigua (baja confianza)
      - 'high_impact'    : errores donde la clase real es la más crítica (alto riesgo)
    """
    slice_idx = np.where(slice_mask)[0]
    if len(slice_idx) == 0:
        return []

    y_true_s  = y_true[slice_mask]
    y_pred_s  = y_pred[slice_mask]
    y_proba_s = y_proba[slice_mask]
    wrong     = y_true_s != y_pred_s

    if not wrong.any():
        return []

    wrong_local = np.where(wrong)[0]

    if error_type == "confident_wrong":
        # Máxima probabilidad de la clase predicha (confianza del error)
        confidences = np.array([y_proba_s[i, y_pred_s[i]] for i in wrong_local])
        order = np.argsort(confidences)[::-1]
    elif error_type == "confused":
        max_prob = y_proba_s[wrong_local].max(axis=1)
        order = np.argsort(max_prob)
    elif error_type == "high_impact":
        # Priorizar errores donde la clase real es "alto riesgo" (2)
        priority = np.array([2 - y_true_s[i] for i in wrong_local])
        order = np.argsort(priority)
    else:
        order = np.arange(len(wrong_local))

    examples = []
    for k in order[:n_examples]:
        local_idx  = wrong_local[k]
        global_idx = slice_idx[local_idx]
        examples.append({
            "global_idx":  int(global_idx),
            "y_true":      int(y_true_s[local_idx]),
            "y_pred":      int(y_pred_s[local_idx]),
            "probs":       [float(p) for p in y_proba_s[local_idx]],
            "confidence":  float(y_proba_s[local_idx, y_pred_s[local_idx]]),
        })

    return examples
