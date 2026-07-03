"""
slicing.py  (Sprint 5)
-----------------------
Análisis de errores por slices siguiendo el protocolo del Ing. Glen Rodríguez
(sesión 11: "Análisis de errores y slicing").

Reglas del asesor (slide 3):
  "cada slice debe tener tamaño mínimo e Intervalo de Confianza
   (normales, o vía Bootstrap si el slice es pequeño o de distribución desconocida)."

Este módulo provee las 4 dimensiones de slicing acordadas:
  1. Por sujeto (subject_id)
  2. Por clase de riesgo (bajo, medio, alto)
  3. Por actividad PAMAP2 (12 actividades)
  4. Por tipo de ventana (estable vs transición)

Para cada slice se calcula: F1-Macro, Recall por clase, Brier score, y matriz de
confusión, con intervalos de confianza al 95% via bootstrap de 1000 iteraciones.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    recall_score,
)
from sklearn.preprocessing import label_binarize


# ─── Bootstrap para intervalos de confianza ───────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Estima intervalo de confianza mediante bootstrap.

    Retorna: (métrica_puntual, ci_lower, ci_upper) al nivel de confianza dado.

    Bootstrap con reemplazo sobre índices de las muestras, no sobre folds.
    Es la técnica recomendada por el asesor cuando el slice es pequeño
    o de distribución desconocida (slide 3).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    if n < 10:
        # slice demasiado pequeño para bootstrap confiable
        point = metric_fn(y_true, y_pred)
        return float(point), float("nan"), float("nan")

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
            scores.append(s)
        except Exception:
            continue

    scores = np.array(scores)
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(scores, alpha * 100))
    ci_upper = float(np.percentile(scores, (1 - alpha) * 100))
    point    = float(metric_fn(y_true, y_pred))
    return point, ci_lower, ci_upper


# ─── Métricas por slice ───────────────────────────────────────────────────────

def compute_slice_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    n_classes: int = 3,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Calcula todas las métricas relevantes para un slice.

    Retorna:
      - n           : tamaño del slice
      - f1_macro    : punto + IC bootstrap
      - recall_by_class : recall por cada clase (con IC si hay casos)
      - confusion   : matriz de confusión
      - class_dist  : distribución de clases en el slice
      - brier       : Brier score global si y_proba está disponible
    """
    n = len(y_true)
    if n == 0:
        return {"n": 0, "empty": True}

    # F1-Macro con IC
    f1_point, f1_lo, f1_hi = bootstrap_ci(
        y_true, y_pred,
        metric_fn=lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        n_bootstrap=n_bootstrap, seed=seed,
    )

    # Recall por clase con IC
    recall_by_class = {}
    for cls in range(n_classes):
        mask_cls = (y_true == cls)
        n_cls = int(mask_cls.sum())
        if n_cls == 0:
            recall_by_class[cls] = {"n": 0, "recall": float("nan"),
                                    "ci_lo": float("nan"), "ci_hi": float("nan")}
            continue
        rec_point, rec_lo, rec_hi = bootstrap_ci(
            y_true, y_pred,
            metric_fn=lambda yt, yp: recall_score(
                yt, yp, labels=[cls], average="macro", zero_division=0
            ),
            n_bootstrap=n_bootstrap, seed=seed + cls,
        )
        recall_by_class[cls] = {
            "n":     n_cls,
            "recall": rec_point,
            "ci_lo":  rec_lo,
            "ci_hi":  rec_hi,
        }

    # Matriz de confusión
    labels_all = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    # Distribución de clases
    class_dist = {int(c): int((y_true == c).sum()) for c in labels_all}

    # Brier score global (multiclase promediado)
    brier_val = None
    if y_proba is not None and len(y_proba) == n:
        try:
            y_bin = label_binarize(y_true, classes=labels_all)
            briers = []
            for cls in range(n_classes):
                if class_dist[cls] > 0:
                    briers.append(brier_score_loss(y_bin[:, cls], y_proba[:, cls]))
            brier_val = float(np.mean(briers)) if briers else None
        except Exception:
            brier_val = None

    return {
        "n":                n,
        "f1_macro":         f1_point,
        "f1_ci_lo":         f1_lo,
        "f1_ci_hi":         f1_hi,
        "f1_ci_width":      f1_hi - f1_lo if not np.isnan(f1_lo) else float("nan"),
        "recall_by_class":  recall_by_class,
        "confusion":        cm.tolist(),
        "class_dist":       class_dist,
        "brier":            brier_val,
    }


# ─── Slicing por sujeto ───────────────────────────────────────────────────────

def slice_by_subject(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    y_proba:     np.ndarray,
    subject_ids: np.ndarray,
    **kwargs,
) -> Dict[int, Dict]:
    """Slice: uno por cada sujeto único."""
    results = {}
    for sid in sorted(np.unique(subject_ids)):
        mask = (subject_ids == sid)
        results[int(sid)] = compute_slice_metrics(
            y_true[mask], y_pred[mask],
            y_proba[mask] if y_proba is not None else None,
            **kwargs,
        )
    log.info(f"Slicing por sujeto: {len(results)} slices calculados")
    return results


# ─── Slicing por clase de riesgo ──────────────────────────────────────────────

RISK_NAMES = {0: "bajo", 1: "medio", 2: "alto"}


def slice_by_class(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray,
    n_classes: int = 3,
    **kwargs,
) -> Dict[str, Dict]:
    """
    Slice: rendimiento del modelo sobre ventanas cuya etiqueta verdadera
    es cada clase específica. Útil para detectar clases con bajo recall.
    """
    results = {}
    for cls in range(n_classes):
        mask = (y_true == cls)
        n = int(mask.sum())
        if n == 0:
            continue
        name = RISK_NAMES.get(cls, f"clase_{cls}")
        results[name] = compute_slice_metrics(
            y_true[mask], y_pred[mask],
            y_proba[mask] if y_proba is not None else None,
            n_classes=n_classes, **kwargs,
        )
        results[name]["class_id"] = cls
    log.info(f"Slicing por clase: {len(results)} slices calculados")
    return results


# ─── Slicing por actividad PAMAP2 ─────────────────────────────────────────────

ACTIVITY_NAMES = {
    1:  "tumbado",
    2:  "sentado",
    3:  "de_pie",
    4:  "caminar",
    5:  "correr",
    6:  "ciclismo",
    7:  "caminata_nordica",
    12: "escaleras_arriba",
    13: "escaleras_abajo",
    16: "aspirar",
    17: "planchar",
    24: "saltar",
}


def slice_by_activity(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    y_proba:      np.ndarray,
    activity_ids: np.ndarray,
    **kwargs,
) -> Dict[str, Dict]:
    """Slice: uno por cada actividad PAMAP2."""
    results = {}
    for aid in sorted(np.unique(activity_ids)):
        mask = (activity_ids == aid)
        name = ACTIVITY_NAMES.get(int(aid), f"actividad_{aid}")
        results[name] = compute_slice_metrics(
            y_true[mask], y_pred[mask],
            y_proba[mask] if y_proba is not None else None,
            **kwargs,
        )
        results[name]["activity_id"] = int(aid)
    log.info(f"Slicing por actividad: {len(results)} slices calculados")
    return results


# ─── Slicing por tipo de ventana (estable vs transición) ──────────────────────

def compute_transition_mask(
    activity_ids: np.ndarray,
    subject_ids: np.ndarray,
    window_span: int = 3,
) -> np.ndarray:
    """
    Marca ventanas como transición si están dentro de `window_span` ventanas
    antes o después de un cambio de actividad dentro del mismo sujeto.

    Retorna array booleano: True = transición, False = estable.
    """
    is_transition = np.zeros(len(activity_ids), dtype=bool)

    for sid in np.unique(subject_ids):
        mask_sub = (subject_ids == sid)
        idx_sub  = np.where(mask_sub)[0]
        acts_sub = activity_ids[idx_sub]

        # Detectar cambios de actividad dentro del sujeto
        changes = np.where(np.diff(acts_sub) != 0)[0]

        for change_pos in changes:
            start = max(0, change_pos - window_span + 1)
            end   = min(len(acts_sub), change_pos + window_span + 1)
            is_transition[idx_sub[start:end]] = True

    return is_transition


def slice_by_transition(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    y_proba:      np.ndarray,
    activity_ids: np.ndarray,
    subject_ids:  np.ndarray,
    window_span:  int = 3,
    **kwargs,
) -> Dict[str, Dict]:
    """Slice: ventanas cerca de cambios de actividad vs estables."""
    is_trans = compute_transition_mask(activity_ids, subject_ids, window_span)

    results = {}
    for label, mask in [("estable", ~is_trans), ("transicion", is_trans)]:
        n = int(mask.sum())
        if n == 0:
            continue
        results[label] = compute_slice_metrics(
            y_true[mask], y_pred[mask],
            y_proba[mask] if y_proba is not None else None,
            **kwargs,
        )
    log.info(f"Slicing por transición: estables={int((~is_trans).sum())}, "
             f"transiciones={int(is_trans.sum())}")
    return results


# ─── Identificación de slices problemáticos ───────────────────────────────────

def identify_problematic_slices(
    slice_results: Dict[str, Dict],
    global_f1: float,
    dimension_name: str,
    min_gap: float = 0.05,
    min_n: int = 30,
) -> List[Dict]:
    """
    Un slice se considera problemático si su F1-Macro puntual es al menos
    `min_gap` por debajo del F1 global, y tiene al menos `min_n` muestras
    para que el hallazgo sea estable.

    Retorna lista ordenada por gap descendente (peores primero).
    """
    problematic = []
    for name, res in slice_results.items():
        if res.get("empty") or res.get("n", 0) < min_n:
            continue
        f1_slice = res.get("f1_macro", float("nan"))
        if np.isnan(f1_slice):
            continue
        gap = global_f1 - f1_slice
        if gap >= min_gap:
            problematic.append({
                "dimension":  dimension_name,
                "slice_name": name,
                "n":          res["n"],
                "f1_macro":   f1_slice,
                "f1_ci_lo":   res.get("f1_ci_lo"),
                "f1_ci_hi":   res.get("f1_ci_hi"),
                "gap":        gap,
                "recall_by_class": res["recall_by_class"],
                "confusion":  res["confusion"],
                "class_dist": res["class_dist"],
            })

    problematic.sort(key=lambda x: x["gap"], reverse=True)
    return problematic
