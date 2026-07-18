"""
mitigations.py  (Sprint 5)
---------------------------
Implementa mitigaciones post-hoc para los slices problemáticos.

Sigue el protocolo del asesor (sesión 11, slides 8-9):
  - Ajuste de umbral por clase (optimiza F1/Recall)
  - Calibración post-hoc (Platt/Isotónica)
  - Experimento mínimo antes/después con mismo split/seed

Ambas mitigaciones son NO invasivas: no requieren re-entrenar el modelo,
solo transforman sus probabilidades o su regla de decisión.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, f1_score, precision_recall_curve, recall_score,
)


# ─── Mitigación 1: Ajuste de umbral por clase ────────────────────────────────

def find_optimal_thresholds(
    y_true:      np.ndarray,
    y_proba:     np.ndarray,
    n_classes:   int = 3,
    metric:      str = "f1",
    class_priority: Optional[Dict[int, float]] = None,
) -> Dict[int, float]:
    """
    Encuentra el umbral óptimo por clase que maximiza la métrica objetivo
    en un esquema one-vs-rest, opcionalmente ponderado por importancia
    de clase.

    Para tu problema:
      - Clase Alto riesgo (2): priorizar recall (falso negativo = peligroso)
      - Clase Medio (1): balance F1
      - Clase Bajo (0): balance F1
    """
    if class_priority is None:
        class_priority = {0: 1.0, 1: 1.0, 2: 1.5}

    thresholds = {}
    for cls in range(n_classes):
        y_bin = (y_true == cls).astype(int)
        proba = y_proba[:, cls]

        if y_bin.sum() == 0:
            thresholds[cls] = 0.5
            continue

        prec, rec, thr = precision_recall_curve(y_bin, proba)

        # Recortar el último punto (thr tiene 1 menos elemento que prec/rec)
        prec = prec[:-1]
        rec  = rec[:-1]

        if metric == "f1":
            with np.errstate(divide="ignore", invalid="ignore"):
                f1s = 2 * prec * rec / (prec + rec + 1e-12)
            # Ponderar por prioridad de clase
            weight = class_priority.get(cls, 1.0)
            f1s = f1s * weight
            best = np.argmax(f1s) if len(f1s) > 0 else 0
        elif metric == "recall":
            best = np.argmax(rec) if len(rec) > 0 else 0
        else:
            raise ValueError(f"Métrica no soportada: {metric}")

        thresholds[cls] = float(thr[best]) if len(thr) > 0 else 0.5

    return thresholds


def apply_thresholds(
    y_proba:    np.ndarray,
    thresholds: Dict[int, float],
    default_class: int = 0,
) -> np.ndarray:
    """
    Aplica los umbrales por clase a las probabilidades para producir
    predicciones finales.

    Estrategia (para multiclase):
      1. Se considera "activada" cada clase cuya proba supera su umbral
      2. Si varias clases activadas, se toma la de mayor probabilidad
      3. Si ninguna activada, se predice la clase con mayor razón proba/umbral
    """
    n_samples, n_classes = y_proba.shape
    y_pred = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        activated = []
        for cls in range(n_classes):
            if y_proba[i, cls] >= thresholds.get(cls, 0.5):
                activated.append(cls)

        if len(activated) == 1:
            y_pred[i] = activated[0]
        elif len(activated) > 1:
            # Múltiples activaciones: la de mayor probabilidad relativa
            best = max(activated, key=lambda c: y_proba[i, c])
            y_pred[i] = best
        else:
            # Ninguna activada: la clase cuya proba está más cerca de su umbral
            ratios = [y_proba[i, c] / thresholds.get(c, 0.5) for c in range(n_classes)]
            y_pred[i] = int(np.argmax(ratios))

    return y_pred


# ─── Mitigación 2: Calibración isotónica post-hoc ─────────────────────────────

def fit_isotonic_calibrators(
    y_true:  np.ndarray,
    y_proba: np.ndarray,
    n_classes: int = 3,
) -> Dict[int, IsotonicRegression]:
    """
    Ajusta un calibrador isotónico por clase (one-vs-rest).

    La regresión isotónica aprende una función monótona que mapea
    probabilidades sin calibrar a probabilidades calibradas, sin asumir
    una forma paramétrica (a diferencia de Platt scaling).
    """
    calibrators = {}
    for cls in range(n_classes):
        y_bin = (y_true == cls).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            calibrators[cls] = None
            continue
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(y_proba[:, cls], y_bin)
        calibrators[cls] = iso
    return calibrators


def apply_isotonic_calibration(
    y_proba:     np.ndarray,
    calibrators: Dict[int, IsotonicRegression],
) -> np.ndarray:
    """
    Aplica los calibradores isotónicos y re-normaliza las probabilidades
    para que sumen a 1 por fila.
    """
    n_samples, n_classes = y_proba.shape
    y_cal = np.zeros_like(y_proba)

    for cls in range(n_classes):
        if calibrators.get(cls) is not None:
            y_cal[:, cls] = calibrators[cls].predict(y_proba[:, cls])
        else:
            y_cal[:, cls] = y_proba[:, cls]

    # Renormalizar por fila
    row_sums = y_cal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return y_cal / row_sums


# ─── Experimento mínimo antes vs después (slide 9) ────────────────────────────

def compare_before_after(
    y_true:        np.ndarray,
    y_pred_before: np.ndarray,
    y_pred_after:  np.ndarray,
    y_proba_before: Optional[np.ndarray] = None,
    y_proba_after:  Optional[np.ndarray] = None,
    n_classes:      int = 3,
) -> Dict:
    """
    Comparación global baseline vs mitigación.

    Retorna dict con métricas antes/después y Δ para cada.
    """
    from sklearn.preprocessing import label_binarize

    result = {}

    # F1-Macro global
    f1_before = f1_score(y_true, y_pred_before, average="macro", zero_division=0)
    f1_after  = f1_score(y_true, y_pred_after,  average="macro", zero_division=0)
    result["f1_macro_before"] = float(f1_before)
    result["f1_macro_after"]  = float(f1_after)
    result["f1_macro_delta"]  = float(f1_after - f1_before)

    # Recall por clase
    result["recall_by_class"] = {}
    for cls in range(n_classes):
        r_before = recall_score(
            y_true, y_pred_before, labels=[cls], average="macro", zero_division=0
        )
        r_after = recall_score(
            y_true, y_pred_after, labels=[cls], average="macro", zero_division=0
        )
        result["recall_by_class"][cls] = {
            "before": float(r_before),
            "after":  float(r_after),
            "delta":  float(r_after - r_before),
        }

    # Brier score (si tenemos probabilidades)
    if y_proba_before is not None and y_proba_after is not None:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        brier_before, brier_after = [], []
        for cls in range(n_classes):
            if y_bin[:, cls].sum() > 0:
                brier_before.append(
                    brier_score_loss(y_bin[:, cls], y_proba_before[:, cls])
                )
                brier_after.append(
                    brier_score_loss(y_bin[:, cls], y_proba_after[:, cls])
                )
        if brier_before:
            result["brier_before"] = float(np.mean(brier_before))
            result["brier_after"]  = float(np.mean(brier_after))
            result["brier_delta"]  = result["brier_after"] - result["brier_before"]

    return result


def validate_hypothesis(
    comparison:   Dict,
    hypothesis:   str,
    target_class: Optional[int] = None,
    min_recall_gain: float = 0.05,
    max_f1_loss:     float = 0.01,
    max_brier_gain:  float = -0.02,  # negativo = reducción
) -> Dict:
    """
    Valida hipótesis específicas del sprint (confirmadas por el usuario):

    H1 (umbral): "El ajuste de umbral por clase mejora el recall de la clase
                  'Alto' en al menos +0.05, sin degradar el F1-Macro global
                  más de 0.01."

    H2 (calibración): "La calibración isotónica reduce el Brier score global
                       en al menos 0.02 sin degradar el F1-Macro."
    """
    if hypothesis == "H1":
        cls = target_class if target_class is not None else 2
        recall_delta = comparison["recall_by_class"][cls]["delta"]
        f1_delta     = comparison["f1_macro_delta"]

        cond_recall = recall_delta >= min_recall_gain
        cond_f1     = f1_delta >= -max_f1_loss

        return {
            "hypothesis":     hypothesis,
            "confirmed":      bool(cond_recall and cond_f1),
            "recall_delta":   recall_delta,
            "recall_pass":    bool(cond_recall),
            "f1_delta":       f1_delta,
            "f1_pass":        bool(cond_f1),
            "threshold_recall": min_recall_gain,
            "threshold_f1":     -max_f1_loss,
        }

    elif hypothesis == "H2":
        brier_delta = comparison.get("brier_delta", 0.0)
        f1_delta    = comparison["f1_macro_delta"]

        cond_brier = brier_delta <= max_brier_gain
        cond_f1    = f1_delta >= -max_f1_loss

        return {
            "hypothesis":    hypothesis,
            "confirmed":     bool(cond_brier and cond_f1),
            "brier_delta":   brier_delta,
            "brier_pass":    bool(cond_brier),
            "f1_delta":      f1_delta,
            "f1_pass":       bool(cond_f1),
            "threshold_brier": max_brier_gain,
            "threshold_f1":    -max_f1_loss,
        }
    else:
        raise ValueError(f"Hipótesis desconocida: {hypothesis}")
