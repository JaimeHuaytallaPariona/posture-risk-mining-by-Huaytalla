"""
pruning.py  (Sprint 6)
-----------------------
Optimización de latencia mediante poda del ensemble XGBoost.

Estrategia: entrenar variantes del modelo con menos árboles (n_estimators)
y comparar F1-Macro vs latencia. La hipótesis es que reducir el número de
árboles reduce proporcionalmente la latencia de inferencia sin sacrificar
significativamente la calidad predictiva, porque los últimos árboles del
ensemble aportan mejoras marginales decrecientes.

Complementa el early_stopping usado durante el HPO: aquí exploramos
sistemáticamente el trade-off en el modelo ya entrenado.
"""

import copy
import time
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger as log
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from posture_risk.experiments.ablations import HPO_WINNER_PARAMS, HPO_FIXED_PARAMS


def build_pruned_pipeline(
    n_estimators: int,
    params:       dict = None,
    seed:         int  = 42,
) -> Pipeline:
    """Construye un pipeline XGBoost con n_estimators específico."""
    if params is None:
        params = HPO_WINNER_PARAMS
    fixed = {**HPO_FIXED_PARAMS, "n_estimators": n_estimators}
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(**fixed, **params)),
    ])


def evaluate_pruning_variant(
    X:             np.ndarray,
    y:             np.ndarray,
    subject_ids:   np.ndarray,
    n_estimators:  int,
    n_folds:       int  = 3,
    seed:          int  = 42,
    n_latency_reps: int = 200,
) -> Dict:
    """
    Evalúa una variante de poda: entrena con n_estimators y mide F1 + latencia.

    Se usa CV(3) para acelerar (no LOSO completo), consistente con el HPO.
    """
    splitter = GroupKFold(n_splits=n_folds)
    splits   = list(splitter.split(X, y, groups=subject_ids))

    fold_f1s = []
    for train_idx, val_idx in splits:
        pipe = build_pruned_pipeline(n_estimators=n_estimators, seed=seed)
        pipe.fit(X[train_idx], y[train_idx])
        y_pred = pipe.predict(X[val_idx])
        fold_f1s.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))

    f1_mean = float(np.mean(fold_f1s))
    f1_std  = float(np.std(fold_f1s))

    # Medir latencia de inferencia (fit una vez, medir predict)
    pipe_full = build_pruned_pipeline(n_estimators=n_estimators, seed=seed)
    pipe_full.fit(X, y)

    sample = X[0:1]
    # Warmup
    for _ in range(30):
        pipe_full.predict_proba(sample)
    # Medición
    times = []
    for _ in range(n_latency_reps):
        t0 = time.perf_counter()
        _ = pipe_full.predict_proba(sample)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        "n_estimators":  n_estimators,
        "f1_macro_mean": f1_mean,
        "f1_macro_std":  f1_std,
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
        "latency_mean_ms": float(np.mean(times)),
    }


def run_pruning_experiment(
    X:                 np.ndarray,
    y:                 np.ndarray,
    subject_ids:       np.ndarray,
    n_estimator_grid:  List[int] = None,
    n_folds:           int  = 3,
    seed:              int  = 42,
    max_f1_loss:       float = 0.01,
) -> Dict:
    """
    Ejecuta el experimento completo de poda: evalúa múltiples n_estimators
    y selecciona el mínimo que mantiene F1 dentro del umbral aceptable.

    Retorna:
      - variants: resultados para cada n_estimators probado
      - baseline (200 árboles): resultado de referencia
      - selected: la variante recomendada (mínimos árboles con F1 acceptable)
    """
    if n_estimator_grid is None:
        n_estimator_grid = [25, 50, 75, 100, 125, 150, 175, 200]

    log.info(f"Iniciando experimento de poda con grid={n_estimator_grid}")

    variants = {}
    for n_est in n_estimator_grid:
        log.info(f"Evaluando n_estimators={n_est}...")
        variants[n_est] = evaluate_pruning_variant(
            X, y, subject_ids,
            n_estimators=n_est,
            n_folds=n_folds,
            seed=seed,
        )

    # Baseline: 200 árboles (config original del HPO winner)
    baseline_f1 = variants[200]["f1_macro_mean"] if 200 in variants \
                  else max(v["f1_macro_mean"] for v in variants.values())

    # Selección: mínimo n_estimators con F1 dentro del margen
    candidates = [
        (n_est, v) for n_est, v in variants.items()
        if v["f1_macro_mean"] >= baseline_f1 - max_f1_loss
    ]
    if candidates:
        selected_n_est, selected_v = min(candidates, key=lambda t: t[0])
    else:
        # Ningún candidato dentro del margen: quedarse con el mejor F1
        selected_n_est = max(variants.keys(), key=lambda k: variants[k]["f1_macro_mean"])
        selected_v = variants[selected_n_est]

    latency_reduction_pct = 100 * (1 - selected_v["latency_p95_ms"] /
                                       variants[200]["latency_p95_ms"]) \
                            if 200 in variants else None

    return {
        "variants":            variants,
        "baseline_f1":         baseline_f1,
        "baseline_latency_p95": variants[200]["latency_p95_ms"] if 200 in variants else None,
        "selected_n_estimators": selected_n_est,
        "selected_f1":         selected_v["f1_macro_mean"],
        "selected_latency_p95": selected_v["latency_p95_ms"],
        "latency_reduction_pct": latency_reduction_pct,
        "max_f1_loss_threshold": max_f1_loss,
    }
