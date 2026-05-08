"""
runner.py — VERSIÓN OPTIMIZADA
Var B usa LinearSVC sin CalibratedClassifierCV (evita 3x overhead).
Para AUC y PR, usa decision_function directamente (no requiere probabilidades reales).
"""

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import mlflow
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from scipy.special import softmax
from tqdm import tqdm

warnings.filterwarnings("ignore")

VARIANTS: Dict[str, dict] = {
    "baseline": {
        "name":          "Baseline RF (temporal+espectral)",
        "feature_set":   "all",
        "classifier":    "rf",
        "window_ms":     200,
        "description":   "Random Forest con features temporales + espectrales, ventana 200 ms",
    },
    "var_a_only_temporal": {
        "name":          "Var A — Solo features temporales",
        "feature_set":   "temporal_only",
        "classifier":    "rf",
        "window_ms":     200,
        "description":   "Random Forest sin features espectrales (FFT eliminada)",
    },
    "var_b_svm_linear": {
        "name":          "Var B — SVM Lineal",
        "feature_set":   "all",
        "classifier":    "svm_linear",
        "window_ms":     200,
        "description":   "SVM lineal (LinearSVC) — alternativa rápida y embebible",
    },
}


def load_dataset(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with h5py.File(h5_path, "r") as f:
        X           = f["X"][:]
        y           = f["y"][:]
        subject_ids = f["subject_ids"][:]
        meta        = dict(f.attrs)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, subject_ids, meta


def filter_feature_set(X: np.ndarray, feature_set: str, n_channels: int = 27) -> np.ndarray:
    n_temporal = 6 * n_channels
    if feature_set == "all":
        return X
    elif feature_set == "temporal_only":
        return X[:, :n_temporal]
    elif feature_set == "spectral_only":
        return X[:, n_temporal:]
    else:
        raise ValueError(f"feature_set desconocido: {feature_set}")


def build_classifier(name: str, seed: int):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators     = 200,
            max_depth        = 20,
            min_samples_split= 5,
            min_samples_leaf = 2,
            class_weight     = "balanced",
            n_jobs           = -1,
            random_state     = seed,
        )
    elif name == "svm_linear":
        # LinearSVC puro: sin calibración (rápido)
        # Para AUC/PR usaremos decision_function + softmax
        return LinearSVC(
            C            = 1.0,
            class_weight = "balanced",
            max_iter     = 3000,
            dual         = "auto",
            random_state = seed,
        )
    else:
        raise ValueError(f"Clasificador desconocido: {name}")


def get_proba(clf, X: np.ndarray) -> np.ndarray:
    """
    Obtiene probabilidades para AUC/PR independiente del tipo de clasificador.
    - Si tiene predict_proba: úsalo directamente.
    - Si solo tiene decision_function: aplica softmax para convertir scores → probas.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    elif hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
        # Para multiclase, decision_function devuelve (n, n_classes)
        if scores.ndim == 1:
            # Caso binario degenerado, no aplica aquí
            scores = np.column_stack([-scores, scores])
        return softmax(scores, axis=1)
    else:
        raise AttributeError("Clasificador sin predict_proba ni decision_function")


def evaluate_loso(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    classifier_name: str,
    seed: int = 42,
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    unique_subjects = np.unique(subject_ids)
    n_classes       = len(np.unique(y))

    fold_results, all_y_true, all_y_pred, all_y_proba = [], [], [], []
    t_total = time.time()

    for test_sub in tqdm(unique_subjects, desc=f"LOSO [{classifier_name}]"):
        train_mask = subject_ids != test_sub
        test_mask  = subject_ids == test_sub

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask],  y[test_mask]

        train_subjects_in_fold = set(subject_ids[train_mask].tolist())
        test_subjects_in_fold  = set(subject_ids[test_mask].tolist())
        assert train_subjects_in_fold.isdisjoint(test_subjects_in_fold), \
            f"LEAKAGE DETECTADO en fold {test_sub}"

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        clf      = build_classifier(classifier_name, seed)
        t0_fold  = time.time()
        clf.fit(X_tr, y_tr)
        t_fit    = time.time() - t0_fold

        y_pred  = clf.predict(X_te)
        y_proba = get_proba(clf, X_te)

        acc    = accuracy_score(y_te, y_pred)
        f1_mac = f1_score(y_te, y_pred, average="macro",    zero_division=0)
        f1_w   = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        y_bin = label_binarize(y_te, classes=list(range(n_classes)))
        try:
            auc    = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
            avg_pr = average_precision_score(y_bin, y_proba, average="macro")
        except Exception:
            auc, avg_pr = float("nan"), float("nan")

        fold_results.append({
            "test_subject": int(test_sub),
            "accuracy":     float(acc),
            "f1_macro":     float(f1_mac),
            "f1_weighted":  float(f1_w),
            "auc_ovr":      float(auc),
            "avg_precision":float(avg_pr),
            "fit_time_s":   float(t_fit),
            "n_test":       int(len(y_te)),
        })

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

    t_total = time.time() - t_total

    accs    = [r["accuracy"]      for r in fold_results]
    f1s     = [r["f1_macro"]      for r in fold_results]
    aucs    = [r["auc_ovr"]       for r in fold_results if not np.isnan(r["auc_ovr"])]
    aprs    = [r["avg_precision"] for r in fold_results if not np.isnan(r["avg_precision"])]

    summary = {
        "n_folds":            len(fold_results),
        "accuracy_mean":      float(np.mean(accs)),
        "accuracy_std":       float(np.std(accs)),
        "f1_macro_mean":      float(np.mean(f1s)),
        "f1_macro_std":       float(np.std(f1s)),
        "auc_ovr_mean":       float(np.mean(aucs)) if aucs else float("nan"),
        "auc_ovr_std":        float(np.std(aucs))  if aucs else float("nan"),
        "avg_precision_mean": float(np.mean(aprs)) if aprs else float("nan"),
        "avg_precision_std":  float(np.std(aprs))  if aprs else float("nan"),
        "total_time_s":       float(t_total),
        "folds":              fold_results,
    }
    return summary, np.array(all_y_true), np.array(all_y_pred), np.array(all_y_proba)


def run_variant(variant_key: str, cfg: dict) -> Dict:
    if variant_key not in VARIANTS:
        raise ValueError(f"Variante desconocida: {variant_key}. Opciones: {list(VARIANTS)}")

    variant = VARIANTS[variant_key]
    logger.info("=" * 70)
    logger.info(f"VARIANTE: {variant['name']}")
    logger.info(f"  {variant['description']}")
    logger.info("=" * 70)

    h5_path = Path(cfg["paths"]["processed"]) / "pamap2_features.h5"

    X_full, y, subject_ids, meta = load_dataset(h5_path)
    X = filter_feature_set(X_full, variant["feature_set"])

    logger.info(f"  X shape: {X.shape}  ({variant['feature_set']})")
    logger.info(f"  y shape: {y.shape}  | clases: {np.unique(y, return_counts=True)}")

    mlflow.set_experiment(cfg["logging"]["experiment_name"])
    with mlflow.start_run(run_name=variant_key):
        mlflow.log_param("variant",        variant_key)
        mlflow.log_param("feature_set",    variant["feature_set"])
        mlflow.log_param("classifier",     variant["classifier"])
        mlflow.log_param("window_ms",      variant["window_ms"])
        mlflow.log_param("n_features",     X.shape[1])
        mlflow.log_param("n_subjects",     len(np.unique(subject_ids)))
        mlflow.log_param("validation",     "LOSO")

        summary, y_true, y_pred, y_proba = evaluate_loso(
            X, y, subject_ids, variant["classifier"], cfg["project"]["seed"]
        )

        mlflow.log_metric("accuracy_mean",      summary["accuracy_mean"])
        mlflow.log_metric("accuracy_std",       summary["accuracy_std"])
        mlflow.log_metric("f1_macro_mean",      summary["f1_macro_mean"])
        mlflow.log_metric("f1_macro_std",       summary["f1_macro_std"])
        mlflow.log_metric("auc_ovr_mean",       summary["auc_ovr_mean"])
        mlflow.log_metric("avg_precision_mean", summary["avg_precision_mean"])
        mlflow.log_metric("total_time_s",       summary["total_time_s"])

        logger.info(f"  Accuracy : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        logger.info(f"  F1-Macro : {summary['f1_macro_mean']:.4f} ± {summary['f1_macro_std']:.4f}")
        logger.info(f"  AUC-OvR  : {summary['auc_ovr_mean']:.4f} ± {summary['auc_ovr_std']:.4f}")
        logger.info(f"  AP-Macro : {summary['avg_precision_mean']:.4f}")
        logger.info(f"  Tiempo   : {summary['total_time_s']:.1f}s")

    out_dir = Path(cfg["paths"]["figures"]).parent / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_dir / f"{variant_key}_predictions.npz",
        y_true  = y_true,
        y_pred  = y_pred,
        y_proba = y_proba,
    )
    with open(out_dir / f"{variant_key}_summary.json", "w") as f:
        json.dump({
            "variant":     variant_key,
            "name":        variant["name"],
            "description": variant["description"],
            "config":      variant,
            "summary":     summary,
        }, f, indent=2)

    logger.info(f"  Predicciones: {out_dir / f'{variant_key}_predictions.npz'}")
    logger.info(f"  Resumen     : {out_dir / f'{variant_key}_summary.json'}")

    return summary


def run_all_variants(cfg: dict) -> Dict[str, dict]:
    results = {}
    for key in ["baseline", "var_a_only_temporal", "var_b_svm_linear"]:
        try:
            results[key] = run_variant(key, cfg)
        except Exception as e:
            logger.error(f"Variante {key} falló: {e}")
            results[key] = None
    return results


def main():
    parser = argparse.ArgumentParser(description="Runner de experimentos A/B con MLflow")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--variant", default="baseline",
                        help="Una de: baseline, var_a_only_temporal, var_b_svm_linear, all")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.variant == "all":
        run_all_variants(cfg)
    else:
        run_variant(args.variant, cfg)


if __name__ == "__main__":
    main()
