"""
hparam_tuning.py  (Sprint 3)
------------------------------
Búsqueda de hiperparámetros para XGBoost con estrategia híbrida:
  Fase 1 (trials 1-15)  : Random (calentamiento)
  Fase 2 (trials 16-30) : Bayesian/TPE (Optuna)

Protocolo del Ing. Glen Rodríguez (sesión 7):
  - Mismo split/seed entre todos los trials (comparabilidad)
  - Pruning con MedianPruner (corta trials malos temprano)
  - Early stopping nativo de XGBoost (evita overfitting en cada trial)
  - Logging completo en logs/hpo_runs.csv (trial_id, params, seed,
    split_cfg, metric_mean, metric_std, tiempo)
  - Tracking con MLflow (un run por trial)
  - Config ganadora + 1-2 configs de respaldo

Errores evitados (diapositiva 18):
  - No se cambia split/seed entre trials
  - No se mira el test durante la búsqueda (solo train+val)
  - Se registra toda configuración + métrica
  - Se reporta latencia/costo (tiempo por trial)
"""

import csv
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import yaml
from loguru import logger as log
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score, f1_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    raise ImportError("XGBoost es requerido para este módulo. pip install xgboost")


# ─── Carga de configuración ───────────────────────────────────────────────────

def load_search_config(path: str = "configs/search.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Muestreo del espacio de búsqueda ────────────────────────────────────────

def sample_params(trial: optuna.Trial, cfg: dict) -> dict:
    """
    Muestrea hiperparámetros desde el espacio definido en search.yaml.
    Usa la API define-by-run de Optuna (flexible, condicional si se necesita).
    """
    space = cfg["search_space"]
    fixed = cfg["fixed_params"]

    params = {}
    for hp_name, hp_cfg in space.items():
        hp_type = hp_cfg["type"]
        if hp_type == "loguniform":
            params[hp_name] = trial.suggest_float(
                hp_name, hp_cfg["low"], hp_cfg["high"], log=True
            )
        elif hp_type == "uniform":
            params[hp_name] = trial.suggest_float(
                hp_name, hp_cfg["low"], hp_cfg["high"]
            )
        elif hp_type == "int":
            params[hp_name] = trial.suggest_int(
                hp_name, hp_cfg["low"], hp_cfg["high"]
            )

    # Parámetros fijos
    params["n_estimators"]          = fixed["n_estimators"]
    params["early_stopping_rounds"] = fixed["early_stopping_rounds"]
    params["tree_method"]           = fixed["tree_method"]
    params["eval_metric"]           = fixed["eval_metric"]
    params["verbosity"]             = fixed["verbosity"]

    return params


# ─── Evaluación de un trial con CV 3-fold ────────────────────────────────────

def evaluate_trial(
    trial:    optuna.Trial,
    params:   dict,
    X:        np.ndarray,
    y:        np.ndarray,
    groups:   np.ndarray,
    cfg:      dict,
    seed:     int = 42,
) -> Tuple[float, float, int]:
    """
    Evalúa una configuración con 3-fold GroupKFold.

    Implementa:
      1. Early stopping nativo de XGBoost (eval_set en 15% del train)
      2. Reporte de valores intermedios a Optuna (para pruning)

    Retorna: (f1_macro_mean, f1_macro_std, n_trees_mean)
    """
    n_splits   = cfg["cv_search"]["n_splits"]
    eval_frac  = cfg["fixed_params"]["eval_fraction"]
    n_classes  = len(np.unique(y))
    splitter   = GroupKFold(n_splits=n_splits)
    rng        = np.random.RandomState(seed)

    fold_f1s, fold_trees = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(
        splitter.split(np.zeros_like(y), y, groups=groups)
    ):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # Normalizar (fit solo en train del fold)
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        # Separar eval set para early stopping (15% random del train)
        n_eval  = max(50, int(len(X_tr) * eval_frac))
        eval_ix = rng.choice(len(X_tr), size=n_eval, replace=False)
        mask    = np.ones(len(X_tr), dtype=bool)
        mask[eval_ix] = False

        X_fit, y_fit     = X_tr[mask],     y_tr[mask]
        X_eval, y_eval   = X_tr[eval_ix],  y_tr[eval_ix]

        # Construir y entrenar XGBoost con early stopping
        clf = XGBClassifier(
            learning_rate       = params["learning_rate"],
            max_depth           = params["max_depth"],
            min_child_weight    = params["min_child_weight"],
            subsample           = params["subsample"],
            colsample_bytree    = params["colsample_bytree"],
            n_estimators        = params["n_estimators"],
            early_stopping_rounds = params["early_stopping_rounds"],
            tree_method         = params["tree_method"],
            eval_metric         = params["eval_metric"],
            verbosity           = params["verbosity"],
            random_state        = seed,
            n_jobs              = -1,
        )
        clf.fit(
            X_fit, y_fit,
            eval_set=[(X_eval, y_eval)],
            verbose=False,
        )

        n_trees_used = clf.best_iteration + 1 if clf.best_iteration else params["n_estimators"]
        fold_trees.append(n_trees_used)

        y_pred = clf.predict(X_te)
        f1     = f1_score(y_te, y_pred, average="macro", zero_division=0)
        fold_f1s.append(f1)

        # Reportar valor intermedio a Optuna para pruning
        running_mean = float(np.mean(fold_f1s))
        trial.report(running_mean, step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_f1s)), float(np.std(fold_f1s)), int(np.mean(fold_trees))


# ─── Logger CSV ──────────────────────────────────────────────────────────────

CSV_HEADER = [
    "trial_id", "phase", "params", "seed", "split_cfg",
    "metric_mean", "metric_std", "n_trees_mean",
    "train_time_s", "status", "timestamp",
]


def log_trial_csv(trial_info: dict, csv_path: Path) -> None:
    """Añade un trial al CSV de logs (append-only, diapositiva 8)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if not file_exists:
            writer.writeheader()
        row = {col: trial_info.get(col, "") for col in CSV_HEADER}
        writer.writerow(row)


# ─── Función objetivo de Optuna ───────────────────────────────────────────────

def make_objective(
    X:              np.ndarray,
    y:              np.ndarray,
    groups:         np.ndarray,
    cfg:            dict,
    mlflow_exp_id:  str,
    csv_path:       Path,
):
    """
    Fábrica de función objetivo para Optuna.
    Cierra sobre los datos y configuración para evitar variables globales.
    """
    seed      = cfg["study"]["seed"]
    split_cfg = f"GroupKFold-{cfg['cv_search']['n_splits']}fold"

    def objective(trial: optuna.Trial) -> float:
        t_start = time.time()
        params  = sample_params(trial, cfg)

        # Determinar fase del trial
        phase = "random" if trial.number < cfg["study"]["n_startup_trials"] else "bayes"

        try:
            f1_mean, f1_std, n_trees = evaluate_trial(
                trial, params, X, y, groups, cfg, seed=seed
            )
            status = "complete"
        except optuna.TrialPruned:
            status  = "pruned"
            f1_mean = float("nan")
            f1_std  = float("nan")
            n_trees = 0
            log.debug(f"Trial {trial.number:03d} [{phase}] PRUNED")
            raise

        elapsed = time.time() - t_start

        # ── Logging MLflow ────────────────────────────────────────────────
        with mlflow.start_run(
            run_name       = f"trial_{trial.number:03d}_{phase}",
            experiment_id  = mlflow_exp_id,
            tags           = {"phase": phase, "trial_id": str(trial.number)},
        ):
            # Parámetros del espacio de búsqueda
            hp_log = {k: v for k, v in params.items()
                      if k in cfg["search_space"]}
            mlflow.log_params(hp_log)
            mlflow.log_param("phase",      phase)
            mlflow.log_param("n_trees",    n_trees)
            mlflow.log_param("split_cfg",  split_cfg)

            if not np.isnan(f1_mean):
                mlflow.log_metrics({
                    "f1_macro_mean": f1_mean,
                    "f1_macro_std":  f1_std,
                    "train_time_s":  elapsed,
                    "n_trees_mean":  n_trees,
                })

        # ── Logging CSV ───────────────────────────────────────────────────
        params_str = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in params.items()
            if k in cfg["search_space"]
        )
        log_trial_csv({
            "trial_id":      trial.number,
            "phase":         phase,
            "params":        params_str,
            "seed":          seed,
            "split_cfg":     split_cfg,
            "metric_mean":   f"{f1_mean:.4f}" if not np.isnan(f1_mean) else "NaN",
            "metric_std":    f"{f1_std:.4f}"  if not np.isnan(f1_std)  else "NaN",
            "n_trees_mean":  n_trees,
            "train_time_s":  f"{elapsed:.1f}",
            "status":        status,
            "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, csv_path)

        log.info(
            f"Trial {trial.number:03d} [{phase:6s}] | "
            f"F1={f1_mean:.4f}±{f1_std:.4f} | "
            f"trees={n_trees} | {elapsed:.0f}s | {status}"
        )
        return f1_mean

    return objective


# ─── Runner principal del estudio ─────────────────────────────────────────────

def run_study(
    X:       np.ndarray,
    y:       np.ndarray,
    groups:  np.ndarray,
    cfg:     dict,
) -> optuna.Study:
    """
    Ejecuta el estudio completo de búsqueda de hiperparámetros.

    Fase 1 (trials 0-14):  TPESampler con n_startup_trials=15 → Random
    Fase 2 (trials 15-29): TPESampler activo → Bayesian/TPE

    Retorna el estudio Optuna completo con todos los trials.
    """
    seed       = cfg["study"]["seed"]
    n_trials   = cfg["study"]["n_trials"]
    n_startup  = cfg["study"]["n_startup_trials"]

    # ── Setup MLflow ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["paths"]["mlflow_uri"])
    exp = mlflow.get_experiment_by_name(cfg["paths"]["mlflow_experiment"])
    if exp is None:
        exp_id = mlflow.create_experiment(cfg["paths"]["mlflow_experiment"])
    else:
        exp_id = exp.experiment_id
    log.info(f"MLflow experiment: {cfg['paths']['mlflow_experiment']} (id={exp_id})")

    # ── Crear estudio Optuna ──────────────────────────────────────────────
    sampler = TPESampler(
        seed             = seed,
        n_startup_trials = n_startup,   # primeros n_startup = Random
    )
    pruner = MedianPruner(
        n_startup_trials = cfg["pruner"]["n_startup_trials"],
        n_warmup_steps   = cfg["pruner"]["n_warmup_steps"],
    )
    study = optuna.create_study(
        study_name = cfg["study"]["name"],
        direction  = cfg["study"]["direction"],
        sampler    = sampler,
        pruner     = pruner,
    )

    csv_path = Path(cfg["paths"]["log_csv"])

    log.info("=" * 65)
    log.info(f"BÚSQUEDA DE HIPERPARÁMETROS — Sprint 3")
    log.info(f"  Presupuesto  : {n_trials} trials totales")
    log.info(f"  Fase Random  : trials 0–{n_startup-1}")
    log.info(f"  Fase Bayes   : trials {n_startup}–{n_trials-1}")
    log.info(f"  Espacio      : {len(cfg['search_space'])} hiperparámetros")
    log.info(f"  CV búsqueda  : {cfg['cv_search']['n_splits']}-fold GroupKFold")
    log.info(f"  Pruner       : MedianPruner")
    log.info(f"  Early stop   : {cfg['fixed_params']['early_stopping_rounds']} rondas")
    log.info("=" * 65)

    objective = make_objective(X, y, groups, cfg, exp_id, csv_path)

    study.optimize(
        objective,
        n_trials        = n_trials,
        timeout         = cfg["study"].get("timeout_per_trial", None),
        show_progress_bar = True,
        catch           = (Exception,),
    )

    # ── Guardar estudio completo ──────────────────────────────────────────
    study_path = Path(cfg["paths"]["optuna_study"])
    study_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(study, study_path)

    # Resumen del estudio
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    log.info("\n" + "=" * 65)
    log.info("BÚSQUEDA COMPLETADA")
    log.info(f"  Trials completados : {len(completed)}")
    log.info(f"  Trials podados     : {len(pruned)}")
    log.info(f"  Mejor F1-Macro     : {study.best_value:.4f}")
    log.info(f"  Mejor trial        : #{study.best_trial.number}")
    log.info(f"  Mejores params     : {study.best_params}")
    log.info("=" * 65)

    return study


# ─── Evaluación final con LOSO completo ──────────────────────────────────────

def evaluate_with_loso(
    params:  dict,
    X:       np.ndarray,
    y:       np.ndarray,
    groups:  np.ndarray,
    cfg:     dict,
    label:   str = "XGB_tuned",
) -> dict:
    """
    Re-evalúa la config ganadora con LOSO completo (9-fold) para
    obtener métricas comparables con el Sprint 2.
    """
    seed      = cfg["study"]["seed"]
    n_splits  = cfg["final_evaluation"]["n_splits"]
    n_classes = len(np.unique(y))
    splitter  = GroupKFold(n_splits=n_splits)
    rng       = np.random.RandomState(seed)
    eval_frac = cfg["fixed_params"]["eval_fraction"]

    fold_f1s, fold_pr, all_y_true, all_y_pred, all_y_proba = [], [], [], [], []
    t_start = time.time()

    for train_idx, test_idx in splitter.split(np.zeros_like(y), y, groups=groups):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        n_eval  = max(50, int(len(X_tr) * eval_frac))
        eval_ix = rng.choice(len(X_tr), size=n_eval, replace=False)
        mask    = np.ones(len(X_tr), dtype=bool)
        mask[eval_ix] = False

        clf = XGBClassifier(
            learning_rate         = params["learning_rate"],
            max_depth             = params["max_depth"],
            min_child_weight      = params["min_child_weight"],
            subsample             = params["subsample"],
            colsample_bytree      = params["colsample_bytree"],
            n_estimators          = params["n_estimators"],
            early_stopping_rounds = params["early_stopping_rounds"],
            tree_method           = params["tree_method"],
            eval_metric           = params["eval_metric"],
            verbosity             = params["verbosity"],
            random_state          = seed,
            n_jobs                = -1,
        )
        clf.fit(X_tr[mask], y_tr[mask],
                eval_set=[(X_tr[eval_ix], y_tr[eval_ix])],
                verbose=False)

        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)

        f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        fold_f1s.append(f1)

        y_te_bin = label_binarize(y_te, classes=list(range(n_classes)))
        try:
            pr = average_precision_score(y_te_bin, y_proba, average="macro")
        except Exception:
            pr = float("nan")
        fold_pr.append(pr)

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_proba.extend(y_proba.tolist())

    elapsed  = time.time() - t_start
    pr_valid = [v for v in fold_pr if not np.isnan(v)]

    result = {
        "exp_name":       label,
        "f1_macro_mean":  float(np.mean(fold_f1s)),
        "f1_macro_std":   float(np.std(fold_f1s)),
        "pr_auc_mean":    float(np.mean(pr_valid)) if pr_valid else float("nan"),
        "pr_auc_std":     float(np.std(pr_valid))  if pr_valid else float("nan"),
        "train_time_s":   elapsed,
        "params":         params,
        "y_true_all":     np.array(all_y_true),
        "y_pred_all":     np.array(all_y_pred),
        "y_proba_all":    np.array(all_y_proba),
    }

    log.info(
        f"{label} LOSO | F1={result['f1_macro_mean']:.4f}±{result['f1_macro_std']:.4f} | "
        f"PR-AUC={result['pr_auc_mean']:.4f} | {elapsed:.0f}s"
    )
    return result


# ─── Guardar el mejor modelo ──────────────────────────────────────────────────

def save_best_model(
    params:  dict,
    X:       np.ndarray,
    y:       np.ndarray,
    cfg:     dict,
    output_path: Path,
) -> None:
    """
    Entrena el modelo ganador sobre TODOS los datos y lo guarda en JSON.
    Este modelo es el artefacto final listo para despliegue en el Jetson.
    """
    seed     = cfg["study"]["seed"]
    eval_frac = cfg["fixed_params"]["eval_fraction"]
    rng      = np.random.RandomState(seed)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_eval  = max(100, int(len(X_scaled) * eval_frac))
    eval_ix = rng.choice(len(X_scaled), size=n_eval, replace=False)
    mask    = np.ones(len(X_scaled), dtype=bool)
    mask[eval_ix] = False

    clf = XGBClassifier(
        learning_rate         = params["learning_rate"],
        max_depth             = params["max_depth"],
        min_child_weight      = params["min_child_weight"],
        subsample             = params["subsample"],
        colsample_bytree      = params["colsample_bytree"],
        n_estimators          = params["n_estimators"],
        early_stopping_rounds = params["early_stopping_rounds"],
        tree_method           = params["tree_method"],
        eval_metric           = params["eval_metric"],
        verbosity             = params["verbosity"],
        random_state          = seed,
        n_jobs                = -1,
    )
    clf.fit(X_scaled[mask], y[mask],
            eval_set=[(X_scaled[eval_ix], y[eval_ix])],
            verbose=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save_model(str(output_path))
    log.info(f"Modelo guardado: {output_path}")


# ─── Guardar configuración ganadora en YAML ───────────────────────────────────

def save_best_config(params: dict, output_path: Path, metadata: dict = None) -> None:
    """Guarda la configuración ganadora en YAML (config/best_config.yaml)."""
    config = {
        "model":      "XGBoost",
        "sprint":     3,
        "params":     {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                       if isinstance(v, (int, np.integer)) else v
                       for k, v in params.items()
                       if k in ["learning_rate","max_depth","min_child_weight",
                                "subsample","colsample_bytree"]},
        "metadata":   metadata or {},
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    log.info(f"Config ganadora guardada: {output_path}")


# ─── Tabla top-k ─────────────────────────────────────────────────────────────

def build_topk_table(study: optuna.Study, cfg: dict, output_path: Path) -> list:
    """
    Construye la tabla top-k de mejores configuraciones (diapositiva 13).
    Columnas: trial, phase, params_resumidos, f1_mean±std, tiempo, notas.
    """
    import pandas as pd

    k = cfg["decision"]["top_k"]
    n_startup = cfg["study"]["n_startup_trials"]

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:k]

    rows = []
    for rank, t in enumerate(top_trials, 1):
        phase = "random" if t.number < n_startup else "bayes"
        params_str = ", ".join(
            f"{k_}={v:.4f}" if isinstance(v, float) else f"{k_}={v}"
            for k_, v in t.params.items()
            if k_ in cfg["search_space"]
        )
        row = {
            "rank":             rank,
            "trial":            t.number,
            "phase":            phase,
            "params_resumidos": params_str,
            "f1_macro_mean":    f"{t.value:.4f}",
            "tiempo_s":         f"{t.duration.total_seconds():.0f}" if t.duration else "N/A",
            "notas":            "ganador" if rank == 1 else f"respaldo {rank-1}" if rank <= 3 else "",
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return rows
