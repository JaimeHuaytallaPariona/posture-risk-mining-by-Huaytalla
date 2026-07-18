"""
hpo.py  (Sprint 3)
-------------------
Búsqueda de hiperparámetros para XGBoost mediante Optuna.

Cumple los principios del Ing. Glen Rodríguez (sesión 7):
  - Espacio acotado (5 hiperparámetros)
  - Validación coherente: GroupKFold por sujeto (sin leakage de identidad)
  - Mismo seed/split entre trials para comparabilidad
  - Pruning de trials malos (MedianPruner)
  - XGBoost early_stopping_rounds nativo (no se tunea n_estimators)
  - Persistencia en SQLite para resumir runs interrumpidos
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from loguru import logger as log
from optuna.pruners import MedianPruner
from optuna.samplers import RandomSampler, TPESampler
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from sklearn.metrics import f1_score


# Silenciar logs verbosos de Optuna (mantenemos los nuestros con loguru)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Función objetivo ─────────────────────────────────────────────────────────

def make_objective(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    cfg: dict,
):
    """
    Construye la función objetivo cerrada sobre los datos.

    El uso de make_objective como factory permite que Optuna serialice
    sólo lo necesario (trial) sin re-pasar los datos en cada llamada.

    Estructura interna de cada trial:
      1. Sugerir hiperparámetros del espacio definido
      2. GroupKFold(3) externa por sujeto
      3. Dentro de cada fold:
         a. Separar 20% de grupos del train para early stopping
         b. Ajustar StandardScaler SOLO en train_inner
         c. Entrenar XGBoost con early_stopping_rounds=20
         d. Evaluar en val_outer
      4. Reportar F1-Macro acumulado para pruning
      5. Retornar F1-Macro promedio entre folds
    """
    n_splits = cfg["validation"]["tuning"]["n_splits"]
    es_ratio = cfg["validation"]["tuning"]["early_stopping_split_ratio"]
    fixed    = cfg["fixed_params"]
    seed     = fixed["random_state"]

    splitter = GroupKFold(n_splits=n_splits)
    splits   = list(splitter.split(X, y, groups=subject_ids))

    def objective(trial: optuna.Trial) -> float:
        # ── Espacio de búsqueda (slide 10) ───────────────────────────────────
        params = {
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 10.0, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        fold_f1s        = []
        fold_n_trees    = []
        fold_train_time = []
        rng             = np.random.RandomState(seed)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            t_fold = time.time()

            # ── Early stopping interno: separar grupos de train_idx ──────────
            train_groups   = np.unique(subject_ids[train_idx])
            n_es_groups    = max(1, int(np.round(len(train_groups) * es_ratio)))
            es_groups      = rng.choice(train_groups, size=n_es_groups, replace=False)
            es_mask        = np.isin(subject_ids[train_idx], es_groups)
            es_train_idx   = train_idx[~es_mask]
            es_val_idx     = train_idx[es_mask]

            X_tr  = X[es_train_idx]
            y_tr  = y[es_train_idx]
            X_es  = X[es_val_idx]
            y_es  = y[es_val_idx]
            X_va  = X[val_idx]
            y_va  = y[val_idx]

            # ── Escalado: fit SOLO en train_inner (sin leakage) ──────────────
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_es_s = scaler.transform(X_es)
            X_va_s = scaler.transform(X_va)

            # ── Entrenar XGBoost con early stopping nativo ───────────────────
            clf = XGBClassifier(
                n_estimators           = fixed["n_estimators"],
                early_stopping_rounds  = fixed["early_stopping_rounds"],
                tree_method            = fixed["tree_method"],
                eval_metric            = fixed["eval_metric"],
                random_state           = seed,
                n_jobs                 = fixed["n_jobs"],
                verbosity              = fixed["verbosity"],
                **params,
            )

            try:
                clf.fit(
                    X_tr_s, y_tr,
                    eval_set=[(X_es_s, y_es)],
                    verbose=False,
                )
            except Exception as exc:
                # Si el trial falla, marcar como podado para no contaminar el estudio
                log.warning(f"Trial {trial.number} fold {fold_idx}: {exc}")
                raise optuna.TrialPruned()

            y_pred = clf.predict(X_va_s)
            f1     = f1_score(y_va, y_pred, average="macro", zero_division=0)
            fold_f1s.append(f1)
            fold_n_trees.append(int(clf.best_iteration) + 1)
            fold_train_time.append(time.time() - t_fold)

            # ── Reporte intermedio para pruning ──────────────────────────────
            mean_so_far = float(np.mean(fold_f1s))
            trial.report(mean_so_far, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # ── Métricas finales del trial ───────────────────────────────────────
        mean_f1 = float(np.mean(fold_f1s))
        std_f1  = float(np.std(fold_f1s))

        # User attributes: información extra que se guarda con el trial
        trial.set_user_attr("f1_macro_mean",    mean_f1)
        trial.set_user_attr("f1_macro_std",     std_f1)
        trial.set_user_attr("n_trees_mean",     float(np.mean(fold_n_trees)))
        trial.set_user_attr("n_trees_std",      float(np.std(fold_n_trees)))
        trial.set_user_attr("train_time_total", float(np.sum(fold_train_time)))
        trial.set_user_attr("n_folds",          n_splits)

        return mean_f1

    return objective


# ─── Creación y ejecución de estudios ────────────────────────────────────────

def create_study(
    study_name:    str,
    sampler_type:  str,
    storage_path:  Path,
    cfg:           dict,
) -> optuna.Study:
    """
    Crea un estudio Optuna con sampler y pruner según configuración.

    storage_path: ruta al archivo SQLite. Si existe, se resume el estudio.
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url  = f"sqlite:///{storage_path.as_posix()}"

    seed = cfg["fixed_params"]["random_state"]

    # ── Sampler ──────────────────────────────────────────────────────────────
    if sampler_type == "RandomSampler":
        sampler = RandomSampler(seed=seed)
    elif sampler_type == "TPESampler":
        n_startup = cfg["studies"]["bayesian"]["n_startup_trials"]
        sampler   = TPESampler(seed=seed, n_startup_trials=n_startup)
    else:
        raise ValueError(f"Sampler desconocido: {sampler_type}")

    # ── Pruner ───────────────────────────────────────────────────────────────
    pruner_cfg = cfg["pruning"]
    if pruner_cfg["enabled"]:
        pruner = MedianPruner(
            n_startup_trials = pruner_cfg["n_startup_trials"],
            n_warmup_steps   = pruner_cfg["n_warmup_steps"],
            interval_steps   = pruner_cfg["interval_steps"],
        )
    else:
        pruner = optuna.pruners.NopPruner()

    direction = cfg["metric"]["direction"]
    study     = optuna.create_study(
        study_name    = study_name,
        storage       = storage_url,
        sampler       = sampler,
        pruner        = pruner,
        direction     = direction,
        load_if_exists= True,
    )

    log.info(f"Estudio creado: {study_name} | sampler={sampler_type} "
             f"| storage={storage_path.name}")
    return study


def run_study(
    study:      optuna.Study,
    objective,
    n_trials:   int,
    study_name: str,
) -> None:
    """Ejecuta n_trials sobre el estudio dado."""
    t0 = time.time()
    log.info(f"Ejecutando {n_trials} trials: {study_name}")

    study.optimize(
        objective,
        n_trials      = n_trials,
        show_progress_bar = True,
        gc_after_trial    = True,
    )

    elapsed = time.time() - t0

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed    = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    log.info(
        f"{study_name} terminado en {elapsed/60:.1f} min: "
        f"{n_completed} completados, {n_pruned} podados, {n_failed} fallidos"
    )
    log.info(f"  Mejor F1-Macro: {study.best_value:.4f}")
    log.info(f"  Mejores params: {study.best_params}")
