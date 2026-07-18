"""
hpo_reporting.py  (Sprint 3)
-----------------------------
Logging persistente y reportes de la búsqueda de hiperparámetros.

Cumple la slide 16 del asesor:
  "logs/hpo_runs.csv: trial_id, params, seed, split_cfg, metric_mean,
   metric_std, tiempo"

Y la slide 20 (Plantilla de resultados):
  - Tabla top-k con params, métrica(mean±std), tiempo, notas
  - Gráfico: métrica vs trial + importancia de hiperparámetros
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from loguru import logger as log


# Estilo visual del proyecto
plt.rcParams.update({
    "figure.dpi":         130,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.size":          11,
})

COLOR_RANDOM   = "#534AB7"   # púrpura — Random Search
COLOR_BAYESIAN = "#1D9E75"   # verde teal — Bayesian (TPE)
COLOR_BASELINE = "#D85A30"   # coral — baseline a superar


# ─── Logging persistente (slide 16) ──────────────────────────────────────────

HPO_CSV_HEADER = [
    "trial_id", "study_name", "sampler",
    "timestamp", "state",
    "learning_rate", "max_depth", "min_child_weight",
    "subsample", "colsample_bytree",
    "n_trees_mean",  "n_trees_std",
    "f1_macro_mean", "f1_macro_std",
    "train_time_s",  "n_folds",
    "seed", "split_cfg",
]


def export_studies_to_csv(
    studies:   Dict[str, optuna.Study],
    csv_path:  Path,
    seed:      int = 42,
    split_cfg: str = "GroupKFold(3)",
) -> pd.DataFrame:
    """
    Exporta todos los trials de los estudios al CSV de logs.
    Resultado: una fila por trial, con todos los campos del slide 16.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for study_name, study in studies.items():
        sampler_name = type(study.sampler).__name__
        for trial in study.trials:
            params = trial.params if trial.params else {}
            attrs  = trial.user_attrs
            row = {
                "trial_id":         f"{study_name}_{trial.number:03d}",
                "study_name":       study_name,
                "sampler":          sampler_name,
                "timestamp":        trial.datetime_start.isoformat(timespec="seconds")
                                    if trial.datetime_start else "",
                "state":            trial.state.name,
                "learning_rate":    f"{params.get('learning_rate', ''):.5e}"
                                    if params.get('learning_rate') is not None else "",
                "max_depth":        params.get("max_depth", ""),
                "min_child_weight": f"{params.get('min_child_weight', ''):.5e}"
                                    if params.get('min_child_weight') is not None else "",
                "subsample":        f"{params.get('subsample', ''):.4f}"
                                    if params.get('subsample') is not None else "",
                "colsample_bytree": f"{params.get('colsample_bytree', ''):.4f}"
                                    if params.get('colsample_bytree') is not None else "",
                "n_trees_mean":     f"{attrs.get('n_trees_mean', ''):.1f}"
                                    if attrs.get('n_trees_mean') is not None else "",
                "n_trees_std":      f"{attrs.get('n_trees_std', ''):.1f}"
                                    if attrs.get('n_trees_std') is not None else "",
                "f1_macro_mean":    f"{attrs.get('f1_macro_mean', trial.value):.4f}"
                                    if trial.value is not None else "",
                "f1_macro_std":     f"{attrs.get('f1_macro_std', ''):.4f}"
                                    if attrs.get('f1_macro_std') is not None else "",
                "train_time_s":     f"{attrs.get('train_time_total', ''):.2f}"
                                    if attrs.get('train_time_total') is not None else "",
                "n_folds":          attrs.get("n_folds", ""),
                "seed":             seed,
                "split_cfg":        split_cfg,
            }
            rows.append(row)

    df = pd.DataFrame(rows, columns=HPO_CSV_HEADER)
    df.to_csv(csv_path, index=False)

    log.info(f"Logs HPO exportados: {len(rows)} trials → {csv_path}")
    return df


# ─── Tabla Top-k (slide 20) ──────────────────────────────────────────────────

def build_topk_table(
    studies:     Dict[str, optuna.Study],
    output_path: Path,
    k:           int = 10,
) -> pd.DataFrame:
    """
    Tabla top-k combinada de todos los estudios.
    Cada fila: rank, trial_id, sampler, params resumidos, F1 (mean±std),
    n_trees usados, tiempo.
    """
    rows = []
    for study_name, study in studies.items():
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            attrs = trial.user_attrs
            rows.append({
                "trial_id":         f"{study_name}_{trial.number:03d}",
                "sampler":          type(study.sampler).__name__,
                "F1_mean":          attrs.get("f1_macro_mean", trial.value),
                "F1_std":           attrs.get("f1_macro_std", 0),
                "n_trees":          attrs.get("n_trees_mean", 0),
                "time_s":           attrs.get("train_time_total", 0),
                "learning_rate":    trial.params.get("learning_rate"),
                "max_depth":        trial.params.get("max_depth"),
                "min_child_weight": trial.params.get("min_child_weight"),
                "subsample":        trial.params.get("subsample"),
                "colsample_bytree": trial.params.get("colsample_bytree"),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("F1_mean", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df_top = df.head(k).copy()

    # Formato presentable
    df_display = df_top[[
        "rank", "trial_id", "sampler",
        "F1_mean", "F1_std", "n_trees", "time_s",
        "learning_rate", "max_depth", "min_child_weight",
        "subsample", "colsample_bytree",
    ]].copy()

    for col in ["F1_mean", "F1_std"]:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
    df_display["learning_rate"]    = df_display["learning_rate"].apply(lambda x: f"{x:.5f}")
    df_display["min_child_weight"] = df_display["min_child_weight"].apply(lambda x: f"{x:.4f}")
    df_display["subsample"]        = df_display["subsample"].apply(lambda x: f"{x:.3f}")
    df_display["colsample_bytree"] = df_display["colsample_bytree"].apply(lambda x: f"{x:.3f}")
    df_display["n_trees"]          = df_display["n_trees"].apply(lambda x: f"{x:.0f}")
    df_display["time_s"]           = df_display["time_s"].apply(lambda x: f"{x:.1f}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_display.to_csv(output_path, index=False)
    log.info(f"Tabla top-{k} guardada: {output_path}")

    return df_top


# ─── Gráfico best-so-far (slide 20) ──────────────────────────────────────────

def plot_best_so_far(
    studies:    Dict[str, optuna.Study],
    output_path: Path,
    baseline_f1: Optional[float] = None,
) -> Path:
    """
    Gráfico de evolución: F1-Macro del mejor trial acumulado hasta ese punto,
    superpuesto para todos los estudios. Si baseline_f1 se provee, se dibuja
    como línea horizontal de referencia.
    """
    fig, ax = plt.subplots(figsize=(11, 6))

    color_map = {"random": COLOR_RANDOM, "bayesian": COLOR_BAYESIAN}

    for study_name, study in studies.items():
        completed_trials = [t for t in study.trials
                            if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            continue
        values = [t.value for t in completed_trials]
        best_so_far = np.maximum.accumulate(values)
        color = color_map.get(study_name.lower(), "#888888")

        ax.step(
            range(1, len(best_so_far) + 1),
            best_so_far,
            where="post", lw=2.4, color=color, alpha=0.92,
            label=f"{study_name.capitalize()}  "
                  f"(mejor F1 = {best_so_far[-1]:.4f})",
        )
        ax.scatter(
            range(1, len(values) + 1),
            values,
            color=color, alpha=0.28, s=22, edgecolors="none",
        )

    if baseline_f1 is not None:
        ax.axhline(
            baseline_f1, color=COLOR_BASELINE, lw=1.8, ls="--",
            label=f"Baseline XGB_n200_baseFeats ({baseline_f1:.4f})",
        )

    ax.set_xlabel("Trial #")
    ax.set_ylabel("F1-Macro (mejor acumulado)")
    ax.set_title(
        "Evolución best-so-far — Búsqueda de Hiperparámetros\n"
        "Random Search vs Bayesian (TPE con warm-up)",
        fontweight="bold",
    )
    ax.legend(loc="lower right", framealpha=0.95)
    ax.grid(alpha=0.18)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    log.info(f"Gráfico best-so-far guardado: {output_path}")
    return output_path


# ─── Importancia de hiperparámetros (slide 20) ───────────────────────────────

def plot_param_importances(
    study:       optuna.Study,
    output_path: Path,
) -> Dict[str, float]:
    """
    Importancia de hiperparámetros mediante fANOVA (nativo de Optuna).
    Identifica cuál hiperparámetro tiene más impacto en F1-Macro.
    """
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 10:
        log.warning("Pocos trials completos para fANOVA confiable")

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception as e:
        log.error(f"No se pudo calcular importancia: {e}")
        return {}

    fig, ax = plt.subplots(figsize=(10, 5))
    params  = list(importances.keys())
    values  = list(importances.values())

    colors = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#E24B4A"][:len(params)]
    bars   = ax.barh(range(len(params)), values, color=colors, alpha=0.88)
    ax.set_yticks(range(len(params)))
    ax.set_yticklabels(params)
    ax.set_xlabel("Importancia relativa (fANOVA)")
    ax.set_title(
        f"Importancia de hiperparámetros — Estudio: {study.study_name}\n"
        f"({len(completed)} trials completos)",
        fontweight="bold",
    )
    ax.invert_yaxis()
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", fontsize=10,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    log.info(f"Gráfico de importancias guardado: {output_path}")

    return importances


# ─── Selección de config ganadora y respaldos (slide 20) ─────────────────────

def select_winner_and_backups(
    studies:    Dict[str, optuna.Study],
    n_backup:   int = 2,
) -> Dict:
    """
    Devuelve la config ganadora global y n configs de respaldo.

    Criterio:
      - Ganadora: la de mayor F1-Macro entre todos los trials completos.
      - Respaldos: trials con F1 cercano (>= 95% de la ganadora) pero
        con parámetros distintos (no redundantes).
    """
    all_trials = []
    for study_name, study in studies.items():
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            all_trials.append({
                "study":   study_name,
                "trial":   t,
                "f1":      t.value,
                "params":  t.params,
                "n_trees": t.user_attrs.get("n_trees_mean", 0),
                "time_s":  t.user_attrs.get("train_time_total", 0),
                "f1_std":  t.user_attrs.get("f1_macro_std", 0),
            })

    if not all_trials:
        return {}

    all_trials.sort(key=lambda x: x["f1"], reverse=True)
    winner    = all_trials[0]
    threshold = winner["f1"] * 0.97

    # Respaldos: F1 cercano pero parámetros distintos
    backups   = []
    for t in all_trials[1:]:
        if t["f1"] < threshold:
            break
        # Tomar configs claramente distintas (diferencia en max_depth o lr)
        is_distinct = True
        for already in backups + [winner]:
            same_depth = t["params"].get("max_depth") == already["params"].get("max_depth")
            close_lr   = abs(
                np.log10(t["params"].get("learning_rate", 1e-3)) -
                np.log10(already["params"].get("learning_rate", 1e-3))
            ) < 0.3
            if same_depth and close_lr:
                is_distinct = False
                break
        if is_distinct:
            backups.append(t)
        if len(backups) >= n_backup:
            break

    return {
        "winner":  winner,
        "backups": backups,
    }


# ─── Serialización JSON de la configuración ganadora ─────────────────────────

def save_winner_config(
    winner:     Dict,
    output_path: Path,
) -> Path:
    """Guarda la configuración ganadora en JSON para reproducibilidad."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "study":   winner["study"],
        "trial_number": winner["trial"].number,
        "f1_macro_mean": float(winner["f1"]),
        "f1_macro_std":  float(winner["f1_std"]),
        "n_trees_mean":  float(winner["n_trees"]),
        "train_time_s":  float(winner["time_s"]),
        "params":  winner["params"],
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    log.info(f"Configuración ganadora guardada: {output_path}")
    return output_path
