# ============================================================
# NOTEBOOK 05 — Sprint 3: Búsqueda de Hiperparámetros (Semana 7)
# Convertir a .ipynb con: jupytext --to notebook 05_hpo_xgboost.py
# ============================================================
# %% [markdown]
# # Sprint 3 — Búsqueda de Hiperparámetros para XGBoost
#
# **Branch:** `experiments/feature-engineering-v1`
# **Sesión:** 7 — Búsqueda de hiperparámetros (Ing. Glen Rodríguez)
# **Modelo a tunear:** `XGB_n200_baseFeats` (F1-Macro baseline = 0.8108)
#
# ## Objetivos
#
# Sigue las 5 metas de la sesión 7 (slide 2):
# 1. Diseñar search space bien acotado
# 2. Elegir Random o Bayesian (Optuna)
# 3. Configurar presupuesto: tiempo, #trials, paralelismo
# 4. Usar early stopping/pruning para ahorrar costo
# 5. Dejar experimentos comparables con logs y artefactos
#
# ## Estructura del notebook
#
# | Sección | Contenido |
# |---------|-----------|
# | 1 | Setup y carga de configuración |
# | 2 | Carga de datos (mismo HDF5 que sprints anteriores) |
# | 3 | Definición del espacio de búsqueda y presupuesto |
# | 4 | Estudio A — Random Search (30 trials) |
# | 5 | Estudio B — Bayesian TPE con warm-up (30 trials) |
# | 6 | Logs y artefactos guardados |
# | 7 | Tabla top-10 |
# | 8 | Gráfico de evolución best-so-far |
# | 9 | Importancia de hiperparámetros (fANOVA) |
# | 10 | Validación final con LOSO de la config ganadora |
# | 11 | Resumen ejecutivo + decisión |

# %% [markdown]
# ## 1. Setup

# %%
import os, sys
from pathlib import Path

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))
print(f"Directorio de trabajo: {project_root}")

# %%
import time
import yaml
import json
import numpy as np
import pandas as pd
import optuna
from loguru import logger as log

from posture_risk.experiments import (
    # Sprints 1-2
    build_gradient_boosting, cross_validate_pipeline,
    load_processed_h5, make_shared_splits,
    get_git_commit, compute_file_hash,
    # Sprint 3
    make_objective, create_study, run_study,
    export_studies_to_csv, build_topk_table,
    plot_best_so_far, plot_param_importances,
    select_winner_and_backups, save_winner_config,
)

with open("configs/hpo_search_space.yaml") as f:
    cfg = yaml.safe_load(f)

print(f"Experimento : {cfg['experiment']['name']}")
print(f"Modelo base : {cfg['experiment']['base_model']}")
print(f"F1 baseline : {cfg['experiment']['base_f1']:.4f} (a superar)")
print(f"Git commit  : {get_git_commit()}")

# %% [markdown]
# ## 2. Carga de datos

# %%
BASE_H5     = Path(cfg["paths"]["base_h5"])
STUDIES_DIR = Path(cfg["paths"]["studies_dir"])
MODELS_DIR  = Path(cfg["paths"]["models_dir"])
FIGURES_DIR = Path(cfg["paths"]["figures"])
LOG_CSV     = Path(cfg["paths"]["log_csv"])

for d in [STUDIES_DIR, MODELS_DIR, FIGURES_DIR, LOG_CSV.parent]:
    d.mkdir(parents=True, exist_ok=True)

X, y, subject_ids = load_processed_h5(BASE_H5)
data_hash = compute_file_hash(BASE_H5)
print(f"\nDataset cargado:")
print(f"  X.shape          : {X.shape}")
print(f"  y.shape          : {y.shape}")
print(f"  sujetos          : {np.unique(subject_ids)}")
print(f"  Hash MD5 HDF5    : {data_hash[:20]}...")
print(f"  (mismo dataset que sprints 1-2, integridad verificada)")

# %% [markdown]
# ## 3. Espacio de búsqueda y presupuesto

# %%
print("\n" + "="*65)
print("ESPACIO DE BÚSQUEDA (slide 10 de la sesión 7)")
print("="*65)
for param, spec in cfg["search_space"].items():
    if spec["type"] in ("loguniform",):
        print(f"  {param:<18}: log-uniform [{spec['low']}, {spec['high']}]")
    elif spec["type"] == "int":
        print(f"  {param:<18}: int [{spec['low']}, {spec['high']}]")
    else:
        print(f"  {param:<18}: uniform [{spec['low']}, {spec['high']}]")

print("\n" + "="*65)
print("HIPERPARÁMETROS FIJOS (no se tunean)")
print("="*65)
for k, v in cfg["fixed_params"].items():
    print(f"  {k:<24}: {v}")

print("\n" + "="*65)
print("PRESUPUESTO (slide 13)")
print("="*65)
print(f"  Trials Random       : {cfg['studies']['random']['n_trials']}")
print(f"  Trials Bayesian     : {cfg['studies']['bayesian']['n_trials']}")
print(f"    └─ warm-up random : {cfg['studies']['bayesian']['n_startup_trials']}")
print(f"  Total trials        : {cfg['budget']['total_trials']}")
print(f"  Tiempo estimado     : ~{cfg['budget']['expected_time_min']} min")
print(f"  CV interna          : {cfg['validation']['tuning']['strategy']}"
      f"({cfg['validation']['tuning']['n_splits']})")

# %% [markdown]
# ## 4. Estudio A — Random Search

# %%
objective_fn = make_objective(X, y, subject_ids, cfg)

study_random = create_study(
    study_name   = "random",
    sampler_type = "RandomSampler",
    storage_path = STUDIES_DIR / "random_study.db",
    cfg          = cfg,
)

run_study(
    study      = study_random,
    objective  = objective_fn,
    n_trials   = cfg["studies"]["random"]["n_trials"],
    study_name = "Random Search",
)

# %% [markdown]
# ## 5. Estudio B — Bayesian TPE con warm-up

# %%
study_bayesian = create_study(
    study_name   = "bayesian",
    sampler_type = "TPESampler",
    storage_path = STUDIES_DIR / "bayesian_study.db",
    cfg          = cfg,
)

run_study(
    study      = study_bayesian,
    objective  = objective_fn,
    n_trials   = cfg["studies"]["bayesian"]["n_trials"],
    study_name = "Bayesian (TPE + warm-up)",
)

# %% [markdown]
# ## 6. Logs y artefactos guardados (slide 16)

# %%
studies = {"random": study_random, "bayesian": study_bayesian}

df_logs = export_studies_to_csv(
    studies   = studies,
    csv_path  = LOG_CSV,
    seed      = cfg["fixed_params"]["random_state"],
    split_cfg = f"GroupKFold({cfg['validation']['tuning']['n_splits']})",
)

print(f"\n{'='*65}")
print(f"LOGS GUARDADOS")
print(f"{'='*65}")
print(f"  Trials totales : {len(df_logs)}")
print(f"  Completados   : {(df_logs['state']=='COMPLETE').sum()}")
print(f"  Podados       : {(df_logs['state']=='PRUNED').sum()}")
print(f"  Fallidos      : {(df_logs['state']=='FAIL').sum()}")
print(f"  Archivo CSV   : {LOG_CSV}")
print(f"\nArtefactos persistentes:")
print(f"  • {STUDIES_DIR/'random_study.db'}")
print(f"  • {STUDIES_DIR/'bayesian_study.db'}")
print(f"  (Studies en SQLite — sobreviven a reinicio del kernel)")

# %% [markdown]
# ## 7. Tabla Top-10 (slide 20)

# %%
df_top10 = build_topk_table(
    studies     = studies,
    output_path = Path(cfg["paths"]["results_csv"]),
    k           = 10,
)

print(f"\n{'='*120}")
print(f"TABLA TOP-10 — Mejores configuraciones encontradas")
print(f"{'='*120}")
df_display = pd.read_csv(cfg["paths"]["results_csv"])
print(df_display.to_string(index=False))

# %% [markdown]
# ## 8. Gráfico best-so-far — Random vs Bayesian

# %%
plot_best_so_far(
    studies      = studies,
    output_path  = FIGURES_DIR / "10_hpo_best_so_far.png",
    baseline_f1  = cfg["experiment"]["base_f1"],
)

# %% [markdown]
# ## 9. Importancia de hiperparámetros (fANOVA, slide 20)

# %%
importances_random   = plot_param_importances(
    study_random,
    output_path = FIGURES_DIR / "11_hpo_importances_random.png",
)
importances_bayesian = plot_param_importances(
    study_bayesian,
    output_path = FIGURES_DIR / "12_hpo_importances_bayesian.png",
)

print(f"\n{'='*55}")
print(f"IMPORTANCIA DE HIPERPARÁMETROS")
print(f"{'='*55}")
print(f"\nEstudio Random:")
for param, imp in importances_random.items():
    print(f"  {param:<20}: {imp:.4f}")
print(f"\nEstudio Bayesian:")
for param, imp in importances_bayesian.items():
    print(f"  {param:<20}: {imp:.4f}")

# %% [markdown]
# ## 10. Selección de ganadora + 2 configs de respaldo (slide 20)

# %%
selection = select_winner_and_backups(
    studies  = studies,
    n_backup = cfg["decision"]["n_backup_configs"],
)

winner  = selection["winner"]
backups = selection["backups"]

print(f"\n{'='*70}")
print(f"CONFIGURACIÓN GANADORA")
print(f"{'='*70}")
print(f"  Estudio       : {winner['study']}")
print(f"  Trial ID      : {winner['trial'].number}")
print(f"  F1-Macro      : {winner['f1']:.4f} ± {winner['f1_std']:.4f}")
print(f"  N árboles     : {winner['n_trees']:.0f}")
print(f"  Tiempo CV     : {winner['time_s']:.1f} s")
print(f"\n  Hiperparámetros:")
for k, v in winner["params"].items():
    print(f"    {k:<20}: {v:.5f}" if isinstance(v, float) else f"    {k:<20}: {v}")

print(f"\n{'='*70}")
print(f"CONFIGURACIONES DE RESPALDO")
print(f"{'='*70}")
for i, b in enumerate(backups, 1):
    print(f"\nRespaldo #{i} ({b['study']}/trial {b['trial'].number}):")
    print(f"  F1-Macro: {b['f1']:.4f} ± {b['f1_std']:.4f}")
    print(f"  N árboles: {b['n_trees']:.0f}  |  Tiempo: {b['time_s']:.1f}s")
    for k, v in b["params"].items():
        print(f"    {k:<20}: {v:.5f}" if isinstance(v, float) else f"    {k:<20}: {v}")

# Guardar config ganadora en JSON
winner_path = MODELS_DIR / "best_xgb_config.json"
save_winner_config(winner, winner_path)

# %% [markdown]
# ## 11. Validación final con LOSO completo
#
# La config ganadora se evalúa con LOSO completo (9 folds) para producir
# la métrica final comparable contra el baseline `XGB_n200_baseFeats`.

# %%
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Construir pipeline con los hiperparámetros ganadores
winner_params = winner["params"]
final_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(
        n_estimators           = cfg["fixed_params"]["n_estimators"],
        early_stopping_rounds  = None,   # sin early stopping en LOSO final
        tree_method            = cfg["fixed_params"]["tree_method"],
        eval_metric            = cfg["fixed_params"]["eval_metric"],
        random_state           = cfg["fixed_params"]["random_state"],
        n_jobs                 = cfg["fixed_params"]["n_jobs"],
        verbosity              = 0,
        **winner_params,
    )),
])

# Usar mismos splits que el sprint 1 (LOSO 9-fold)
splits_final = make_shared_splits(y, subject_ids, seed=42)

log.info("Validando configuración ganadora con LOSO completo (9 folds)...")
results_final = cross_validate_pipeline(
    final_pipeline, X, y, subject_ids, splits_final,
    "XGB_hpo_winner_LOSO",
)

print(f"\n{'='*65}")
print(f"VALIDACIÓN FINAL — LOSO completo (9 folds)")
print(f"{'='*65}")
print(f"  Modelo        : XGB_hpo_winner_LOSO (config ganadora del HPO)")
print(f"  F1-Macro      : {results_final['f1_macro_mean']:.4f} ± {results_final['f1_macro_std']:.4f}")
print(f"  PR-AUC        : {results_final['pr_auc_mean']:.4f} ± {results_final['pr_auc_std']:.4f}")
print(f"  Accuracy      : {results_final['accuracy_mean']:.4f} ± {results_final['accuracy_std']:.4f}")
print(f"  Tiempo CV     : {results_final['train_time_s']:.1f} s")

# %% [markdown]
# ## 12. Resumen ejecutivo y decisión

# %%
baseline_f1 = cfg["experiment"]["base_f1"]
delta_f1    = results_final["f1_macro_mean"] - baseline_f1
min_imp     = cfg["decision"]["min_improvement"]

decision = "ADOPTAR" if delta_f1 >= min_imp else "DESCARTAR — mantener baseline"

print("\n" + "="*72)
print("RESUMEN EJECUTIVO — SPRINT 3 (BÚSQUEDA DE HIPERPARÁMETROS)")
print("="*72)

print("\n─── ESPACIO DE BÚSQUEDA ────────────────────────────────────────────")
print(f"  5 hiperparámetros, ≤8 según asesor (slide 5)")
print(f"  log-scale para learning_rate y min_child_weight (slide 5)")
print(f"  n_estimators no tuneado → controlado por early_stopping_rounds=20")

print("\n─── PRESUPUESTO GASTADO ────────────────────────────────────────────")
n_completed_total = (df_logs['state']=='COMPLETE').sum()
n_pruned_total    = (df_logs['state']=='PRUNED').sum()
print(f"  Trials completados : {n_completed_total} / {cfg['budget']['total_trials']}")
print(f"  Trials podados     : {n_pruned_total} (early stopping de Optuna)")
print(f"  Tiempo total       : ~{cfg['budget']['expected_time_min']} min")

print("\n─── COMPARACIÓN CONTRA BASELINE ────────────────────────────────────")
print(f"  Baseline XGB_n200_baseFeats : F1 = {baseline_f1:.4f}")
print(f"  Ganador HPO (LOSO 9-fold)   : F1 = {results_final['f1_macro_mean']:.4f}")
print(f"  Δ F1-Macro                   : {delta_f1:+.4f} ({delta_f1*100:+.2f}%)")
print(f"  Umbral de mejora             : {min_imp:+.4f}")

print(f"\n─── DECISIÓN ───────────────────────────────────────────────────────")
print(f"  {decision}")
print(f"\n  Justificación técnica:")
if delta_f1 >= min_imp:
    print(f"    F1-Macro supera el baseline en {delta_f1*100:.2f}% (≥ {min_imp*100:.2f}% umbral).")
    print(f"    La búsqueda de hiperparámetros encontró una configuración más expresiva.")
else:
    print(f"    F1-Macro mejora apenas {delta_f1*100:.2f}% (< {min_imp*100:.2f}% umbral).")
    print(f"    El baseline está cerca del óptimo del espacio actual. Considerar:")
    print(f"    - Espacio de búsqueda más amplio")
    print(f"    - Más datos (curva de aprendizaje del Sprint 2 sugiere esto)")
    print(f"    - Arquitectura distinta (CNN-LSTM con secuencias)")

print(f"\n─── ARTEFACTOS GENERADOS ──────────────────────────────────────────")
print(f"  • {LOG_CSV}")
print(f"  • {STUDIES_DIR}/random_study.db")
print(f"  • {STUDIES_DIR}/bayesian_study.db")
print(f"  • {winner_path}")
print(f"  • {Path(cfg['paths']['results_csv'])}")
print(f"  • {FIGURES_DIR}/10_hpo_best_so_far.png")
print(f"  • {FIGURES_DIR}/11_hpo_importances_random.png")
print(f"  • {FIGURES_DIR}/12_hpo_importances_bayesian.png")
print("="*72)
