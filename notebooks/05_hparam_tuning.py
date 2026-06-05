# ============================================================
# NOTEBOOK 05 — Sprint 3: Búsqueda de Hiperparámetros
# Convertir a .ipynb con: jupytext --to notebook 05_hparam_tuning.py
# ============================================================
# %% [markdown]
# # Sprint 3 — Búsqueda de Hiperparámetros (HPO)
#
# **Branch:** `experiments/feature-engineering-v1`
# **Sesión:** 7 — Búsqueda de Hiperparámetros (Ing. Glen Rodríguez)
# **Modelo base:** `XGB_n200_baseFeats` (F1-Macro = 0.8108, Sprint 2)
#
# ## Estrategia híbrida (diapositiva 19)
# | Fase | Trials | Estrategia | Descripción |
# |------|--------|------------|-------------|
# | Calentamiento | 0-14 | Random | Exploración sin sesgo del espacio |
# | Bayesian | 15-29 | TPE (Optuna) | Explota zonas prometedoras aprendidas |
#
# ## Espacio de búsqueda XGBoost (diapositiva 2)
# | Hiperparámetro | Distribución | Rango |
# |---|---|---|
# | `learning_rate` | loguniform | [1e-3, 0.2] |
# | `max_depth` | int | [3, 10] |
# | `min_child_weight` | loguniform | [1e-2, 10] |
# | `subsample` | uniform | [0.5, 1.0] |
# | `colsample_bytree` | uniform | [0.5, 1.0] |
# | `n_estimators` | fijo=1000 + early stopping | — |

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
import numpy as np
import pandas as pd
import yaml
import optuna
import mlflow
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from loguru import logger as log
from pathlib import Path

from posture_risk.experiments.ab_runner import load_processed_h5, make_shared_splits
from posture_risk.experiments.hparam_tuning import (
    load_search_config, run_study, evaluate_with_loso,
    save_best_model, save_best_config, build_topk_table,
)

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 11})
COLORS = {"random": "#534AB7", "bayes": "#1D9E75", "baseline": "#D85A30"}

# Cargar configuraciones
cfg_search  = load_search_config("configs/search.yaml")
SEED        = cfg_search["study"]["seed"]
FIGURES_DIR = Path(cfg_search["paths"]["figures"])
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("logs").mkdir(exist_ok=True)

BASELINE_F1 = cfg_search["baseline"]["f1_macro"]

print(f"Baseline Sprint 2  : F1-Macro = {BASELINE_F1}")
print(f"Presupuesto total  : {cfg_search['study']['n_trials']} trials")
print(f"Fase Random        : {cfg_search['study']['n_startup_trials']} trials")
print(f"Fase Bayes         : {cfg_search['study']['n_trials'] - cfg_search['study']['n_startup_trials']} trials")
print(f"CV durante búsqueda: {cfg_search['cv_search']['n_splits']}-fold GroupKFold")
print(f"Early stopping     : {cfg_search['fixed_params']['early_stopping_rounds']} rondas")
print(f"MLflow URI         : {cfg_search['paths']['mlflow_uri']}")

# %% [markdown]
# ## 2. Carga de datos

# %%
BASE_H5    = Path(cfg_search["paths"]["base_h5"])
X, y, subject_ids = load_processed_h5(BASE_H5)

print(f"Dataset: X={X.shape}, y={y.shape}")
print(f"Sujetos : {np.unique(subject_ids)}")
print(f"Clases  : {dict(zip(*np.unique(y, return_counts=True)))}")

# %% [markdown]
# ## 3. Ejecución del estudio Optuna
#
# **Tiempo estimado:** 20-40 minutos (30 trials × ~3 folds × early stopping)
#
# El progreso se muestra trial a trial en el log.
# Para ver el tracking en tiempo real: abrir MLflow UI en otra terminal.

# %%
print("="*65)
print("INICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
print("Para ver el tracking en tiempo real, abrir una nueva terminal y ejecutar:")
print(f"  cd {project_root}")
print(f"  conda activate posture-risk")
print(f"  mlflow ui --backend-store-uri mlruns")
print(f"  → Abrir http://localhost:5000 en el navegador")
print("="*65 + "\n")

study = run_study(X, y, subject_ids, cfg_search)

# %% [markdown]
# ## 4. Resumen del estudio

# %%
completed = [t for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in study.trials
             if t.state == optuna.trial.TrialState.PRUNED]
n_startup = cfg_search["study"]["n_startup_trials"]

print(f"\n{'='*65}")
print(f"RESUMEN DEL ESTUDIO")
print(f"{'='*65}")
print(f"Trials totales     : {len(study.trials)}")
print(f"Trials completados : {len(completed)}")
print(f"Trials podados     : {len(pruned)} "
      f"({len(pruned)/len(study.trials)*100:.0f}% → ahorro real por pruning)")
print(f"\nMejor trial        : #{study.best_trial.number}")
phase_best = "random" if study.best_trial.number < n_startup else "bayes"
print(f"Fase               : {phase_best}")
print(f"Mejor F1-Macro     : {study.best_value:.4f}")
print(f"Mejores parámetros :")
for k, v in study.best_params.items():
    print(f"  {k:<22}: {v}")
print(f"\nΔF1 vs baseline    : {study.best_value - BASELINE_F1:+.4f} "
      f"({(study.best_value - BASELINE_F1)/BASELINE_F1*100:+.2f}%)")

# %% [markdown]
# ## 5. Tabla top-5 (diapositiva 13)

# %%
table_path = Path(cfg_search["paths"]["table_topk"])
top_rows   = build_topk_table(study, cfg_search, table_path)

df_top = pd.DataFrame(top_rows)
print(f"\nTABLA TOP-{cfg_search['decision']['top_k']} — Mejores configuraciones")
print("="*90)
print(df_top.to_string(index=False))
print(f"\nTabla guardada: {table_path}")

# %% [markdown]
# ## 6. Gráfico de evolución best-so-far (diapositiva 13)

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ── Panel izquierdo: best-so-far por fase ────────────────────────────────────
ax = axes[0]
random_vals, bayes_vals = [], []
best_random_so_far = -np.inf
best_bayes_so_far  = -np.inf

random_x, random_y_plot = [], []
bayes_x,  bayes_y_plot  = [], []

for t in sorted(completed, key=lambda x: x.number):
    if t.number < n_startup:
        best_random_so_far = max(best_random_so_far, t.value)
        random_x.append(t.number)
        random_y_plot.append(best_random_so_far)
    else:
        best_bayes_so_far = max(best_bayes_so_far, t.value)
        bayes_x.append(t.number)
        bayes_y_plot.append(best_bayes_so_far)

if random_x:
    ax.plot(random_x, random_y_plot, "o-",
            color=COLORS["random"], lw=2.2, label=f"Random (fase 1, {len(random_x)} trials)",
            alpha=0.9)
if bayes_x:
    ax.plot(bayes_x, bayes_y_plot, "s-",
            color=COLORS["bayes"], lw=2.2, label=f"Bayes/TPE (fase 2, {len(bayes_x)} trials)",
            alpha=0.9)

ax.axhline(BASELINE_F1, color=COLORS["baseline"], ls="--", lw=1.5,
           label=f"Baseline Sprint 2 (F1={BASELINE_F1:.4f})", alpha=0.8)
ax.axvline(n_startup - 0.5, color="#888888", ls=":", lw=1, alpha=0.5)
ax.text(n_startup * 0.5, ax.get_ylim()[0] + 0.01, "Random\n(calent.)",
        ha="center", fontsize=9, color=COLORS["random"], alpha=0.7)
ax.text(n_startup + (cfg_search["study"]["n_trials"] - n_startup) * 0.5,
        ax.get_ylim()[0] + 0.01, "Bayes\n(TPE)",
        ha="center", fontsize=9, color=COLORS["bayes"], alpha=0.7)

ax.set_xlabel("Número de trial")
ax.set_ylabel("Mejor F1-Macro acumulado (best-so-far)")
ax.set_title("Evolución de la búsqueda — Best-so-far por fase\n"
             "(cada punto = mejor encontrado hasta ese trial)", fontweight="bold")
ax.legend(loc="lower right")
ax.grid(alpha=0.18)

# ── Panel derecho: F1 individual por trial ────────────────────────────────────
ax2 = axes[1]
for t in completed:
    phase = "random" if t.number < n_startup else "bayes"
    marker = "o" if phase == "random" else "s"
    ax2.scatter(t.number, t.value, color=COLORS[phase], alpha=0.7,
                s=50, marker=marker)

ax2.axhline(BASELINE_F1, color=COLORS["baseline"], ls="--", lw=1.5,
            label=f"Baseline (F1={BASELINE_F1:.4f})", alpha=0.8)
ax2.axhline(study.best_value, color="#222222", ls="-.", lw=1,
            label=f"Mejor trial (F1={study.best_value:.4f})", alpha=0.7)
ax2.axvline(n_startup - 0.5, color="#888888", ls=":", lw=1, alpha=0.5)

from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0],[0], marker="o", color=COLORS["random"], lw=0, markersize=8,
           label=f"Random (trials 0–{n_startup-1})"),
    Line2D([0],[0], marker="s", color=COLORS["bayes"],  lw=0, markersize=8,
           label=f"Bayes (trials {n_startup}–{cfg_search['study']['n_trials']-1})"),
    Line2D([0],[0], color=COLORS["baseline"], ls="--", lw=1.5, label="Baseline S2"),
    Line2D([0],[0], color="#222222", ls="-.", lw=1, label="Mejor trial"),
]
ax2.legend(handles=legend_elems, fontsize=9)
ax2.set_xlabel("Número de trial")
ax2.set_ylabel("F1-Macro del trial")
ax2.set_title("F1-Macro por trial individual\n(todos los trials completados)", fontweight="bold")
ax2.grid(alpha=0.18)

plt.suptitle("Búsqueda de Hiperparámetros — Evolución completa\n"
             "Sprint 3 | XGB_n200_baseFeats como referencia",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "10_hparam_evolution.png", bbox_inches="tight", dpi=150)
plt.show()
print(f"Gráfico guardado: {FIGURES_DIR / '10_hparam_evolution.png'}")

# %% [markdown]
# ## 7. Importancia de hiperparámetros (Optuna nativo)
#
# Usa FAnova para estimar qué hiperparámetros tuvieron
# más impacto en F1-Macro a lo largo de los 30 trials.

# %%
try:
    importances = optuna.importance.get_param_importances(study)

    fig, ax = plt.subplots(figsize=(10, 5))
    names   = list(importances.keys())
    values  = list(importances.values())

    # Ordenar por importancia
    sorted_idx = np.argsort(values)
    colors_bar = [COLORS["bayes"] if values[i] == max(values) else
                  COLORS["random"] if values[i] == sorted(values)[-2] else
                  "#AAAAAA" for i in sorted_idx]

    bars = ax.barh(range(len(names)), [values[i] for i in sorted_idx],
                   color=colors_bar, alpha=0.85, height=0.65)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in sorted_idx])
    ax.set_xlabel("Importancia relativa (FAnova)")
    ax.set_title("Importancia de Hiperparámetros\n"
                 "(¿cuánto impacta cada HP en F1-Macro?)", fontweight="bold")

    for bar, val_idx in zip(bars, sorted_idx):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{values[val_idx]:.3f}", va="center", fontsize=10)

    ax.axvline(1/len(names), color="#D85A30", ls="--", lw=1.2, alpha=0.7,
               label=f"Importancia uniforme (1/{len(names)})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_hparam_importance.png", bbox_inches="tight", dpi=150)
    plt.show()

    print("\nImportancia de hiperparámetros (FAnova):")
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {name:<22}: {imp:.4f}  {bar}")
    print(f"\nGráfico guardado: {FIGURES_DIR / '11_hparam_importance.png'}")

except Exception as e:
    print(f"Importancias de HP no disponibles (necesita ≥10 trials completados): {e}")

# %% [markdown]
# ## 8. Evaluación final del ganador con LOSO completo (9-fold)
#
# Solo se evalúa con LOSO el mejor trial (config ganadora).
# Esto produce métricas directamente comparables con el Sprint 2.

# %%
best_params_search = dict(study.best_params)
# Añadir params fijos
best_params_search.update({
    k: v for k, v in cfg_search["fixed_params"].items()
    if k not in best_params_search
})

print("Evaluando config ganadora con LOSO completo...")
result_winner = evaluate_with_loso(
    best_params_search, X, y, subject_ids,
    cfg_search, label="XGB_tuned_bayes"
)

# %% [markdown]
# ## 9. Decisión y configs de respaldo

# %%
BASELINE_PR  = cfg_search["baseline"]["pr_auc"]
min_improve  = cfg_search["decision"]["min_improvement"]
delta_f1     = result_winner["f1_macro_mean"] - BASELINE_F1
adopt        = delta_f1 >= min_improve

print(f"\n{'='*65}")
print(f"DECISIÓN FINAL — Sprint 3")
print(f"{'='*65}")
print(f"\n  Baseline Sprint 2 (XGB_n200_baseFeats)")
print(f"    F1-Macro : {BASELINE_F1:.4f}")
print(f"    PR-AUC   : {BASELINE_PR:.4f}")
print(f"\n  Config ganadora (XGB_tuned_bayes — trial #{study.best_trial.number})")
print(f"    F1-Macro : {result_winner['f1_macro_mean']:.4f} ± {result_winner['f1_macro_std']:.4f}")
print(f"    PR-AUC   : {result_winner['pr_auc_mean']:.4f} ± {result_winner['pr_auc_std']:.4f}")
print(f"    ΔF1      : {delta_f1:+.4f}  ({'supera' if adopt else 'NO supera'} umbral {min_improve})")
print(f"\n  DECISIÓN  : {'✓ ADOPTAR XGB_tuned_bayes' if adopt else '⚠ MANTENER XGB_n200_baseFeats (sin mejora suficiente)'}")

# Configs de respaldo (top-2 tras el ganador)
print(f"\n  Configs de respaldo (diapositiva 13):")
completed_sorted = sorted(
    [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
    key=lambda t: t.value, reverse=True
)
n_backup = cfg_search["decision"]["n_backup_configs"]
for i, t in enumerate(completed_sorted[1:n_backup+1], 1):
    phase = "random" if t.number < n_startup else "bayes"
    print(f"  Respaldo {i}: trial #{t.number} [{phase}] | F1={t.value:.4f}")
    for k, v in t.params.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

# %% [markdown]
# ## 10. Guardar artefactos finales

# %%
# 1. Mejor modelo entrenado sobre todos los datos
print("Guardando modelo ganador...")
save_best_model(
    best_params_search, X, y, cfg_search,
    Path(cfg_search["paths"]["best_bayes"])
)

# 2. Config ganadora en YAML
print("Guardando best_config.yaml...")
save_best_config(
    best_params_search,
    Path(cfg_search["paths"]["best_config"]),
    metadata={
        "trial_id":     study.best_trial.number,
        "phase":        "bayes" if study.best_trial.number >= n_startup else "random",
        "f1_macro_loso": float(result_winner["f1_macro_mean"]),
        "sprint":       3,
    }
)

# 3. El estudio Optuna ya se guardó durante run_study()
print(f"Estudio Optuna ya guardado: {cfg_search['paths']['optuna_study']}")

print(f"\nArtefactos guardados:")
for label, path in [
    ("Mejor modelo (JSON)",  cfg_search["paths"]["best_bayes"]),
    ("Config ganadora YAML", cfg_search["paths"]["best_config"]),
    ("Estudio Optuna",       cfg_search["paths"]["optuna_study"]),
    ("Logs CSV",             cfg_search["paths"]["log_csv"]),
    ("Tabla top-5",          cfg_search["paths"]["table_topk"]),
]:
    exists = "✓" if Path(path).exists() else "✗"
    print(f"  {exists} {label:<25} → {path}")

# %% [markdown]
# ## 11. Instrucciones MLflow UI
#
# Para revisar todos los trials visualmente:
#
# ```powershell
# # En una nueva terminal (con el entorno activo):
# conda activate posture-risk
# mlflow ui --backend-store-uri mlruns
# # → Abrir http://localhost:5000
# ```
#
# En la interfaz verás:
# - Tabla con todos los runs (trials) y sus métricas
# - Filtros por fase (random/bayes), métrica, parámetros
# - Gráficos de dispersión para correlacionar HP con F1-Macro
# - Comparación lado a lado de cualquier par de trials

# %% [markdown]
# ## 12. Resumen ejecutivo del Sprint 3

# %%
print("\n" + "="*70)
print("RESUMEN EJECUTIVO — SPRINT 3 (SEMANA 7)")
print("="*70)
print(f"\n─── PRESUPUESTO ────────────────────────────────────────────────────")
print(f"  Trials totales   : {len(study.trials)}")
print(f"  Trials random    : {len([t for t in completed if t.number < n_startup])}")
print(f"  Trials bayes     : {len([t for t in completed if t.number >= n_startup])}")
print(f"  Trials podados   : {len(pruned)} (ahorro por pruning)")
print(f"\n─── ESPACIO DE BÚSQUEDA ────────────────────────────────────────────")
for hp, space_cfg in cfg_search["search_space"].items():
    dist = space_cfg["type"]
    rng_str = f"[{space_cfg['low']}, {space_cfg['high']}]"
    print(f"  {hp:<22}: {dist:<12} {rng_str}")
print(f"\n─── RESULTADO ──────────────────────────────────────────────────────")
print(f"  Baseline S2 F1   : {BASELINE_F1:.4f}")
print(f"  Ganador F1       : {result_winner['f1_macro_mean']:.4f} ± {result_winner['f1_macro_std']:.4f}")
print(f"  ΔF1              : {delta_f1:+.4f}")
print(f"  Decisión         : {'ADOPTAR XGB_tuned_bayes' if adopt else 'MANTENER baseline'}")
print(f"\n─── ARTEFACTOS ─────────────────────────────────────────────────────")
print(f"  logs/hpo_runs.csv       → {len(study.trials)} trials registrados")
print(f"  models/best_bayes.json  → modelo ganador listo para Jetson")
print(f"  configs/best_config.yaml→ config ganadora reproducible")
print(f"  mlruns/                 → MLflow tracking (mlflow ui)")
print("="*70)
