# ============================================================
# NOTEBOOK 03 — Experimentos A/B (Semana 5)
# Convertir a .ipynb con: jupytext --to notebook 03_ab_experiments.py
# ============================================================
# %% [markdown]
# # Experimentos A/B — Feature Engineering vs Baseline
#
# **Branch:** `experiments/feature-engineering-v1`
# **Tesis:** Detección de Posturas Riesgosas en Operadores Perforistas
#
# ## Protocolo del Ing. Glen Rodríguez (sesiones 4 y 5)
#
# - **Un cambio único por variante** — no mezclar cambios
# - **Mismo split / seed / datos** — para aislar el efecto causal
# - **Máximo 3 variantes** — baseline + Var1 + Var2
# - **Nombrado técnico claro** — patrón `Modelo_Hiperparámetro_FeatureSet`
# - **Trazabilidad completa** — logs CSV + decisión final adoptar/descartar
#
# ## Variantes
#
# | Rol | Nombre | Cambio único vs Baseline |
# |-----|--------|--------------------------|
# | A (baseline) | `RF_n200_baseFeats` | — (referencia) |
# | B (Var1) | `RF_n200_biomechFeats` | **features**: +73 biomecánicas de dominio |
# | C (Var2) | `XGB_n200_baseFeats` | **modelo**: RF → XGBoost |

# %% [markdown]
# ## 1. Setup y corrección de directorio de trabajo

# %%
import os
import sys
from pathlib import Path

# Asegurar que el directorio de trabajo sea la raíz del proyecto
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))
print(f"Directorio de trabajo: {project_root}")

# %%
import numpy as np
import pandas as pd
import yaml
import h5py
from loguru import logger as log

from posture_risk.experiments import (
    build_random_forest, build_gradient_boosting,
    cross_validate_pipeline, load_processed_h5, make_shared_splits,
    XGB_AVAILABLE, XGB_BACKEND,
    append_experiment, make_record,
    build_comparison_table, plot_pr_curves_comparison,
    plot_confusion_matrices_comparison, make_decision,
)

# Cargar configuración del experimento
with open("configs/experiments.yaml") as f:
    cfg = yaml.safe_load(f)

print(f"Experimento  : {cfg['experiment']['name']}")
print(f"Branch       : {cfg['experiment']['branch']}")
print(f"Backend GB   : {XGB_BACKEND}")
print(f"Seed         : {cfg['validation']['seed']}")
print(f"CV strategy  : {cfg['validation']['cv_strategy']} (n={cfg['validation']['n_folds']})")

# %% [markdown]
# ## 2. Carga de datasets — base y biomecánico
#
# El dataset biomecánico debe haberse generado previamente:
# ```bash
# python -m posture_risk.experiments.prepare_biomech_dataset --config configs/default.yaml
# ```

# %%
base_h5    = Path(cfg["paths"]["base_h5"])
biomech_h5 = Path(cfg["paths"]["biomech_h5"])

X_base, y, subject_ids = load_processed_h5(base_h5)

if not biomech_h5.exists():
    raise FileNotFoundError(
        f"No existe {biomech_h5}. Genera primero con:\n"
        "  python -m posture_risk.experiments.prepare_biomech_dataset"
    )

with h5py.File(biomech_h5, "r") as f:
    X_biomech_only = f["X_biomech"][:]
    y_check        = f["y"][:]
    subj_check     = f["subject_ids"][:]

# Verificación crítica: alineamiento 1:1 entre ambos HDF5
assert np.array_equal(y, y_check),           "Etiquetas no alineadas entre base y biomech"
assert np.array_equal(subject_ids, subj_check), "Subject IDs no alineados"
print(f"✓ Alineamiento 1:1 verificado: {X_base.shape[0]} ventanas en ambos HDF5")

# Construir feature set Var1 (base + biomecánicas)
X_base_plus_biomech = np.concatenate([X_base, X_biomech_only], axis=1)
print(f"\nFeature sets:")
print(f"  base (Baseline + Var2): {X_base.shape}")
print(f"  base+biomech (Var1)   : {X_base_plus_biomech.shape}")

# %% [markdown]
# ## 3. Generación de splits compartidos
#
# **Este es el paso que garantiza la comparación A/B justa.**
# Los splits se generan UNA SOLA VEZ y se reutilizan en las 3 variantes.
# Cumple la regla del asesor: *"mismo split entre baseline y variantes"*.

# %%
SEED   = cfg["validation"]["seed"]
splits = make_shared_splits(y, subject_ids, seed=SEED)

print(f"Splits generados: {len(splits)} folds (uno por sujeto)")
print(f"Estos mismos splits se aplicarán a las 3 variantes para garantizar")
print(f"que cualquier diferencia en métricas sea atribuible al cambio único de cada variante.")

# %% [markdown]
# ## 4. Ejecución de las 3 variantes

# %%
results = []

# ─── A: BASELINE ──────────────────────────────────────────────────────────────
log.info("\n" + "="*70)
log.info("VARIANTE A — BASELINE: RF_n200_baseFeats")
log.info("="*70)

pipe_baseline = build_random_forest(seed=SEED, n_estimators=200)
r_baseline = cross_validate_pipeline(
    pipe_baseline, X_base, y, subject_ids, splits, "RF_n200_baseFeats"
)
r_baseline["notes"] = "Baseline reproducido: RF + 297 features estadísticas"
results.append(r_baseline)

# ─── B: VAR1 — FEATURE ENGINEERING ─────────────────────────────────────────────
log.info("\n" + "="*70)
log.info("VARIANTE B (Var1) — RF_n200_biomechFeats")
log.info("Cambio único: enriquecimiento con features biomecánicas de dominio")
log.info("="*70)

pipe_var1 = build_random_forest(seed=SEED, n_estimators=200)
r_var1 = cross_validate_pipeline(
    pipe_var1, X_base_plus_biomech, y, subject_ids, splits, "RF_n200_biomechFeats"
)
r_var1["notes"] = "Mismo RF. +73 features biomecánicas (magnitudes, tilt, jerk, SMA, ratios)"
results.append(r_var1)

# ─── C: VAR2 — XGBOOST ────────────────────────────────────────────────────────
log.info("\n" + "="*70)
log.info(f"VARIANTE C (Var2) — XGB_n200_baseFeats   [backend: {XGB_BACKEND}]")
log.info("Cambio único: RF → XGBoost (mismas features del baseline)")
log.info("="*70)

pipe_var2 = build_gradient_boosting(seed=SEED, n_estimators=200)
r_var2 = cross_validate_pipeline(
    pipe_var2, X_base, y, subject_ids, splits, "XGB_n200_baseFeats"
)
r_var2["notes"] = f"Mismas 297 features. Cambio único: modelo → {XGB_BACKEND}"
results.append(r_var2)

# %% [markdown]
# ## 5. Tabla estándar comparativa
#
# Formato exigido por la diapositiva 9 de la sesión 5:
# Modelo | Métrica principal | Métricas secundarias | Costo/Tiempo | Notas

# %%
table_path = Path(cfg["paths"]["table_csv"])
df_comparison = build_comparison_table(results, table_path)

print("\n" + "="*100)
print("TABLA ESTÁNDAR A/B — Resultado del experimento")
print("="*100)
print(df_comparison.to_string(index=False))
print(f"\nTabla guardada en: {table_path}")

# %% [markdown]
# ## 6. Gráfico clave — Curvas Precision-Recall comparadas
#
# Formato exigido por la diapositiva 10 de la sesión 5:
# "PR curve o matriz de confusión comparada"

# %%
figures_dir = Path(cfg["paths"]["figures"])
figures_dir.mkdir(parents=True, exist_ok=True)

pr_path = figures_dir / "05_ab_pr_curves_comparison.png"
plot_pr_curves_comparison(results, pr_path, n_classes=3)
print(f"Gráfico PR guardado en: {pr_path}")

# %% [markdown]
# ## 7. Complemento — Matrices de confusión comparadas

# %%
cm_path = figures_dir / "06_ab_confusion_matrices.png"
plot_confusion_matrices_comparison(results, cm_path)
print(f"Matrices de confusión guardadas en: {cm_path}")

# %% [markdown]
# ## 8. Logging persistente en CSV
#
# Cada variante se registra en `logs/metrics_experimentos.csv`
# con su `exp_id` único, configuración y métricas.

# %%
log_csv_path = Path(cfg["paths"]["log_csv"])

for result in results:
    record = make_record(
        exp_name    = result["exp_name"],
        model       = "random_forest" if result["exp_name"].startswith("RF_") else "gradient_boosting",
        feature_set = "biomech" if "biomechFeats" in result["exp_name"] else "base",
        metrics     = result,
        config      = {
            "cv_strategy": f"{cfg['validation']['cv_strategy']}-LOSO",
            "n_folds":     cfg["validation"]["n_folds"],
            "seed":        SEED,
        },
        notes       = result.get("notes", ""),
        log_path    = str(figures_dir / f"{result['exp_name']}.json"),
        csv_path    = log_csv_path,
    )
    append_experiment(record, log_csv_path)
    print(f"  Registrado: {record['exp_id']} | {record['exp_name']}")

print(f"\nHistorial completo en: {log_csv_path}")

# %% [markdown]
# ## 9. Decisión final: adoptar / descartar
#
# Criterio exigido por la diapositiva 14 de la sesión 5:
# "¿La variante mejora la métrica central de forma consistente?"

# %%
decision = make_decision(
    results,
    threshold_improvement=cfg["decision"]["min_improvement"],
)

print("="*70)
print("DECISIÓN FINAL DEL EXPERIMENTO A/B")
print("="*70)
print(f"\nBaseline: {decision['baseline_name']}")
print(f"  F1-Macro: {decision['baseline_f1']:.4f}\n")
print(f"{'Variante':<28} {'ΔF1':>8} {'ΔPR-AUC':>10} {'ΔTime':>8} {'Decisión':>12}")
print("-" * 70)
for d in decision["decisions"]:
    sign = "+" if d["delta_f1"] >= 0 else ""
    print(
        f"{d['variant_name']:<28} {sign}{d['delta_f1']:>7.4f} "
        f"{sign}{d['delta_pr_auc']:>9.4f} {d['time_overhead_s']:>+7.1f}s "
        f"{d['decision']:>12}"
    )
print("\n" + "="*70)
print(f"Mejor variante global: {decision['best_variant']}")
print("="*70)

# %% [markdown]
# ## 10. Resumen ejecutivo
#
# Este experimento responde dos preguntas con un solo cambio cada una:
#
# 1. **¿Aporta más añadir features biomecánicas que cambiar de modelo?**
#    Var1 vs Var2 → comparar Δ F1-Macro de cada uno respecto al baseline
#
# 2. **¿La mejora justifica el costo computacional adicional?**
#    Verificar columna "ΔTime" de la tabla de decisión
#
# Los resultados se documentan en `docs/experiments/feature-engineering-v1.md`
# junto con la decisión final y los próximos pasos del siguiente sprint.
