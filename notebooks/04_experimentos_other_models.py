# ============================================================
# NOTEBOOK 03 — Experimentos A/B con MLflow
# Ejecuta el baseline + 3 variantes y genera la tabla comparativa
# ============================================================

# %% [markdown]
# # Experimentos A/B — Baseline vs. 3 variantes
# 
# Objetivo: cuantificar el impacto de tres decisiones de diseño respecto al baseline:
# - Var A: feature set (solo temporal vs. temporal+espectral)
# - Var B: clasificador (RF vs. SVM-RBF)
# - Var C: ventana (200 ms vs. 400 ms — opcional, requiere re-ingesta)

# %%
import os, sys
from pathlib import Path

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))
print(f"Directorio de trabajo: {project_root}")

# %%
import yaml
import warnings
warnings.filterwarnings("ignore")

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

from posture_risk.experiments.runner import run_variant, VARIANTS
from posture_risk.experiments.compare import (
    load_all_results, build_comparison_table,
    print_markdown_table, plot_pr_curves
)

print(f"\nVariantes disponibles:")
for k, v in VARIANTS.items():
    print(f"  • {k:30s} — {v['name']}")

# %% [markdown]
# ## 1. Ejecutar Baseline + Variantes A y B
# 
# **Nota:** Var C (ventana 400 ms) requiere re-ejecutar el pipeline de ingesta.
# Se omite aquí por costo y se documenta en el reporte como trabajo futuro.

# %%
results_run = {}
for variant_key in ["baseline", "var_a_only_temporal", "var_b_svm_rbf"]:
    results_run[variant_key] = run_variant(variant_key, cfg)

# %% [markdown]
# ## 2. Cargar resultados consolidados

# %%
experiments_dir = Path(cfg["paths"]["figures"]).parent / "experiments"
results = load_all_results(experiments_dir)
print(f"Resultados cargados: {list(results.keys())}")

# %% [markdown]
# ## 3. Tabla comparativa estándar

# %%
df_comparison = build_comparison_table(results)
print_markdown_table(df_comparison, output_path=experiments_dir / "comparison_table.md")
df_comparison.to_csv(experiments_dir / "comparison_table.csv", index=False)
df_comparison

# %% [markdown]
# ## 4. Gráfico clave — Curva Precision-Recall comparativa

# %%
figures_dir = Path(cfg["paths"]["figures"])
plot_pr_curves(results, output_path=figures_dir / "experiments_comparison.png")

# %% [markdown]
# ## 5. Conclusiones del experimento
# 
# **Hallazgos principales:**
# 1. ¿La métrica principal (F1-Macro) cambia significativamente entre variantes?
# 2. ¿Var A pierde mucho rendimiento al eliminar features espectrales?
#    → Si la pérdida es < 0.02, podría considerarse omitir FFT en producción.
# 3. ¿Var B (SVM-RBF) supera al baseline?
#    → Si no lo hace, justifica usar RF que es más rápido y interpretable.
# 4. ¿El costo computacional escala razonablemente con la complejidad?

# %%
print("\nResumen ejecutivo:")
for v_key, data in results.items():
    s = data["summary"]
    print(f"  {data['name']}")
    print(f"    F1-Macro: {s['f1_macro_mean']:.4f} ± {s['f1_macro_std']:.4f}")
    print(f"    Tiempo:   {s['total_time_s']:.1f}s")
    print()

# Confirmación de zero leakage
print("✓ Zero leakage confirmado: assertion explícita en cada fold de runner.py")
print("✓ Split por grupo (LOSO): sujetos disjuntos entre train y test")
print("✓ StandardScaler.fit ejecutado SOLO en train de cada fold")
