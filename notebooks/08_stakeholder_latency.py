# ============================================================
# NOTEBOOK 08 — Sprint 6: Prueba con stakeholder y latencia
# Convertir a .ipynb con: jupytext --to notebook 08_stakeholder_latency.py
# ============================================================
# %% [markdown]
# # Sprint 6 — Prueba con stakeholder y latencia
#
# **Sesión 12:** Prueba con stakeholder y latencia
# **Meta:** validar utilidad (¿sirve para el usuario?)
# y desempeño operativo (¿responde rápido / es estable?).

# %% [markdown]
# ## 1. Setup y carga de modelos

# %%
import os, sys
from pathlib import Path

project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(project_root)
sys.path.insert(0, str(project_root / "src"))
print(f"Directorio de trabajo: {project_root}")

# %%
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger as log

from posture_risk.experiments import (
    load_processed_h5, build_random_forest,
)
from posture_risk.experiments.ablations import (
    HPO_WINNER_PARAMS, build_xgb_with_scaler,
)
from posture_risk.analysis import (
    fit_isotonic_calibrators, apply_isotonic_calibration,
    find_optimal_thresholds, apply_thresholds,
)
from posture_risk.latency import (
    SLOs, evaluate_all_slos, export_slo_report,
    measure_stage_latency, measure_pipeline_latency, measure_throughput,
    format_latency_table,
    run_pruning_experiment,
)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

SEED = 42
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ### 1.1 Cargar datos y predicciones OOF del Sprint 5

# %%
X, y, subject_ids = load_processed_h5(Path("data/processed/pamap2_features.h5"))
print(f"Dataset: X={X.shape}, y={y.shape}")

oof_path = REPORTS_DIR / "oof_predictions.npz"
oof_data = np.load(oof_path)
y_pred_oof  = oof_data["y_pred"]
y_proba_oof = oof_data["y_proba"]

# Reajustar calibradores y umbrales sobre OOF (como en Sprint 5)
calibrators = fit_isotonic_calibrators(y_true=y, y_proba=y_proba_oof, n_classes=3)
thresholds  = find_optimal_thresholds(
    y_true=y, y_proba=y_proba_oof, metric="f1",
    class_priority={0: 1.0, 1: 1.0, 2: 1.5},
)
print(f"Umbrales óptimos: {thresholds}")

# %% [markdown]
# ### 1.2 Entrenar los modelos para las mediciones de latencia

# %%
log.info("Entrenando baseline RF...")
baseline_pipeline = build_random_forest(seed=SEED, n_estimators=200)
baseline_pipeline.fit(X, y)

log.info("Entrenando modelo actual XGB HPO winner...")
actual_pipeline = build_xgb_with_scaler(HPO_WINNER_PARAMS, seed=SEED)
actual_pipeline.fit(X, y)

# %% [markdown]
# ## 2. Declaración formal de los SLOs adoptados

# %%
print("\n" + "="*72)
print("SLOs ADOPTADOS PARA EL SISTEMA")
print("="*72)
for key, slo in SLOs.items():
    marker = " [CRÍTICO]" if slo.critical else ""
    print(f"\n  [{key}]{marker}")
    print(f"    {slo.name}")
    print(f"    Umbral: {slo.comparison} {slo.threshold} {slo.unit}")

# %% [markdown]
# ## 3. Definición de pipelines por etapa
#
# **Modelo actual (5 etapas):** preprocess → scale → infer_xgb → calibrate → threshold
#
# **Baseline RF (3 etapas):** scale → infer_rf → argmax

# %%
xgb_scaler = actual_pipeline.named_steps["scaler"]
xgb_model  = actual_pipeline.named_steps["clf"]
rf_scaler  = baseline_pipeline.named_steps["scaler"]
rf_model   = baseline_pipeline.named_steps["clf"]

sample_input = X[100:101]   # 1 ventana de 297 features

def stage_preprocess(x): return x.astype(np.float32)
def stage_scale_xgb(x):  return xgb_scaler.transform(x)
def stage_infer_xgb(x):  return xgb_model.predict_proba(x)
def stage_calibrate(x):  return apply_isotonic_calibration(x, calibrators)
def stage_threshold(x):  return apply_thresholds(x, thresholds)

actual_stages = {
    "1_preprocess": stage_preprocess,
    "2_scale":      stage_scale_xgb,
    "3_infer_xgb":  stage_infer_xgb,
    "4_calibrate":  stage_calibrate,
    "5_threshold":  stage_threshold,
}

def stage_scale_rf(x): return rf_scaler.transform(x)
def stage_infer_rf(x): return rf_model.predict_proba(x)
def stage_argmax(x):   return np.argmax(x, axis=1)

baseline_stages = {
    "1_scale":    stage_scale_rf,
    "2_infer_rf": stage_infer_rf,
    "3_argmax":   stage_argmax,
}

# %% [markdown]
# ## 4. Perfilado de latencia por etapa

# %%
log.info("Perfilado del modelo actual (5 etapas)...")
actual_lat = measure_pipeline_latency(
    stages=actual_stages, input_sample=sample_input,
    n_repeats=500, n_warmup=50,
)

log.info("Perfilado del baseline RF (3 etapas)...")
baseline_lat = measure_pipeline_latency(
    stages=baseline_stages, input_sample=sample_input,
    n_repeats=500, n_warmup=50,
)

print(format_latency_table(actual_lat,   title="MODELO ACTUAL — Latencia por etapa"))
print(format_latency_table(baseline_lat, title="BASELINE RF   — Latencia por etapa"))

# %% [markdown]
# ## 5. Throughput sostenido (10 segundos)

# %%
def full_actual_pipeline(x):
    x = stage_preprocess(x); x = stage_scale_xgb(x); x = stage_infer_xgb(x)
    x = stage_calibrate(x);  x = stage_threshold(x)
    return x

def full_baseline_pipeline(x):
    x = stage_scale_rf(x); x = stage_infer_rf(x); x = stage_argmax(x)
    return x

throughput_samples = [X[i:i+1] for i in range(100)]

log.info("Midiendo throughput sostenido — modelo actual...")
throughput_actual = measure_throughput(
    pipeline_fn=full_actual_pipeline, input_samples=throughput_samples,
    duration_s=10.0, n_warmup=100,
)

log.info("Midiendo throughput sostenido — baseline...")
throughput_baseline = measure_throughput(
    pipeline_fn=full_baseline_pipeline, input_samples=throughput_samples,
    duration_s=10.0, n_warmup=100,
)

print("\n" + "="*60)
print("THROUGHPUT SOSTENIDO (10 segundos)")
print("="*60)
print(f"  Modelo actual: {throughput_actual['throughput_rps']:>8.1f} pred/s")
print(f"  Baseline RF:   {throughput_baseline['throughput_rps']:>8.1f} pred/s")

# %% [markdown]
# ## 6. Optimización: poda del ensemble XGBoost

# %%
log.info("Ejecutando experimento de poda (puede tardar ~15 minutos)...")
pruning_results = run_pruning_experiment(
    X, y, subject_ids,
    n_estimator_grid=[25, 50, 75, 100, 125, 150, 175, 200],
    n_folds=3, seed=SEED, max_f1_loss=0.01,
)

pruning_df = pd.DataFrame([
    {
        "n_arboles": v["n_estimators"],
        "F1-Macro":  f"{v['f1_macro_mean']:.4f} ± {v['f1_macro_std']:.4f}",
        "Latencia p50 (ms)": f"{v['latency_p50_ms']:.3f}",
        "Latencia p95 (ms)": f"{v['latency_p95_ms']:.3f}",
    }
    for v in pruning_results["variants"].values()
]).sort_values("n_arboles").reset_index(drop=True)

print("\n" + "="*70)
print("EXPERIMENTO DE PODA DEL ENSEMBLE XGBOOST")
print("="*70)
print(pruning_df.to_string(index=False))
print(f"\nSelección: n_estimators = {pruning_results['selected_n_estimators']}")
print(f"  F1-Macro:     {pruning_results['selected_f1']:.4f}")
print(f"  Latencia p95: {pruning_results['selected_latency_p95']:.3f} ms")
if pruning_results['latency_reduction_pct'] is not None:
    print(f"  Reducción latencia p95 vs 200 árboles: "
          f"{pruning_results['latency_reduction_pct']:.1f}%")

# Gráfico F1 vs latencia
fig, ax1 = plt.subplots(figsize=(9, 5))
n_ests = sorted(pruning_results["variants"].keys())
f1s    = [pruning_results["variants"][n]["f1_macro_mean"] for n in n_ests]
lats   = [pruning_results["variants"][n]["latency_p95_ms"] for n in n_ests]

ax1.plot(n_ests, f1s, "o-", color="#1D9E75", lw=2, label="F1-Macro (CV 3-fold)")
ax1.set_xlabel("Número de árboles")
ax1.set_ylabel("F1-Macro", color="#1D9E75")
ax1.tick_params(axis="y", labelcolor="#1D9E75")
ax1.grid(alpha=0.2)

ax2 = ax1.twinx()
ax2.plot(n_ests, lats, "s-", color="#D85A30", lw=2, label="Latencia p95 (ms)")
ax2.set_ylabel("Latencia p95 (ms)", color="#D85A30")
ax2.tick_params(axis="y", labelcolor="#D85A30")

sel = pruning_results["selected_n_estimators"]
ax1.axvline(sel, color="black", ls="--", alpha=0.5,
            label=f"Selección: {sel} árboles")

ax1.set_title(f"Trade-off F1-Macro vs Latencia por número de árboles\n"
              f"Selección: {sel} árboles",
              fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
plt.tight_layout()
fig_path = FIGURES_DIR / "21_pruning_tradeoff.png"
plt.savefig(fig_path, bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# ## 7. A/B controlado simulado sobre las 194,205 ventanas

# %%
from sklearn.metrics import f1_score, precision_score, recall_score

y_proba_oof_cal = apply_isotonic_calibration(y_proba_oof, calibrators)
y_pred_actual = apply_thresholds(y_proba_oof_cal, thresholds)

log.info("Generando predicciones del baseline para A/B...")
y_pred_baseline = baseline_pipeline.predict(X)

def metric_bundle(y_true, y_pred, name):
    return {
        "modelo":  name,
        "F1-Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision (alto)": precision_score(y_true, y_pred, labels=[2],
                                             average="macro", zero_division=0),
        "Recall (alto)": recall_score(y_true, y_pred, labels=[2],
                                       average="macro", zero_division=0),
    }

metrics_baseline = metric_bundle(y, y_pred_baseline, "Baseline RF")
metrics_actual   = metric_bundle(y, y_pred_actual,   "Modelo actual")

n_total = len(y)
alerts_baseline  = int((y_pred_baseline == 2).sum())
alerts_actual    = int((y_pred_actual == 2).sum())
false_positives_baseline = int(((y_pred_baseline == 2) & (y != 2)).sum())
false_positives_actual   = int(((y_pred_actual   == 2) & (y != 2)).sum())
missed_high_baseline = int(((y_pred_baseline != 2) & (y == 2)).sum())
missed_high_actual   = int(((y_pred_actual   != 2) & (y == 2)).sum())

# 1 hora = 3600 s × 5 pred/s = 18000 predicciones
alertas_por_hora_baseline = alerts_baseline / n_total * 18000
alertas_por_hora_actual   = alerts_actual   / n_total * 18000
falsas_alarmas_por_hora_baseline = false_positives_baseline / n_total * 18000
falsas_alarmas_por_hora_actual   = false_positives_actual   / n_total * 18000

print("\n" + "="*75)
print("A/B CONTROLADO SIMULADO (194,205 ventanas)")
print("="*75)
print(f"\nMétricas técnicas:")
print(f"  {'Métrica':<25} {'Baseline':>13} {'Actual':>13}   Δ")
for k in ["F1-Macro", "Precision (alto)", "Recall (alto)"]:
    v_base = metrics_baseline[k]
    v_act  = metrics_actual[k]
    print(f"  {k:<25} {v_base:>13.4f} {v_act:>13.4f}   {v_act-v_base:+.4f}")

print(f"\nProyección por hora de operación (5 pred/s):")
print(f"  Alertas emitidas/hora:  "
      f"{alertas_por_hora_baseline:>10.1f}   {alertas_por_hora_actual:>10.1f}")
print(f"  Falsas alarmas/hora:    "
      f"{falsas_alarmas_por_hora_baseline:>10.1f}   {falsas_alarmas_por_hora_actual:>10.1f}")

# %% [markdown]
# ## 8. Evaluación de SLOs

# %%
measurements = {
    "latency_p95_ms":         actual_lat["total"]["p95_ms"],
    "latency_p50_ms":         actual_lat["total"]["p50_ms"],
    "throughput_req_per_sec": throughput_actual["throughput_rps"],
    "error_rate_pct":         actual_lat["total"]["error_rate_pct"],
    "f1_macro_min":           metrics_actual["F1-Macro"],
}

slo_report = evaluate_all_slos(measurements)

print("\n" + "="*72)
print("EVALUACIÓN DE SLOs")
print("="*72)
for key, res in slo_report["slos"].items():
    icon = "PASS" if res["passed"] else "FAIL"
    crit = " [CRÍTICO]" if res.get("critical") else ""
    print(f"\n  [{icon}]{crit} {res['slo_name']}")
    print(f"      Umbral: {res['comparison']} {res['threshold']} {res['unit']}")
    print(f"      Medido: {res['measured']:.3f} {res['unit']}")

print(f"\n{'='*72}")
if slo_report["deployment_ready"]:
    print("VEREDICTO: sistema APTO para despliegue (todos los SLOs críticos pasan)")
else:
    print("VEREDICTO: sistema NO APTO. SLOs críticos que fallan:")
    for s in slo_report["critical_failing"]:
        print(f"   • {s}")

# %% [markdown]
# ## 9. Tabla comparativa final baseline vs actual

# %%
comparison_table = pd.DataFrame([
    {"Métrica": "F1-Macro (CV LOSO)", "Baseline RF": "0.7942 ± 0.2003",
     "Modelo actual": f"{metrics_actual['F1-Macro']:.4f}"},
    {"Métrica": "Recall clase Alto",
     "Baseline RF": f"{metrics_baseline['Recall (alto)']:.4f}",
     "Modelo actual": f"{metrics_actual['Recall (alto)']:.4f}"},
    {"Métrica": "Latencia p50 (ms)",
     "Baseline RF": f"{baseline_lat['total']['p50_ms']:.3f}",
     "Modelo actual": f"{actual_lat['total']['p50_ms']:.3f}"},
    {"Métrica": "Latencia p95 (ms)",
     "Baseline RF": f"{baseline_lat['total']['p95_ms']:.3f}",
     "Modelo actual": f"{actual_lat['total']['p95_ms']:.3f}"},
    {"Métrica": "Latencia p99 (ms)",
     "Baseline RF": f"{baseline_lat['total']['p99_ms']:.3f}",
     "Modelo actual": f"{actual_lat['total']['p99_ms']:.3f}"},
    {"Métrica": "Throughput (pred/s)",
     "Baseline RF": f"{throughput_baseline['throughput_rps']:.1f}",
     "Modelo actual": f"{throughput_actual['throughput_rps']:.1f}"},
    {"Métrica": "Falsas alarmas/hora",
     "Baseline RF": f"{falsas_alarmas_por_hora_baseline:.1f}",
     "Modelo actual": f"{falsas_alarmas_por_hora_actual:.1f}"},
])

print("\n" + "="*80)
print("COMPARATIVO BASELINE vs MODELO ACTUAL")
print("="*80)
print(comparison_table.to_string(index=False))

comparison_table.to_csv(REPORTS_DIR / "sprint6_comparison.csv", index=False)

# %% [markdown]
# ## 10. Persistencia

# %%
final_report = {
    "slos":              slo_report,
    "latency_baseline":  baseline_lat,
    "latency_actual":    actual_lat,
    "throughput_baseline": throughput_baseline,
    "throughput_actual":   throughput_actual,
    "pruning": {
        "baseline_n_est":       200,
        "baseline_f1":          pruning_results["baseline_f1"],
        "baseline_latency_p95": pruning_results["baseline_latency_p95"],
        "selected_n_est":       pruning_results["selected_n_estimators"],
        "selected_f1":          pruning_results["selected_f1"],
        "selected_latency_p95": pruning_results["selected_latency_p95"],
        "latency_reduction_pct": pruning_results["latency_reduction_pct"],
    },
    "ab_comparison": {
        "metrics_baseline": metrics_baseline,
        "metrics_actual":   metrics_actual,
        "false_alarms_per_hour_baseline": falsas_alarmas_por_hora_baseline,
        "false_alarms_per_hour_actual":   falsas_alarmas_por_hora_actual,
    },
}

with open(REPORTS_DIR / "sprint6_summary.json", "w") as f:
    json.dump(final_report, f, indent=2, default=str)

export_slo_report(slo_report, REPORTS_DIR / "sprint6_slo_report.json")

print(f"\nArtefactos generados:")
print(f"  • {REPORTS_DIR / 'sprint6_summary.json'}")
print(f"  • {REPORTS_DIR / 'sprint6_slo_report.json'}")
print(f"  • {REPORTS_DIR / 'sprint6_comparison.csv'}")
print(f"  • {fig_path}")
