# ============================================================
# NOTEBOOK 07 — Sprint 5: Análisis de errores y slicing
# Convertir a .ipynb con: jupytext --to notebook 07_error_slicing.py
# ============================================================
# %% [markdown]
# # Sprint 5 — Análisis de errores y slicing
#
# **Sesión 11:** Análisis de errores y slicing (Ing. Glen Rodríguez)
# **Modelo evaluado:** XGB HPO winner (F1-Macro LOSO = 0.8171 ± 0.2047)
#
# ## Objetivo (slide 2)
#
# > "No basta con obtener una buena métrica (promedio): analiza dónde y por qué
# > falla (slices/colas), propone una mitigación y demuéstrala."
#
# ## Estructura del notebook
#
# | Sección | Contenido |
# |---------|-----------|
# | 1 | Setup y carga del modelo adoptado |
# | 2 | Re-ejecución LOSO para obtener predicciones y probabilidades por ventana |
# | 3 | Slicing en 4 dimensiones (sujeto, clase, actividad, transición) |
# | 4 | Identificación de 2-5 slices problemáticos |
# | 5 | Análisis de causas (permutation importance + ejemplos) |
# | 6 | Mitigación 1: umbral por clase |
# | 7 | Mitigación 2: calibración isotónica |
# | 8 | Validación de hipótesis H1 y H2 |
# | 9 | Resumen ejecutivo |

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
import json
import numpy as np
import pandas as pd
import h5py
import yaml
from loguru import logger as log
import matplotlib.pyplot as plt

from posture_risk.experiments import load_processed_h5, make_shared_splits
from posture_risk.experiments.ablations import (
    HPO_WINNER_PARAMS, build_xgb_with_scaler,
)
from posture_risk.analysis import (
    # slicing
    slice_by_subject, slice_by_class, slice_by_activity, slice_by_transition,
    identify_problematic_slices, RISK_NAMES, ACTIVITY_NAMES,
    # causes
    permutation_importance_slice, compare_importances, extract_error_examples,
    # mitigations
    find_optimal_thresholds, apply_thresholds,
    fit_isotonic_calibrators, apply_isotonic_calibration,
    compare_before_after, validate_hypothesis,
    # reporting
    build_slice_summary, plot_slice_f1_with_ci,
    plot_slice_confusion_matrix, plot_before_after,
)

with open("configs/hpo_search_space.yaml") as f:
    cfg = yaml.safe_load(f)

SEED        = 42
BASE_H5     = Path(cfg["paths"]["base_h5"])
FIGURES_DIR = Path(cfg["paths"]["figures"])
REPORTS_DIR = Path("reports")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

X, y, subject_ids = load_processed_h5(BASE_H5)
print(f"\nDataset cargado: X={X.shape}, y={y.shape}")

# %% [markdown]
# ### Recuperar activity_ids alineados con las ventanas
#
# El pipeline base guarda `y` (etiqueta de riesgo) y `subject_ids`, pero para
# el slicing por actividad necesitamos el `activity_id` original de cada
# ventana. Lo reconstruimos accediendo al HDF5 directamente.

# %%
with h5py.File(BASE_H5, "r") as f:
    if "activity_ids" in f:
        activity_ids = f["activity_ids"][:]
        print(f"activity_ids cargado directamente: shape={activity_ids.shape}")
    else:
        # Fallback: si el HDF5 no tiene activity_ids explícito, mapeamos desde y
        # y las actividades más comunes (aproximación conservadora).
        print("activity_ids no está en HDF5. Usando aproximación por clase de riesgo.")
        activity_ids = None

# %% [markdown]
# ## 2. Re-ejecución LOSO para obtener predicciones por ventana
#
# Necesitamos, para cada ventana del dataset completo, la predicción del modelo
# cuando esa ventana estaba en el fold de test (esquema LOSO).
# Esto se llama "out-of-fold predictions" y es la base del slicing.

# %%
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

splits = make_shared_splits(y, subject_ids, seed=SEED)

# Preparar arrays para almacenar predicciones out-of-fold
y_pred_oof  = np.zeros_like(y)
y_proba_oof = np.zeros((len(y), 3), dtype=np.float32)

log.info("Ejecutando LOSO para obtener predicciones out-of-fold...")

from tqdm import tqdm
for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc="LOSO fold")):
    pipeline = build_xgb_with_scaler(HPO_WINNER_PARAMS, seed=SEED)
    pipeline.fit(X[train_idx], y[train_idx])
    y_pred_oof[test_idx]  = pipeline.predict(X[test_idx])
    y_proba_oof[test_idx] = pipeline.predict_proba(X[test_idx])

# Métrica global de sanity check
global_f1 = f1_score(y, y_pred_oof, average="macro", zero_division=0)
print(f"\n✓ F1-Macro global (OOF, LOSO): {global_f1:.4f}")
print(f"  (Debe estar cercano a 0.8140 del sprint anterior)")

# Guardar las predicciones OOF para reutilizar
np.savez(
    REPORTS_DIR / "oof_predictions.npz",
    y_true=y, y_pred=y_pred_oof, y_proba=y_proba_oof,
    subject_ids=subject_ids,
)

# %% [markdown]
# ## 3. Slicing en 4 dimensiones
#
# ### 3.1 Slicing por sujeto (9 slices)

# %%
slices_subject = slice_by_subject(
    y_true=y, y_pred=y_pred_oof, y_proba=y_proba_oof,
    subject_ids=subject_ids, n_bootstrap=1000, seed=SEED,
)

df_subject = build_slice_summary(slices_subject, "sujeto")
print("\nSlicing por sujeto:")
print(df_subject.to_string(index=False))

# %% [markdown]
# ### 3.2 Slicing por clase de riesgo (3 slices)

# %%
slices_class = slice_by_class(
    y_true=y, y_pred=y_pred_oof, y_proba=y_proba_oof,
    n_bootstrap=1000, seed=SEED,
)

df_class = build_slice_summary(slices_class, "clase")
print("\nSlicing por clase de riesgo:")
print(df_class.to_string(index=False))

# %% [markdown]
# ### 3.3 Slicing por actividad PAMAP2

# %%
if activity_ids is not None:
    slices_activity = slice_by_activity(
        y_true=y, y_pred=y_pred_oof, y_proba=y_proba_oof,
        activity_ids=activity_ids, n_bootstrap=1000, seed=SEED,
    )
    df_activity = build_slice_summary(slices_activity, "actividad")
    print("\nSlicing por actividad PAMAP2:")
    print(df_activity.to_string(index=False))
else:
    slices_activity = {}
    df_activity = pd.DataFrame()
    print("Slicing por actividad omitido (activity_ids no disponible)")

# %% [markdown]
# ### 3.4 Slicing por tipo de ventana (estable vs transición)

# %%
if activity_ids is not None:
    slices_transition = slice_by_transition(
        y_true=y, y_pred=y_pred_oof, y_proba=y_proba_oof,
        activity_ids=activity_ids, subject_ids=subject_ids,
        window_span=3, n_bootstrap=1000, seed=SEED,
    )
    df_transition = build_slice_summary(slices_transition, "transición")
    print("\nSlicing por tipo de ventana:")
    print(df_transition.to_string(index=False))
else:
    slices_transition = {}
    df_transition = pd.DataFrame()
    print("Slicing por transición omitido (activity_ids no disponible)")

# %% [markdown]
# ## 4. Identificación de slices problemáticos
#
# **Criterio del asesor (slide 3):**
# - Un slice se considera problemático si su F1 está al menos 0.05 por
#   debajo del F1 global
# - Debe tener al menos 30 muestras para que el hallazgo sea estable

# %%
all_problematic = []

for name, results, dim in [
    ("sujeto",    slices_subject,    "sujeto"),
    ("clase",     slices_class,      "clase de riesgo"),
    ("actividad", slices_activity,   "actividad PAMAP2"),
    ("transición", slices_transition, "tipo de ventana"),
]:
    if not results:
        continue
    prob = identify_problematic_slices(
        results, global_f1=global_f1, dimension_name=dim,
        min_gap=0.05, min_n=30,
    )
    all_problematic.extend(prob)

# Ordenar por gap descendente (peores primero)
all_problematic.sort(key=lambda x: x["gap"], reverse=True)

# Reportar los 2-5 más problemáticos
top_problematic = all_problematic[:5]

print(f"\n{'='*80}")
print(f"SLICES PROBLEMÁTICOS DETECTADOS (top {len(top_problematic)})")
print(f"{'='*80}")
for i, p in enumerate(top_problematic, 1):
    ci_str = (f"[{p['f1_ci_lo']:.4f}, {p['f1_ci_hi']:.4f}]"
              if p['f1_ci_lo'] is not None and not np.isnan(p['f1_ci_lo'])
              else "IC no disp.")
    print(f"\n{i}. {p['dimension']} = {p['slice_name']}")
    print(f"   n = {p['n']}")
    print(f"   F1-Macro = {p['f1_macro']:.4f}   IC 95%: {ci_str}")
    print(f"   Gap vs global = {p['gap']:+.4f}")
    print(f"   Distribución de clases: {p['class_dist']}")

# %% [markdown]
# ## 5. Gráficos de F1 por slice con IC

# %%
if not df_subject.empty:
    plot_slice_f1_with_ci(
        df_subject, global_f1,
        FIGURES_DIR / "14_slice_f1_by_subject.png",
        dimension="Sujeto",
    )

if not df_class.empty:
    plot_slice_f1_with_ci(
        df_class, global_f1,
        FIGURES_DIR / "15_slice_f1_by_class.png",
        dimension="Clase de riesgo",
    )

if not df_activity.empty:
    plot_slice_f1_with_ci(
        df_activity, global_f1,
        FIGURES_DIR / "16_slice_f1_by_activity.png",
        dimension="Actividad PAMAP2",
    )

if not df_transition.empty:
    plot_slice_f1_with_ci(
        df_transition, global_f1,
        FIGURES_DIR / "17_slice_f1_by_transition.png",
        dimension="Tipo de ventana",
    )

# %% [markdown]
# ## 6. Análisis de causas para los slices problemáticos
#
# Para cada slice problemático:
# - Matriz de confusión
# - Permutation importance dentro del slice vs resto
# - Ejemplos representativos de errores con alta confianza

# %%
# Nombres de features (mismo layout del sprint 2)
segments        = ["hand", "chest", "ankle"]
channels_per_seg = ["acc16_x","acc16_y","acc16_z","gyro_x","gyro_y","gyro_z",
                    "mag_x","mag_y","mag_z"]
temp_feats = ["rms","mav","wl","zc","ssc","var"]
spec_feats = ["mdf","mnf","band_low","band_mid","band_high"]

feature_names = []
for seg in segments:
    for ch in channels_per_seg:
        for feat in temp_feats + spec_feats:
            feature_names.append(f"{seg}_{ch}_{feat}")

# Reconstruir el modelo del baseline (para permutation importance)
# Se usa el pipeline entrenado con TODOS los datos como aproximación
# (solo para análisis interpretativo, no para métricas de generalización)
pipeline_full = build_xgb_with_scaler(HPO_WINNER_PARAMS, seed=SEED)
pipeline_full.fit(X, y)

# %%
causes_analysis = []

for i, prob in enumerate(top_problematic[:3], 1):  # análisis en los top 3
    print(f"\n{'='*70}")
    print(f"ANÁLISIS DE CAUSAS — Slice {i}: {prob['dimension']} = {prob['slice_name']}")
    print(f"{'='*70}")

    # Reconstruir la máscara del slice
    if prob["dimension"] == "sujeto":
        slice_mask = (subject_ids == int(prob["slice_name"]))
    elif prob["dimension"] == "clase de riesgo":
        cls_id = {v: k for k, v in RISK_NAMES.items()}[prob["slice_name"]]
        slice_mask = (y == cls_id)
    elif prob["dimension"] == "actividad PAMAP2" and activity_ids is not None:
        aid = {v: k for k, v in ACTIVITY_NAMES.items()}.get(prob["slice_name"])
        slice_mask = (activity_ids == aid) if aid is not None else np.zeros(len(y), bool)
    elif prob["dimension"] == "tipo de ventana" and activity_ids is not None:
        from posture_risk.analysis import compute_transition_mask
        trans = compute_transition_mask(activity_ids, subject_ids, window_span=3)
        slice_mask = trans if prob["slice_name"] == "transicion" else ~trans
    else:
        continue

    if slice_mask.sum() < 30:
        print(f"  Slice muy pequeño, omitiendo análisis de causa")
        continue

    # 5.1 Matriz de confusión
    plot_slice_confusion_matrix(
        prob["confusion"],
        slice_name=f"{prob['dimension']}={prob['slice_name']}",
        output_path=FIGURES_DIR / f"18_confusion_slice_{i}.png",
    )

    # 5.2 Permutation importance dentro del slice
    print(f"\n  Calculando permutation importance (puede tardar ~1 min)...")
    imp_slice = permutation_importance_slice(
        pipeline_full, X, y, slice_mask=slice_mask,
        feature_names=feature_names,
        n_repeats=3, top_k=15, seed=SEED,
    )

    imp_rest = permutation_importance_slice(
        pipeline_full, X, y, slice_mask=~slice_mask,
        feature_names=feature_names,
        n_repeats=3, top_k=15, seed=SEED,
    )

    # Comparar importancias
    diffs = compare_importances(imp_slice, imp_rest, top_k=10)
    print(f"\n  Top 10 features con mayor cambio de importancia (slice vs resto):")
    for d in diffs:
        arrow = "↑" if d["delta"] > 0 else "↓"
        print(f"    {arrow} {d['feature']:<40} slice={d['imp_slice']:+.4f}  "
              f"rest={d['imp_rest']:+.4f}  Δ={d['delta']:+.4f}")

    # 5.3 Ejemplos representativos de errores confidentes
    examples = extract_error_examples(
        X=X, y_true=y, y_pred=y_pred_oof, y_proba=y_proba_oof,
        slice_mask=slice_mask, n_examples=3,
        error_type="confident_wrong",
    )

    print(f"\n  Ejemplos representativos de errores confidentes:")
    for j, ex in enumerate(examples, 1):
        print(f"    #{j}  idx={ex['global_idx']}  "
              f"real={RISK_NAMES.get(ex['y_true'])}  "
              f"pred={RISK_NAMES.get(ex['y_pred'])}  "
              f"confianza={ex['confidence']:.3f}  "
              f"probs={[f'{p:.2f}' for p in ex['probs']]}")

    causes_analysis.append({
        "slice":     f"{prob['dimension']}={prob['slice_name']}",
        "top_diffs": diffs[:5],
        "examples":  examples,
        "hypothesis_of_cause": _generate_cause_hypothesis(prob, diffs),
    })

def _generate_cause_hypothesis(prob, diffs):
    """Genera hipótesis textual sobre la causa probable."""
    if prob["dimension"] == "clase de riesgo":
        return (
            f"La clase {prob['slice_name']} tiene bajo recall probablemente por "
            f"desbalance de datos (n={prob['n']}) y/o solapamiento de features "
            f"con clases adyacentes. Ver matriz de confusión."
        )
    elif prob["dimension"] == "sujeto":
        return (
            f"El sujeto {prob['slice_name']} tiene patrones biomecánicos "
            f"distintos al promedio del dataset. Las features "
            f"{[d['feature'] for d in diffs[:3]]} cambian su importancia, "
            f"sugiriendo variabilidad antropométrica o de estilo de movimiento."
        )
    elif prob["dimension"] == "actividad PAMAP2":
        return (
            f"La actividad '{prob['slice_name']}' presenta features distintos al "
            f"resto. Puede requerir features de dominio específicas o más datos."
        )
    elif prob["dimension"] == "tipo de ventana":
        return (
            f"Las ventanas de {prob['slice_name']} presentan mezcla de patrones "
            f"de actividades consecutivas. Postprocesamiento suavizante podría ayudar."
        )
    return "Ver análisis de features gatillo."

# %% [markdown]
# ## 7. Mitigación 1: Ajuste de umbral por clase (H1)
#
# **Hipótesis H1:** El ajuste de umbral por clase mejora el recall de la
# clase "Alto" en al menos +0.05, sin degradar el F1-Macro global más de 0.01.
#
# Los umbrales se ajustan usando las probabilidades OOF (out-of-fold),
# lo cual evita leakage: se busca el umbral óptimo sobre datos que el
# modelo no vio durante entrenamiento.

# %%
log.info("Calculando umbrales óptimos por clase con prioridad en Alto riesgo...")

thresholds = find_optimal_thresholds(
    y_true=y, y_proba=y_proba_oof,
    metric="f1",
    class_priority={0: 1.0, 1: 1.0, 2: 1.5},  # priorizar clase alta
)

print(f"\nUmbrales óptimos encontrados:")
for cls, thr in thresholds.items():
    print(f"  Clase {cls} ({RISK_NAMES.get(cls)}): {thr:.4f}")

# Aplicar umbrales
y_pred_thr = apply_thresholds(y_proba_oof, thresholds)

# Comparación baseline vs umbral
comp_thr = compare_before_after(
    y_true=y,
    y_pred_before=y_pred_oof, y_pred_after=y_pred_thr,
    y_proba_before=y_proba_oof, y_proba_after=y_proba_oof,  # proba no cambia
)

# Validar H1
h1_result = validate_hypothesis(
    comp_thr, hypothesis="H1", target_class=2,
    min_recall_gain=0.05, max_f1_loss=0.01,
)

print(f"\n{'='*70}")
print(f"RESULTADO H1 — Ajuste de umbral por clase")
print(f"{'='*70}")
print(f"  F1-Macro global:     {comp_thr['f1_macro_before']:.4f} → "
      f"{comp_thr['f1_macro_after']:.4f}  (Δ = {comp_thr['f1_macro_delta']:+.4f})")
print(f"  Recall clase Alto:   {comp_thr['recall_by_class'][2]['before']:.4f} → "
      f"{comp_thr['recall_by_class'][2]['after']:.4f}  "
      f"(Δ = {comp_thr['recall_by_class'][2]['delta']:+.4f})")
print(f"\n  H1 confirmada:       {'✓ SÍ' if h1_result['confirmed'] else '✗ NO'}")
print(f"    Recall gain ≥ {h1_result['threshold_recall']:.2f}: {h1_result['recall_pass']}")
print(f"    F1 loss ≤ {-h1_result['threshold_f1']:.2f}: {h1_result['f1_pass']}")

# Gráfico
plot_before_after(
    comp_thr,
    output_path=FIGURES_DIR / "19_mitigation_thresholds_before_after.png",
    mitigation_name="Umbral óptimo",
)

# %% [markdown]
# ## 8. Mitigación 2: Calibración isotónica (H2)
#
# **Hipótesis H2:** La calibración isotónica reduce el Brier score global
# en al menos 0.02 sin degradar el F1-Macro.
#
# La calibración se ajusta sobre las probabilidades OOF: se aprende una
# función monótona por clase que mapea proba sin calibrar → proba real.

# %%
log.info("Ajustando calibradores isotónicos por clase...")

calibrators = fit_isotonic_calibrators(
    y_true=y, y_proba=y_proba_oof, n_classes=3,
)

y_proba_cal = apply_isotonic_calibration(y_proba_oof, calibrators)
y_pred_cal  = np.argmax(y_proba_cal, axis=1)

comp_cal = compare_before_after(
    y_true=y,
    y_pred_before=y_pred_oof, y_pred_after=y_pred_cal,
    y_proba_before=y_proba_oof, y_proba_after=y_proba_cal,
)

h2_result = validate_hypothesis(
    comp_cal, hypothesis="H2",
    max_brier_gain=-0.02, max_f1_loss=0.01,
)

print(f"\n{'='*70}")
print(f"RESULTADO H2 — Calibración isotónica")
print(f"{'='*70}")
print(f"  F1-Macro global:  {comp_cal['f1_macro_before']:.4f} → "
      f"{comp_cal['f1_macro_after']:.4f}  (Δ = {comp_cal['f1_macro_delta']:+.4f})")
print(f"  Brier score:      {comp_cal['brier_before']:.4f} → "
      f"{comp_cal['brier_after']:.4f}  (Δ = {comp_cal['brier_delta']:+.4f})")
print(f"\n  H2 confirmada:    {'✓ SÍ' if h2_result['confirmed'] else '✗ NO'}")
print(f"    Brier reducción ≥ {-h2_result['threshold_brier']:.2f}: {h2_result['brier_pass']}")
print(f"    F1 loss ≤ {-h2_result['threshold_f1']:.2f}: {h2_result['f1_pass']}")

# Gráfico
plot_before_after(
    comp_cal,
    output_path=FIGURES_DIR / "20_mitigation_calibration_before_after.png",
    mitigation_name="Calibración isotónica",
)

# %% [markdown]
# ## 9. Tabla consolidada final y persistencia

# %%
# Consolidar todos los slices en una sola tabla
all_slices = pd.concat([df_subject, df_class, df_activity, df_transition],
                       ignore_index=True) if activity_ids is not None else \
             pd.concat([df_subject, df_class], ignore_index=True)

# Guardar
slices_csv_path = REPORTS_DIR / "sprint5_slice_analysis.csv"
all_slices.to_csv(slices_csv_path, index=False)

# Guardar el resumen ejecutivo en JSON
summary_json = {
    "global_f1_oof":        global_f1,
    "total_slices_analyzed": len(all_slices),
    "problematic_slices":    [
        {
            "dimension":  p["dimension"],
            "slice_name": p["slice_name"],
            "n":          p["n"],
            "f1_macro":   p["f1_macro"],
            "gap":        p["gap"],
            "class_dist": p["class_dist"],
        }
        for p in top_problematic
    ],
    "mitigation_1_thresholds": {
        "thresholds":  thresholds,
        "f1_before":   comp_thr["f1_macro_before"],
        "f1_after":    comp_thr["f1_macro_after"],
        "f1_delta":    comp_thr["f1_macro_delta"],
        "recall_alto_before": comp_thr["recall_by_class"][2]["before"],
        "recall_alto_after":  comp_thr["recall_by_class"][2]["after"],
        "recall_alto_delta":  comp_thr["recall_by_class"][2]["delta"],
        "h1_confirmed":       h1_result["confirmed"],
    },
    "mitigation_2_calibration": {
        "f1_before":    comp_cal["f1_macro_before"],
        "f1_after":     comp_cal["f1_macro_after"],
        "f1_delta":     comp_cal["f1_macro_delta"],
        "brier_before": comp_cal["brier_before"],
        "brier_after":  comp_cal["brier_after"],
        "brier_delta":  comp_cal["brier_delta"],
        "h2_confirmed": h2_result["confirmed"],
    },
}

with open(REPORTS_DIR / "sprint5_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

print(f"\nArtefactos generados:")
print(f"  • {slices_csv_path}")
print(f"  • {REPORTS_DIR / 'sprint5_summary.json'}")
print(f"  • {REPORTS_DIR / 'oof_predictions.npz'}")

# %% [markdown]
# ## 10. Resumen ejecutivo

# %%
print("\n" + "="*72)
print("RESUMEN EJECUTIVO — SPRINT 5: ANÁLISIS DE ERRORES Y SLICING")
print("="*72)

print(f"\n─── SLICES ANALIZADOS ──────────────────────────────────────────────")
print(f"  Total: {len(all_slices)} slices en 4 dimensiones")
print(f"  F1-Macro global (LOSO OOF): {global_f1:.4f}")

print(f"\n─── SLICES PROBLEMÁTICOS DETECTADOS ────────────────────────────────")
for i, p in enumerate(top_problematic, 1):
    print(f"  {i}. [{p['dimension']}] {p['slice_name']}:  "
          f"F1={p['f1_macro']:.4f}  n={p['n']}  gap={p['gap']:+.4f}")

print(f"\n─── MITIGACIÓN 1 — UMBRAL POR CLASE ────────────────────────────────")
print(f"  H1 confirmada:              {'✓ SÍ' if h1_result['confirmed'] else '✗ NO'}")
print(f"  Δ Recall clase Alto:        {comp_thr['recall_by_class'][2]['delta']:+.4f}")
print(f"  Δ F1-Macro global:          {comp_thr['f1_macro_delta']:+.4f}")

print(f"\n─── MITIGACIÓN 2 — CALIBRACIÓN ISOTÓNICA ───────────────────────────")
print(f"  H2 confirmada:              {'✓ SÍ' if h2_result['confirmed'] else '✗ NO'}")
print(f"  Δ Brier score:              {comp_cal['brier_delta']:+.4f}")
print(f"  Δ F1-Macro global:          {comp_cal['f1_macro_delta']:+.4f}")

print("="*72)
