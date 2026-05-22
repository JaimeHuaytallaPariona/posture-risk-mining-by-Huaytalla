"""
reporting.py
------------
Genera la tabla estándar y el gráfico clave para reporte A/B.

Cumple el formato exigido por el asesor (diapositivas 9 y 10 de la sesión 5):
  - Tabla: Baseline / Var1 / Var2 → métrica principal + secundarias + costo
  - Gráfico único: curvas PR por clase con las 3 variantes superpuestas
  - Complemento: matrices de confusión comparadas en panel
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve,
)
from sklearn.preprocessing import label_binarize


# Convenciones visuales del proyecto
RISK_LABELS = ["Bajo (0)", "Medio (1)", "Alto (2)"]
VARIANT_COLORS = {
    "baseline": "#534AB7",  # púrpura — A
    "var1":     "#1D9E75",  # verde teal — B
    "var2":     "#D85A30",  # coral — C
}

plt.rcParams.update({
    "figure.dpi":          120,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "font.size":           11,
})


# ─── Tabla estándar ───────────────────────────────────────────────────────────

def build_comparison_table(results: List[Dict], output_path: Path) -> pd.DataFrame:
    """
    Construye la tabla estándar exigida por la consigna.

    Formato (sesión 5, diapositiva 9):
      Modelo | F1-Macro | PR-AUC | Accuracy | Tiempo (s) | Notas

    El asesor especifica: "métrica principal + secundarias + costo/latencia".
    """
    rows = []
    for r in results:
        rows.append({
            "Modelo":              r["exp_name"],
            "F1-Macro":            f"{r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}",
            "PR-AUC (OvR)":        f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f}",
            "Accuracy":            f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
            "Tiempo (s)":          f"{r['train_time_s']:.1f}",
            "N features":          r["n_features"],
            "Notas":               r.get("notes", ""),
        })

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


# ─── Gráfico clave: curvas PR comparadas ──────────────────────────────────────

def plot_pr_curves_comparison(
    results: List[Dict],
    output_path: Path,
    n_classes: int = 3,
) -> Path:
    """
    Gráfico único: curvas Precision-Recall por clase superpuestas
    para las 3 variantes.

    Por qué este gráfico:
      Es el que el asesor menciona específicamente en la diapositiva 10
      de la sesión 5: "PR curve o matriz de confusión comparada".
      Para clasificación con desbalance la PR curve es más informativa
      que ROC porque enfatiza el desempeño en la clase minoritaria.
    """
    fig, axes = plt.subplots(1, n_classes, figsize=(16, 5), sharey=True)

    variant_keys = ["baseline", "var1", "var2"]

    for class_idx in range(n_classes):
        ax = axes[class_idx]
        for variant_key, result in zip(variant_keys, results):
            y_true = result["y_true_all"]
            y_proba = result["y_proba_all"]

            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            try:
                precision, recall, _ = precision_recall_curve(
                    y_bin[:, class_idx], y_proba[:, class_idx]
                )
                ap = average_precision_score(y_bin[:, class_idx], y_proba[:, class_idx])
            except Exception:
                continue

            ax.plot(
                recall, precision,
                color=VARIANT_COLORS[variant_key],
                lw=2.2, alpha=0.92,
                label=f"{result['exp_name']} (AP={ap:.3f})",
            )

        # Línea de no-skill (proporción de la clase)
        y_true_all = results[0]["y_true_all"]
        baseline_prop = np.mean(y_true_all == class_idx)
        ax.axhline(baseline_prop, color="gray", lw=0.8, ls="--", alpha=0.6,
                   label=f"No-skill ({baseline_prop:.2f})")

        ax.set_title(f"Clase: {RISK_LABELS[class_idx]}", fontweight="bold")
        ax.set_xlabel("Recall")
        if class_idx == 0:
            ax.set_ylabel("Precision")
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.05])
        ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
        ax.grid(alpha=0.18)

    plt.suptitle(
        "Curvas Precision-Recall por clase — Comparación A/B (validación LOSO)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    return output_path


# ─── Complemento: matrices de confusión comparadas ────────────────────────────

def plot_confusion_matrices_comparison(
    results: List[Dict],
    output_path: Path,
) -> Path:
    """
    Panel con las 3 matrices de confusión normalizadas por fila.

    Complemento al gráfico PR principal. Permite identificar qué clase
    confunde cada variante y si la mejora es uniforme o asimétrica.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, result in zip(axes, results):
        cm = confusion_matrix(result["y_true_all"], result["y_pred_all"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=["Bajo", "Medio", "Alto"],
            yticklabels=["Bajo", "Medio", "Alto"],
            ax=ax, cbar=False, linewidths=0.5,
            annot_kws={"fontsize": 9},
        )
        ax.set_title(
            f"{result['exp_name']}\n"
            f"F1={result['f1_macro_mean']:.3f} | PR-AUC={result['pr_auc_mean']:.3f}",
            fontweight="bold", fontsize=10
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    plt.suptitle(
        "Matrices de confusión por variante (acumuladas sobre 9 folds LOSO)",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    return Path(output_path)


# ─── Decisión final: adoptar / descartar ─────────────────────────────────────

def make_decision(results: List[Dict], threshold_improvement: float = 0.01) -> Dict:
    """
    Implementa la decisión final exigida por el asesor (sesión 5, diapositiva 14):
    "¿La variante mejora la métrica central de forma consistente?"

    Criterio operativo:
      - Mejora de F1-Macro ≥ threshold (default 1%) respecto al baseline → adoptar
      - Mejora menor o negativa → descartar
      - Si dos variantes mejoran, se adopta la que tenga mejor F1-Macro
    """
    baseline = results[0]
    variants = results[1:]

    base_f1 = baseline["f1_macro_mean"]
    decisions = []
    for v in variants:
        delta = v["f1_macro_mean"] - base_f1
        adopt = delta >= threshold_improvement
        decisions.append({
            "variant_name": v["exp_name"],
            "delta_f1":     delta,
            "delta_pr_auc": v["pr_auc_mean"] - baseline["pr_auc_mean"],
            "time_overhead_s": v["train_time_s"] - baseline["train_time_s"],
            "decision":     "ADOPTAR" if adopt else "DESCARTAR",
        })

    return {
        "baseline_name": baseline["exp_name"],
        "baseline_f1":   base_f1,
        "decisions":     decisions,
        "best_variant":  max(results, key=lambda r: r["f1_macro_mean"])["exp_name"],
    }
