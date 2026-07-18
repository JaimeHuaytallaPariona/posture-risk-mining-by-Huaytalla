"""
slice_reporting.py  (Sprint 5)
-------------------------------
Reportes tabulares y gráficos para el análisis de slicing.

Genera:
  - Tabla consolidada de slices con IC y tamaño
  - Ranking de slices problemáticos por dimensión
  - Gráficos: F1 por slice con barras de error, matrices de confusión,
    comparación baseline vs mitigación
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "figure.dpi":         130,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "font.size":          10,
})

COLOR_BASELINE = "#534AB7"
COLOR_MITIG    = "#1D9E75"
COLOR_ALERT    = "#D85A30"


# ─── Tabla consolidada de slices ──────────────────────────────────────────────

def build_slice_summary(
    slice_results: Dict[str, Dict],
    dimension:     str,
) -> pd.DataFrame:
    """Convierte los resultados de slicing en un DataFrame legible."""
    rows = []
    for name, res in slice_results.items():
        if res.get("empty"):
            continue
        row = {
            "dimensión":    dimension,
            "slice":        str(name),
            "n":            res.get("n", 0),
            "F1-Macro":     res.get("f1_macro", float("nan")),
            "IC_lo":        res.get("f1_ci_lo", float("nan")),
            "IC_hi":        res.get("f1_ci_hi", float("nan")),
            "IC_ancho":     res.get("f1_ci_width", float("nan")),
            "recall_bajo":  res["recall_by_class"].get(0, {}).get("recall", float("nan")),
            "recall_medio": res["recall_by_class"].get(1, {}).get("recall", float("nan")),
            "recall_alto":  res["recall_by_class"].get(2, {}).get("recall", float("nan")),
            "brier":        res.get("brier", float("nan")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("F1-Macro").reset_index(drop=True)
    return df


# ─── Gráfico: F1 por slice con IC ─────────────────────────────────────────────

def plot_slice_f1_with_ci(
    df_slices:   pd.DataFrame,
    global_f1:   float,
    output_path: Path,
    dimension:   str,
    title_suffix: str = "",
) -> Path:
    """
    Gráfico horizontal: F1-Macro por slice con barras de error (IC bootstrap).
    Referencia: línea vertical del F1 global.
    """
    if df_slices.empty:
        return output_path

    df = df_slices.copy()
    df["ci_lower_err"] = df["F1-Macro"] - df["IC_lo"]
    df["ci_upper_err"] = df["IC_hi"] - df["F1-Macro"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df) * 0.4 + 1)))

    y_pos = np.arange(len(df))
    colors = [COLOR_ALERT if f < global_f1 - 0.05 else COLOR_BASELINE
              for f in df["F1-Macro"]]

    ax.barh(
        y_pos, df["F1-Macro"],
        xerr=[df["ci_lower_err"], df["ci_upper_err"]],
        color=colors, alpha=0.85,
        error_kw={"capsize": 3, "elinewidth": 1.2, "ecolor": "#444444"},
    )

    ax.axvline(global_f1, color="black", ls="--", lw=1.5,
               label=f"F1 global = {global_f1:.3f}")
    ax.axvline(global_f1 - 0.05, color=COLOR_ALERT, ls=":", lw=1.0,
               label=f"Umbral crítico (F1 global − 0.05)")

    # Etiquetas: nombre del slice + tamaño
    labels = [f"{n}  (n={sz})" for n, sz in zip(df["slice"], df["n"])]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("F1-Macro (con IC 95% bootstrap)")
    ax.set_xlim(0, 1.05)
    ax.set_title(
        f"F1-Macro por slice — Dimensión: {dimension}{title_suffix}\n"
        f"(barras rojas: F1 al menos 0.05 por debajo del global)",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.15, axis="x")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    return output_path


# ─── Gráfico: matriz de confusión de un slice ────────────────────────────────

def plot_slice_confusion_matrix(
    confusion:   List[List[int]],
    slice_name:  str,
    output_path: Path,
    class_names: Optional[List[str]] = None,
) -> Path:
    """Matriz de confusión para un slice problemático."""
    if class_names is None:
        class_names = ["Bajo", "Medio", "Alto"]

    cm = np.array(confusion, dtype=float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, data, title, fmt in [
        (axes[0], cm.astype(int), "Conteos absolutos", "d"),
        (axes[1], cm_norm,        "Normalizada por fila (recall)", ".2f"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, linewidths=0.4, annot_kws={"fontsize": 10},
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_title(title, fontweight="bold")

    plt.suptitle(
        f"Matriz de confusión — Slice: {slice_name}",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    return output_path


# ─── Gráfico: comparación baseline vs mitigación ─────────────────────────────

def plot_before_after(
    comparison:  Dict,
    output_path: Path,
    mitigation_name: str,
) -> Path:
    """
    Panel comparativo:
      - Barras de F1-Macro global antes vs después
      - Barras de recall por clase antes vs después
      - Barras de Brier antes vs después (si disponible)
    """
    n_panels = 3 if "brier_before" in comparison else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))

    # Panel 1: F1-Macro global
    ax = axes[0]
    labels = ["Baseline", mitigation_name]
    values = [comparison["f1_macro_before"], comparison["f1_macro_after"]]
    bars = ax.bar(labels, values, color=[COLOR_BASELINE, COLOR_MITIG], alpha=0.88)
    ax.set_ylabel("F1-Macro global")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"F1-Macro global\nΔ = {comparison['f1_macro_delta']:+.4f}",
                 fontweight="bold")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 2: Recall por clase
    ax = axes[1]
    classes = ["Bajo", "Medio", "Alto"]
    x = np.arange(len(classes))
    w = 0.35
    before_vals = [comparison["recall_by_class"][c]["before"] for c in range(3)]
    after_vals  = [comparison["recall_by_class"][c]["after"]  for c in range(3)]
    ax.bar(x - w/2, before_vals, w, label="Baseline",     color=COLOR_BASELINE, alpha=0.88)
    ax.bar(x + w/2, after_vals,  w, label=mitigation_name, color=COLOR_MITIG,   alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.0)
    ax.set_title("Recall por clase", fontweight="bold")
    ax.legend(fontsize=9)
    for i, (b, a) in enumerate(zip(before_vals, after_vals)):
        delta = a - b
        color = "green" if delta > 0 else ("red" if delta < 0 else "gray")
        ax.text(i, max(a, b) + 0.03, f"Δ={delta:+.3f}",
                ha="center", fontsize=9, color=color, fontweight="bold")

    # Panel 3: Brier (si disponible)
    if n_panels == 3:
        ax = axes[2]
        vals = [comparison["brier_before"], comparison["brier_after"]]
        bars = ax.bar(labels, vals, color=[COLOR_BASELINE, COLOR_MITIG], alpha=0.88)
        ax.set_ylabel("Brier score (↓ mejor)")
        ax.set_title(f"Brier score\nΔ = {comparison['brier_delta']:+.4f}",
                     fontweight="bold")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle(
        f"Antes vs Después — {mitigation_name}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    return output_path
