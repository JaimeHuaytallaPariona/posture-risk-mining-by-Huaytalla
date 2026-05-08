"""
compare.py
----------
Genera la tabla comparativa final + curva PR (gráfico clave) a partir
de los resultados de todas las variantes ejecutadas.

Uso:
    python -m posture_risk.experiments.compare
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

plt.rcParams.update({
    "figure.dpi":           120,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "font.size":            10,
})

COLORS_VARIANT = {
    "baseline":              "#1D9E75",
    "var_a_only_temporal":   "#534AB7",
    "var_b_svm_rbf":         "#D85A30",
    "var_c_window_400ms":    "#BA7517",
}
RISK_LABELS = ["Bajo", "Medio", "Alto"]


# =============================================================================
# Carga consolidada
# =============================================================================

def load_all_results(experiments_dir: Path) -> Dict[str, dict]:
    """Carga todos los resúmenes JSON y predicciones .npz disponibles."""
    results = {}
    for json_file in sorted(experiments_dir.glob("*_summary.json")):
        variant = json_file.stem.replace("_summary", "")
        with open(json_file) as f:
            results[variant] = json.load(f)

        npz_path = experiments_dir / f"{variant}_predictions.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            results[variant]["y_true"]  = data["y_true"]
            results[variant]["y_pred"]  = data["y_pred"]
            results[variant]["y_proba"] = data["y_proba"]

    logger.info(f"Variantes cargadas: {list(results.keys())}")
    return results


# =============================================================================
# Tabla comparativa estándar
# =============================================================================

def build_comparison_table(results: Dict[str, dict]) -> pd.DataFrame:
    """
    Tabla estándar: Modelo / Métrica principal / Métricas secundarias / Costo.
    Métrica principal: F1-Macro (insensible al desbalance de clases).
    """
    rows = []
    for variant, data in results.items():
        s = data["summary"]
        rows.append({
            "Modelo":      data["name"],
            "F1-Macro ★":  f"{s['f1_macro_mean']:.4f} ± {s['f1_macro_std']:.4f}",
            "Accuracy":    f"{s['accuracy_mean']:.4f} ± {s['accuracy_std']:.4f}",
            "AUC-OvR":     f"{s['auc_ovr_mean']:.4f} ± {s['auc_ovr_std']:.4f}",
            "AP-Macro":    f"{s['avg_precision_mean']:.4f} ± {s['avg_precision_std']:.4f}",
            "Tiempo (s)":  f"{s['total_time_s']:.1f}",
            "Variante":    variant,
        })
    df = pd.DataFrame(rows)
    return df


def print_markdown_table(df: pd.DataFrame, output_path: Path = None):
    """Imprime y guarda la tabla en formato Markdown listo para tesis."""
    md_table = df.drop(columns=["Variante"]).to_markdown(index=False)
    print("\n" + md_table + "\n")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Tabla comparativa de experimentos A/B\n\n")
            f.write("**Métrica principal (★):** F1-Macro — insensible al desbalance.\n")
            f.write("**Métricas secundarias:** Accuracy, AUC-OvR, AP-Macro.\n")
            f.write("**Validación:** LOSO (Leave-One-Subject-Out, n=9 folds).\n\n")
            f.write(md_table)
            f.write("\n")
        logger.info(f"Tabla guardada en {output_path}")


# =============================================================================
# Gráfico clave: Curva Precision-Recall
# =============================================================================

def plot_pr_curves(results: Dict[str, dict], output_path: Path):
    """
    Genera la figura central comparativa con:
      A) Curvas PR macro (una por variante, todas las clases)
      B) Curvas PR por clase del baseline (detalle por nivel de riesgo)
      C) Comparación de F1-Macro por variante (barras con error)
      D) Curvas PR de la clase 'Riesgo Alto' para todas las variantes
    """
    fig = plt.figure(figsize=(16, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)

    # ── A) Curvas PR macro por variante ───────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for variant, data in results.items():
        if "y_true" not in data:
            continue
        y_true  = data["y_true"]
        y_proba = data["y_proba"]
        n_classes = y_proba.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        precs, recs = [], []
        for k in range(n_classes):
            p, r, _ = precision_recall_curve(y_bin[:, k], y_proba[:, k])
            precs.append(p)
            recs.append(r)
        # Usar AP-Macro como anotación
        ap = data["summary"]["avg_precision_mean"]
        # Curva representativa: clase de riesgo alto (la más crítica)
        p_high, r_high, _ = precision_recall_curve(y_bin[:, 2], y_proba[:, 2])
        ax_a.plot(r_high, p_high,
                  color=COLORS_VARIANT.get(variant, "gray"),
                  lw=2, label=f"{data['name']}\n(AP={ap:.3f})")

    ax_a.set_xlabel("Recall")
    ax_a.set_ylabel("Precision")
    ax_a.set_title("A) Curva PR — Clase 'Riesgo Alto' (la más crítica)",
                   fontweight="bold")
    ax_a.set_xlim(0, 1.02)
    ax_a.set_ylim(0, 1.02)
    ax_a.legend(loc="lower left", fontsize=8)
    ax_a.grid(True, alpha=0.3)

    # ── B) Curvas PR por clase — solo baseline ────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if "baseline" in results and "y_true" in results["baseline"]:
        y_true  = results["baseline"]["y_true"]
        y_proba = results["baseline"]["y_proba"]
        n_classes = y_proba.shape[1]
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        class_colors = ["#1D9E75", "#BA7517", "#D85A30"]
        for k in range(n_classes):
            p, r, _ = precision_recall_curve(y_bin[:, k], y_proba[:, k])
            ap = average_precision_score(y_bin[:, k], y_proba[:, k])
            ax_b.plot(r, p, color=class_colors[k], lw=2,
                      label=f"{RISK_LABELS[k]} (AP={ap:.3f})")
    ax_b.set_xlabel("Recall")
    ax_b.set_ylabel("Precision")
    ax_b.set_title("B) Curva PR por clase — Baseline",
                   fontweight="bold")
    ax_b.set_xlim(0, 1.02)
    ax_b.set_ylim(0, 1.02)
    ax_b.legend(loc="lower left", fontsize=9)
    ax_b.grid(True, alpha=0.3)

    # ── C) Barras F1-Macro con error ──────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    variants_list = list(results.keys())
    f1_means = [results[v]["summary"]["f1_macro_mean"] for v in variants_list]
    f1_stds  = [results[v]["summary"]["f1_macro_std"]  for v in variants_list]
    colors   = [COLORS_VARIANT.get(v, "gray") for v in variants_list]
    bars = ax_c.bar(range(len(variants_list)), f1_means, yerr=f1_stds,
                     color=colors, alpha=0.85, edgecolor="white",
                     capsize=6, error_kw={"ecolor": "#444", "lw": 1.2})
    ax_c.set_xticks(range(len(variants_list)))
    ax_c.set_xticklabels([results[v]["name"].replace(" — ", "\n") for v in variants_list],
                          fontsize=8, rotation=0, ha="center")
    ax_c.set_ylabel("F1-Macro (LOSO)")
    ax_c.set_title("C) F1-Macro por variante (media ± std sobre 9 folds)",
                   fontweight="bold")
    ax_c.set_ylim(0, 1.05)
    ax_c.axhline(f1_means[0], color="#1D9E75", ls="--", lw=1, alpha=0.5,
                 label=f"Baseline = {f1_means[0]:.3f}")
    ax_c.legend(loc="lower right", fontsize=9)
    for bar, mean in zip(bars, f1_means):
        ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                  f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    # ── D) Costo / Tiempo de entrenamiento por variante ───────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    times = [results[v]["summary"]["total_time_s"] for v in variants_list]
    bars = ax_d.barh(range(len(variants_list)), times, color=colors, alpha=0.85)
    ax_d.set_yticks(range(len(variants_list)))
    ax_d.set_yticklabels([results[v]["name"] for v in variants_list], fontsize=9)
    ax_d.set_xlabel("Tiempo total LOSO (segundos)")
    ax_d.set_title("D) Costo computacional por variante",
                   fontweight="bold")
    for bar, t in zip(bars, times):
        ax_d.text(bar.get_width() + max(times)*0.01, bar.get_y() + bar.get_height()/2,
                  f"{t:.1f}s", va="center", fontsize=9)

    plt.suptitle(
        "Comparación A/B de variantes — Validación LOSO sobre PAMAP2",
        fontsize=13, fontweight="bold", y=0.995
    )
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    logger.info(f"Figura guardada en {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    experiments_dir = Path(cfg["paths"]["figures"]).parent / "experiments"
    figures_dir     = Path(cfg["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(experiments_dir)
    if not results:
        raise RuntimeError("No se encontraron resultados. Ejecuta primero runner.py")

    df = build_comparison_table(results)
    print_markdown_table(df, output_path=experiments_dir / "comparison_table.md")
    df.to_csv(experiments_dir / "comparison_table.csv", index=False)

    plot_pr_curves(results, output_path=figures_dir / "experiments_comparison.png")


if __name__ == "__main__":
    main()
