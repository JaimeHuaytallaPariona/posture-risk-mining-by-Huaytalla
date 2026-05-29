"""
diagnostics.py  (Sprint 2)
---------------------------
Implementa los tres diagnósticos de la sesión 6:

  1. Feature Importance + estabilidad entre folds (diapositivas 3–6)
  2. Curvas de aprendizaje — diagnóstico bias/varianza (diapositivas 10–12)
  3. Curvas de calibración — probabilidades vs frecuencias reales (diapositiva 12)

Todos los diagnósticos operan sobre el modelo adoptado (XGB_n200_baseFeats)
sin re-entrenamiento adicional: reutilizan los objetos pipeline y splits
ya generados en el notebook 03.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from loguru import logger as log
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss, f1_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

plt.rcParams.update({
    "figure.dpi":      130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size":       11,
})
COLORS = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517", "#E24B4A",
          "#2196F3", "#FF9800", "#9C27B0", "#607D8B", "#F44336"]


# ─── 1. Feature Importance ────────────────────────────────────────────────────

def plot_feature_importance(
    fold_importances: List[Optional[np.ndarray]],
    feature_names: Optional[List[str]],
    output_path: Path,
    top_n: int = 15,
    exp_name: str = "XGB_n200_baseFeats",
) -> Dict:
    """
    Genera el gráfico de importancia de variables con análisis de
    estabilidad entre folds.

    Retorna un dict con el ranking de features y la estabilidad del top-5.

    Estabilidad (sesión 6, diapositiva 3):
      Para cada fold, se identifica el top-5 de features.
      La estabilidad es el porcentaje de folds donde cada feature del
      top-5 global aparece también en el top-5 del fold individual.
    """
    # Filtrar folds sin importancias
    valid = [imp for imp in fold_importances if imp is not None]
    if not valid:
        log.warning("No hay importancias disponibles para este modelo.")
        return {}

    imp_matrix = np.stack(valid, axis=0)   # shape: (n_folds, n_features)
    mean_imp   = imp_matrix.mean(axis=0)
    std_imp    = imp_matrix.std(axis=0)
    n_features = imp_matrix.shape[1]

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(n_features)]

    # Ranking global por importancia media
    top_idx    = np.argsort(mean_imp)[::-1][:top_n]
    top_names  = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
                  for i in top_idx]
    top_mean   = mean_imp[top_idx]
    top_std    = std_imp[top_idx]

    # Estabilidad: ¿el top-5 global aparece en el top-5 de cada fold?
    top5_global = set(top_idx[:5])
    stability   = {}
    for feat_idx in top5_global:
        count = sum(
            1 for fold_imp in valid
            if feat_idx in np.argsort(fold_imp)[::-1][:5]
        )
        fname = (feature_names[feat_idx]
                 if feat_idx < len(feature_names) else f"feat_{feat_idx}")
        stability[fname] = count / len(valid)

    # ── Figura ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Panel izquierdo: importancias con barras de error (std entre folds)
    ax = axes[0]
    colors_bar = [COLORS[0] if i < 5 else "#AAAAAA" for i in range(top_n)]
    bars = ax.barh(
        range(top_n), top_mean[::-1],
        xerr=top_std[::-1],
        color=colors_bar[::-1],
        alpha=0.85, height=0.7,
        error_kw={"elinewidth": 1, "capsize": 3, "ecolor": "#555555"},
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importancia media (Gini gain)")
    ax.set_title(f"Top {top_n} features — {exp_name}\n"
                 "(barras de error = std entre {len(valid)} folds)",
                 fontweight="bold")
    ax.axvline(0, color="black", lw=0.5)

    # Leyenda de colores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS[0], alpha=0.85, label="Top-5 global"),
        Patch(facecolor="#AAAAAA", alpha=0.85, label="Posiciones 6–15"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Panel derecho: estabilidad del top-5 entre folds
    ax2 = axes[1]
    stab_names  = list(stability.keys())
    stab_values = [stability[n] * 100 for n in stab_names]
    bars2 = ax2.bar(range(len(stab_names)), stab_values,
                    color=COLORS[:len(stab_names)], alpha=0.85)
    ax2.set_xticks(range(len(stab_names)))
    ax2.set_xticklabels(stab_names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("% de folds donde aparece en top-5")
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color="#D85A30", ls="--", lw=1.2,
                label="Umbral de estabilidad (80%)")
    ax2.set_title("Estabilidad del Top-5 entre folds LOSO\n"
                  "(alta estabilidad → feature genuinamente discriminativa)",
                  fontweight="bold")
    ax2.legend(fontsize=9)

    for bar, val in zip(bars2, stab_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    plt.suptitle(
        f"Importancia de Variables y Estabilidad entre Folds\n{exp_name}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    log.info(f"Gráfico de importancia guardado: {output_path}")

    return {
        "top_features":    list(zip(top_names, top_mean.tolist(), top_std.tolist())),
        "stability_top5":  stability,
        "n_folds_used":    len(valid),
    }


# ─── 2. Curvas de aprendizaje ────────────────────────────────────────────────

def plot_learning_curves(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    output_path: Path,
    train_fractions: List[float] = None,
    exp_name: str = "XGB_n200_baseFeats",
    seed: int = 42,
) -> Dict:
    """
    Genera curvas de aprendizaje: F1-Macro vs tamaño del set de entrenamiento.

    Implementación manual con GroupKFold para respetar la separación por sujeto.
    Usa el mismo esquema que los experimentos A/B: para cada fracción de datos
    de entrenamiento, se toman aleatoriamente esa proporción de los sujetos de
    train del fold, se entrena y se evalúa en el test fijo.

    Diagnóstico (sesión 6, diapositiva 10):
      - Curvas juntas y bajas (bias alto)     → más features / modelo más expresivo
      - Train alta, valid baja (alta varianza) → más datos / regularización
      - Ambas aún suben                        → recolectar más datos
    """
    if train_fractions is None:
        train_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    rng = np.random.RandomState(seed)

    # Usar un fold fijo como validación (el más grande, que tiene más muestras)
    n_groups = len(np.unique(groups))
    splitter = GroupKFold(n_splits=n_groups)
    all_splits = list(splitter.split(np.zeros_like(y), y, groups=groups))

    # Elegir el fold cuyo test set tiene más muestras (más representativo)
    val_fold_idx = np.argmax([len(s[1]) for s in all_splits])
    train_idx_full, test_idx = all_splits[val_fold_idx]

    X_val, y_val = X[test_idx], y[test_idx]

    train_f1s = []
    val_f1s   = []
    n_samples  = []

    for frac in tqdm(train_fractions, desc="Learning curve"):
        n_take = max(50, int(len(train_idx_full) * frac))
        chosen = rng.choice(train_idx_full, size=n_take, replace=False)
        X_tr, y_tr = X[chosen], y[chosen]

        import copy
        pipe_copy = copy.deepcopy(pipeline)
        pipe_copy.fit(X_tr, y_tr)

        f1_tr  = f1_score(y_tr,  pipe_copy.predict(X_tr),  average="macro", zero_division=0)
        f1_val = f1_score(y_val, pipe_copy.predict(X_val), average="macro", zero_division=0)
        train_f1s.append(f1_tr)
        val_f1s.append(f1_val)
        n_samples.append(n_take)

    # ── Diagnóstico ───────────────────────────────────────────────────────────
    final_gap   = train_f1s[-1] - val_f1s[-1]
    still_rising = val_f1s[-1] > val_f1s[-2] + 0.005 if len(val_f1s) > 1 else False

    if final_gap > 0.15:
        diagnosis = "ALTA VARIANZA: brecha train/val amplia → regularizar o añadir más datos"
    elif train_f1s[-1] < 0.75 and val_f1s[-1] < 0.75:
        diagnosis = "ALTO BIAS: ambas curvas bajas → modelo/features más expresivos"
    elif still_rising:
        diagnosis = "DATOS INSUFICIENTES: curvas aún suben → priorizar recolección de datos propios"
    else:
        diagnosis = "BIEN CALIBRADO: brecha moderada, curvas estabilizadas"

    # ── Figura ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_samples, train_f1s, "o-", color=COLORS[0], lw=2.2,
            label="F1-Macro (train)", alpha=0.9)
    ax.plot(n_samples, val_f1s,   "s--", color=COLORS[2], lw=2.2,
            label=f"F1-Macro (validación — sujeto {val_fold_idx+1})", alpha=0.9)
    ax.fill_between(n_samples, train_f1s, val_f1s, alpha=0.08, color=COLORS[2],
                    label=f"Brecha (gap = {final_gap:.3f})")
    ax.axhline(val_f1s[-1], color="#888888", ls=":", lw=1, alpha=0.6)
    ax.set_xlabel("Muestras de entrenamiento")
    ax.set_ylabel("F1-Macro")
    ax.set_title(f"Curvas de Aprendizaje — {exp_name}\nDiagnóstico: {diagnosis}",
                 fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.18)

    # Anotación del diagnóstico
    ax.text(0.02, 0.05, f"Diagnóstico: {diagnosis}",
            transform=ax.transAxes, fontsize=9,
            color="#D85A30" if "VARIANZA" in diagnosis or "BIAS" in diagnosis else "#1D9E75",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#CCCCCC", alpha=0.9))

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    log.info(f"Curva de aprendizaje guardada: {output_path}")

    return {
        "diagnosis":     diagnosis,
        "final_gap":     final_gap,
        "train_f1_final": train_f1s[-1],
        "val_f1_final":   val_f1s[-1],
        "still_rising":  still_rising,
    }


# ─── 3. Curvas de calibración ─────────────────────────────────────────────────

def plot_calibration_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
    exp_name: str = "XGB_n200_baseFeats",
) -> Dict:
    """
    Genera curvas de calibración (reliability diagrams) por clase (OvR)
    y calcula el Brier Score.

    Verificar que las probabilidades del modelo corresponden a frecuencias reales.
    Un modelo bien calibrado: si dice P=0.8, el 80% de esos casos es positivo.

    Diagnóstico (sesión 6, diapositiva 12):
      - Calibración mala + ranks buenas → calibrar / ajustar umbral
      - Curva por encima de diagonal     → modelo sobreconfiado
      - Curva por debajo de diagonal     → modelo subconfiado
    """
    n_classes    = y_proba.shape[1]
    risk_labels  = ["Bajo riesgo (0)", "Riesgo medio (1)", "Riesgo alto (2)"]
    y_bin        = label_binarize(y_true, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, n_classes + 1, figsize=(18, 5))
    brier_scores = {}

    for cls_idx in range(n_classes):
        ax = axes[cls_idx]
        y_bin_cls = y_bin[:, cls_idx]
        proba_cls = y_proba[:, cls_idx]

        # Curva de calibración
        try:
            frac_pos, mean_pred = calibration_curve(
                y_bin_cls, proba_cls, n_bins=n_bins, strategy="uniform"
            )
            ax.plot(mean_pred, frac_pos, "s-", color=COLORS[cls_idx],
                    lw=2.2, label="Modelo", alpha=0.9)
        except ValueError:
            ax.text(0.5, 0.5, "Clase no representada", ha="center", va="center")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfectamente calibrado")
        ax.set_xlabel("Probabilidad predicha media")
        ax.set_ylabel("Fracción de positivos reales")
        ax.set_title(f"{risk_labels[cls_idx]}", fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.18)

        # Brier Score (más bajo = mejor calibración)
        bs = brier_score_loss(y_bin_cls, proba_cls)
        brier_scores[f"class_{cls_idx}"] = float(bs)
        ax.text(0.05, 0.92, f"Brier = {bs:.4f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC"))

    # Panel derecho: comparación de Brier scores
    ax_bs = axes[n_classes]
    bs_vals   = list(brier_scores.values())
    bs_labels = [risk_labels[i] for i in range(n_classes)]
    ax_bs.bar(range(n_classes), bs_vals, color=COLORS[:n_classes], alpha=0.85)
    ax_bs.set_xticks(range(n_classes))
    ax_bs.set_xticklabels(["Bajo", "Medio", "Alto"])
    ax_bs.set_ylabel("Brier Score (↓ mejor)")
    ax_bs.set_title("Brier Score por clase\n(0 = calibración perfecta)",
                    fontweight="bold")
    for i, val in enumerate(bs_vals):
        ax_bs.text(i, val + 0.002, f"{val:.4f}", ha="center", fontsize=9)

    # Diagnóstico global
    mean_brier = np.mean(bs_vals)
    if mean_brier < 0.10:
        cal_diag = "Calibración buena (Brier < 0.10)"
    elif mean_brier < 0.20:
        cal_diag = "Calibración aceptable (0.10 ≤ Brier < 0.20)"
    else:
        cal_diag = "Calibración deficiente (Brier ≥ 0.20) → considerar Platt scaling"

    plt.suptitle(
        f"Curvas de Calibración (Reliability Diagrams) — {exp_name}\n"
        f"Diagnóstico: {cal_diag} | Brier medio = {mean_brier:.4f}",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.show()
    log.info(f"Curvas de calibración guardadas: {output_path}")

    return {
        "brier_scores":       brier_scores,
        "mean_brier":         float(mean_brier),
        "calibration_diagnosis": cal_diag,
    }
