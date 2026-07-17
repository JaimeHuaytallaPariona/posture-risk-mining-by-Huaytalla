"""
scripts/train_and_persist.py
-----------------------------
Entrena el modelo actual (XGB HPO winner + calibración isotónica + umbrales
por clase) y persiste los tres artefactos necesarios para el módulo CLI:

    models/best_xgb.pkl       - Pipeline sklearn con scaler + XGBoost
    models/calibrators.pkl    - Calibradores isotónicos por clase (dict)
    models/thresholds.json    - Umbrales óptimos por clase (JSON)

Uso:
    python -m scripts.train_and_persist --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import yaml
from loguru import logger as log

from posture_risk.experiments import load_processed_h5, make_shared_splits
from posture_risk.experiments.ablations import (
    HPO_WINNER_PARAMS, build_xgb_with_scaler,
)
from posture_risk.analysis import (
    fit_isotonic_calibrators, find_optimal_thresholds,
)


SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar datos ─────────────────────────────────────────────────────────
    h5_path = Path(cfg["paths"]["processed"]) / "pamap2_features.h5"
    log.info(f"Cargando datos desde {h5_path}")
    X, y, subject_ids = load_processed_h5(h5_path)

    # ── Cargar predicciones OOF del Sprint 5 (para calibración y umbrales) ──
    oof_path = Path("reports/oof_predictions.npz")
    if not oof_path.exists():
        raise FileNotFoundError(
            f"No se encuentra {oof_path}. Ejecute primero el notebook 07 "
            "del Sprint 5 para generar las predicciones OOF."
        )
    oof = np.load(oof_path)
    y_proba_oof = oof["y_proba"]

    # ── 1. Entrenar pipeline XGB HPO sobre todo el dataset ──────────────────
    log.info("Entrenando pipeline XGB HPO sobre todo el dataset...")
    pipeline = build_xgb_with_scaler(HPO_WINNER_PARAMS, seed=SEED)
    pipeline.fit(X, y)

    model_path = args.output_dir / "best_xgb.pkl"
    joblib.dump(pipeline, model_path)
    log.info(f"Modelo persistido: {model_path}")

    # ── 2. Ajustar calibradores isotónicos sobre OOF ────────────────────────
    log.info("Ajustando calibradores isotónicos sobre OOF...")
    calibrators = fit_isotonic_calibrators(
        y_true=y, y_proba=y_proba_oof, n_classes=3,
    )
    calibrators_path = args.output_dir / "calibrators.pkl"
    joblib.dump(calibrators, calibrators_path)
    log.info(f"Calibradores persistidos: {calibrators_path}")

    # ── 3. Encontrar umbrales óptimos sobre OOF ─────────────────────────────
    log.info("Calculando umbrales óptimos por clase...")
    thresholds = find_optimal_thresholds(
        y_true=y, y_proba=y_proba_oof,
        metric="f1",
        class_priority={0: 1.0, 1: 1.0, 2: 1.5},
    )
    thresholds_path = args.output_dir / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump({str(k): float(v) for k, v in thresholds.items()}, f, indent=2)
    log.info(f"Umbrales persistidos: {thresholds_path}")

    # ── 4. Generar sample de ventanas para E2E y golden test ────────────────
    log.info("Generando sample_windows.npz con las 194,205 ventanas...")
    sample_path = Path("data/examples/sample_windows.npz")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sample_path, X=X.astype(np.float32), y=y)
    log.info(f"Sample persistido: {sample_path}")

    log.info("Entrenamiento completo. Artefactos listos.")


if __name__ == "__main__":
    main()
