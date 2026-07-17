"""
tests/generate_golden.py
-------------------------
Genera el archivo dorado (tests/golden/golden_predictions.json) tomando
10 ventanas fijas de PAMAP2 y ejecutando el pipeline completo.

Se ejecuta UNA SOLA VEZ tras el entrenamiento estable del modelo, o cada
vez que el modelo se actualice INTENCIONALMENTE (nuevo HPO, nueva
calibración, etc.).

Uso:
    python -m tests.generate_golden
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
GOLDEN_DIR   = Path(__file__).parent / "golden"
GOLDEN_FILE  = GOLDEN_DIR / "golden_predictions.json"
SAMPLE_FILE  = PROJECT_ROOT / "data" / "examples" / "sample_windows.npz"
GOLDEN_INDICES = [100, 5000, 15000, 30000, 50000, 80000, 120000, 150000, 180000, 194000]


def main():
    print("=" * 60)
    print("GENERACIÓN DEL ARCHIVO DORADO")
    print("=" * 60)

    # Validar disponibilidad
    if not SAMPLE_FILE.exists():
        raise FileNotFoundError(
            f"No se encuentra {SAMPLE_FILE}. Ejecute 'make setup' primero."
        )

    data = np.load(SAMPLE_FILE)
    X    = data["X"]
    n    = X.shape[0]
    print(f"  Sample cargado: {X.shape}")

    # Cargar bundle
    from posture_risk.predict import ModelBundle, predict_single
    from posture_risk.contracts.io_schemas import PredictionInput

    bundle = ModelBundle(
        model_path       = PROJECT_ROOT / "models" / "best_xgb.pkl",
        calibrators_path = PROJECT_ROOT / "models" / "calibrators.pkl",
        thresholds_path  = PROJECT_ROOT / "models" / "thresholds.json",
    )

    # Predecir
    valid_indices = [i for i in GOLDEN_INDICES if i < n]
    print(f"  Ejecutando predicción sobre {len(valid_indices)} ventanas...")

    predictions = []
    for idx in valid_indices:
        inp = PredictionInput(
            request_id = f"golden-{idx:07d}",
            features   = X[idx].tolist(),
        )
        out = predict_single(inp, bundle)
        predictions.append({
            "index":         idx,
            "prediction":    out.prediction.value,
            "probabilities": [round(p, 6) for p in out.probabilities],
            "confidence":    round(out.confidence, 6),
        })

    # Persistir
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    golden = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "model_version": bundle.model_version,
        "artifact_hash": bundle.artifact_hash,
        "n_windows":     len(predictions),
        "indices":       valid_indices,
        "tolerance":     0.005,
        "predictions":   predictions,
    }

    with open(GOLDEN_FILE, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"\nArchivo dorado guardado: {GOLDEN_FILE}")
    print(f"  Modelo: {bundle.model_version} (hash {bundle.artifact_hash[:12]})")
    print(f"  Predicciones: {len(predictions)}")
    print(f"  Distribución de clases: "
          f"bajo={sum(1 for p in predictions if p['prediction'] == 'bajo')}  "
          f"medio={sum(1 for p in predictions if p['prediction'] == 'medio')}  "
          f"alto={sum(1 for p in predictions if p['prediction'] == 'alto')}")


if __name__ == "__main__":
    main()
