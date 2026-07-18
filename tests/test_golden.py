"""
tests/test_golden.py
---------------------
Golden test siguiendo el asesor (sesión 13, slide 13):
  "Golden test (opcional): comparas la salida actual contra un 'resultado
   dorado' (conocido) almacenado. Mismo input → misma salida (dentro de
   tolerancia)."

Estrategia:
  - Se toman 10 ventanas fijas del dataset PAMAP2 (indices deterministas).
  - Se ejecuta el pipeline completo (modelo + calibración + umbrales).
  - Se compara la salida contra un archivo dorado guardado en
    tests/golden/golden_predictions.json.
  - Tolerancia: probabilidades ±0.005, predicciones idénticas.

Para regenerar el archivo dorado tras un cambio LEGÍTIMO del modelo:
  python -m pytest tests/test_golden.py --regenerate-golden

Ejecutar con: pytest tests/test_golden.py -v
"""

import json
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
GOLDEN_DIR   = Path(__file__).parent / "golden"
GOLDEN_FILE  = GOLDEN_DIR / "golden_predictions.json"
SAMPLE_FILE  = PROJECT_ROOT / "data" / "examples" / "sample_windows.npz"

TOLERANCE_PROB = 0.005   # tolerancia en probabilidades
GOLDEN_INDICES = [100, 5000, 15000, 30000, 50000, 80000, 120000, 150000, 180000, 194000]


# ─── Fixture: cargar 10 ventanas fijas ──────────────────────────────────────

@pytest.fixture(scope="module")
def golden_windows():
    """Toma 10 ventanas del sample con índices deterministas."""
    if not SAMPLE_FILE.exists():
        pytest.skip(f"Sample no disponible: {SAMPLE_FILE}. Ejecute make setup primero.")

    data = np.load(SAMPLE_FILE)
    X = data["X"]
    n = X.shape[0]
    valid_indices = [i for i in GOLDEN_INDICES if i < n]
    return {
        "indices":  valid_indices,
        "features": [X[i].tolist() for i in valid_indices],
    }


# ─── Fixture: cargar bundle del modelo (una sola vez) ───────────────────────

@pytest.fixture(scope="module")
def model_bundle():
    model_path       = PROJECT_ROOT / "models" / "best_xgb.pkl"
    calibrators_path = PROJECT_ROOT / "models" / "calibrators.pkl"
    thresholds_path  = PROJECT_ROOT / "models" / "thresholds.json"

    missing = [p for p in (model_path, calibrators_path, thresholds_path)
               if not p.exists()]
    if missing:
        pytest.skip(f"Artefactos del modelo no disponibles: {missing}. "
                     "Ejecute make train o make setup primero.")

    from posture_risk.predict import ModelBundle
    return ModelBundle(model_path, calibrators_path, thresholds_path)


# ─── Predicción actual ──────────────────────────────────────────────────────

def _predict_all(golden_windows, model_bundle):
    """Ejecuta la predicción sobre las 10 ventanas doradas."""
    from posture_risk.predict import predict_single
    from posture_risk.contracts.io_schemas import PredictionInput

    results = []
    for idx, features in zip(golden_windows["indices"], golden_windows["features"]):
        inp = PredictionInput(
            request_id  = f"golden-{idx:07d}",
            features    = features,
            subject_hint = None,
        )
        out = predict_single(inp, model_bundle)
        results.append({
            "index":         idx,
            "prediction":    out.prediction.value,
            "probabilities": [round(p, 6) for p in out.probabilities],
            "confidence":    round(out.confidence, 6),
        })
    return results


# ─── Test ───────────────────────────────────────────────────────────────────

def test_golden_predictions_match(golden_windows, model_bundle):
    """
    Compara las predicciones actuales contra las doradas.
    Falla si:
      - Alguna clase predicha difiere
      - Alguna probabilidad difiere en más de TOLERANCE_PROB
    """
    if not GOLDEN_FILE.exists():
        pytest.skip(
            f"Archivo dorado no existe todavía: {GOLDEN_FILE}\n"
            "Genérelo con: python -m tests.generate_golden\n"
            "(script provisto en tests/generate_golden.py)"
        )

    with open(GOLDEN_FILE) as f:
        golden = json.load(f)

    current = _predict_all(golden_windows, model_bundle)

    assert len(current) == len(golden["predictions"]), (
        f"Cantidad de predicciones difiere: actual={len(current)} "
        f"vs golden={len(golden['predictions'])}"
    )

    failures = []
    for cur, gold in zip(current, golden["predictions"]):
        assert cur["index"] == gold["index"], (
            f"Índices desalineados: actual={cur['index']} vs golden={gold['index']}"
        )

        if cur["prediction"] != gold["prediction"]:
            failures.append(
                f"idx={cur['index']}: predicción cambió "
                f"{gold['prediction']} → {cur['prediction']}"
            )
            continue

        for j, (p_cur, p_gold) in enumerate(zip(cur["probabilities"],
                                                 gold["probabilities"])):
            if abs(p_cur - p_gold) > TOLERANCE_PROB:
                failures.append(
                    f"idx={cur['index']}: probabilidad clase {j} cambió "
                    f"{p_gold:.6f} → {p_cur:.6f} "
                    f"(Δ={abs(p_cur - p_gold):.6f} > tol={TOLERANCE_PROB})"
                )

    assert not failures, (
        f"\n{len(failures)} predicciones difieren del dorado:\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\n\nSi el cambio es INTENCIONAL (nuevo modelo, retraining), "
        "regenere el archivo dorado con:\n"
        "  python -m tests.generate_golden"
    )


def test_golden_file_exists_and_is_valid_json():
    """El archivo dorado debe existir y ser JSON válido con la estructura esperada."""
    if not GOLDEN_FILE.exists():
        pytest.skip(f"Archivo dorado no existe: {GOLDEN_FILE}")

    with open(GOLDEN_FILE) as f:
        golden = json.load(f)

    assert "predictions" in golden
    assert "generated_at" in golden
    assert "model_version" in golden
    assert isinstance(golden["predictions"], list)
    assert len(golden["predictions"]) > 0

    for p in golden["predictions"]:
        assert "index"         in p
        assert "prediction"    in p
        assert "probabilities" in p
        assert len(p["probabilities"]) == 3
        assert abs(sum(p["probabilities"]) - 1.0) < 0.02
