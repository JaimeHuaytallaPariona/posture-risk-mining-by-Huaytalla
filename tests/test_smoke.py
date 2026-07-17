"""
tests/test_smoke.py
--------------------
Smoke tests siguiendo el asesor (sesión 13, slide 13):
  "Smoke tests: una prueba mínima para verificar que 'el sistema respira'
   1 input válido → predicción válida
   1 input inválido → error claro"

Ejecutar con: pytest tests/test_smoke.py -v
"""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).parent.parent


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_features_valid():
    """Un vector de 297 features en rango razonable."""
    rng = np.random.RandomState(42)
    return rng.uniform(-1.0, 1.0, size=297).tolist()


@pytest.fixture(scope="module")
def sample_features_with_nan():
    """Un vector de 297 features con un NaN en posición 42."""
    rng = np.random.RandomState(42)
    v = rng.uniform(-1.0, 1.0, size=297).tolist()
    v[42] = float("nan")
    return v


@pytest.fixture(scope="module")
def sample_features_wrong_length():
    """Un vector de 296 features (uno menos del esperado)."""
    rng = np.random.RandomState(42)
    return rng.uniform(-1.0, 1.0, size=296).tolist()


# ─── Test 1: validación del esquema de entrada (input válido) ───────────────

def test_valid_input_passes_validation(sample_features_valid):
    """
    Smoke: un input válido pasa la validación de Pydantic sin errores.
    """
    from posture_risk.contracts.io_schemas import PredictionInput

    inp = PredictionInput(
        request_id="a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
        features=sample_features_valid,
        subject_hint=101,
    )
    assert inp.request_id == "a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d"
    assert len(inp.features) == 297
    assert inp.subject_hint == 101


# ─── Test 2: validación del esquema de entrada (input inválido) ─────────────

def test_input_with_nan_is_rejected(sample_features_with_nan):
    """
    Smoke: un input con NaN es rechazado con mensaje claro que apunta al índice.
    """
    from posture_risk.contracts.io_schemas import PredictionInput

    with pytest.raises(ValidationError) as exc_info:
        PredictionInput(
            request_id="b4a9d0e3-2f3a-5b6c-9d0e-1f2a3b4c5d6e",
            features=sample_features_with_nan,
        )
    err_text = str(exc_info.value)
    assert "NaN" in err_text
    assert "42" in err_text  # índice del NaN


def test_input_with_wrong_length_is_rejected(sample_features_wrong_length):
    """
    Smoke: un input con longitud incorrecta es rechazado.
    """
    from posture_risk.contracts.io_schemas import PredictionInput

    with pytest.raises(ValidationError) as exc_info:
        PredictionInput(
            request_id="c5b0e1f4-3a4b-6c7d-0e1f-2a3b4c5d6e7f",
            features=sample_features_wrong_length,
        )
    err_text = str(exc_info.value)
    assert "297" in err_text or "at least" in err_text.lower()


def test_input_with_infinity_is_rejected(sample_features_valid):
    """Smoke: NaN, Inf y valores extremos deben rechazarse."""
    from posture_risk.contracts.io_schemas import PredictionInput

    v = list(sample_features_valid)
    v[10] = float("inf")

    with pytest.raises(ValidationError) as exc_info:
        PredictionInput(
            request_id="d6c1f2a5-4b5c-7d8e-1f2a-3b4c5d6e7f80",
            features=v,
        )
    assert "Inf" in str(exc_info.value)


# ─── Test 3: validación de la salida (formato correcto) ─────────────────────

def test_valid_output_passes_validation():
    """
    Smoke: una salida bien formada pasa validación (probabilidades suman ~1).
    """
    from posture_risk.contracts.io_schemas import PredictionOutput, RiskLevel

    out = PredictionOutput(
        request_id="a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
        prediction=RiskLevel.MEDIO,
        probabilities=[0.15, 0.72, 0.13],
        confidence=0.72,
        latency_ms=12.4,
        model_version="v1.0.0",
        pipeline_stages=["preprocess", "scale", "infer_xgb", "calibrate", "threshold"],
    )
    assert out.prediction == RiskLevel.MEDIO
    assert abs(sum(out.probabilities) - 1.0) < 0.01


def test_output_with_wrong_probability_sum_is_rejected():
    """Smoke: probabilidades que no suman ~1 deben rechazarse."""
    from posture_risk.contracts.io_schemas import PredictionOutput, RiskLevel

    with pytest.raises(ValidationError) as exc_info:
        PredictionOutput(
            request_id="a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
            prediction=RiskLevel.MEDIO,
            probabilities=[0.5, 0.5, 0.5],   # suma = 1.5
            confidence=0.5,
            latency_ms=12.4,
            model_version="v1.0.0",
            pipeline_stages=["preprocess"],
        )
    assert "suman" in str(exc_info.value).lower() or "sum" in str(exc_info.value).lower()


def test_output_with_invalid_model_version_is_rejected():
    """Smoke: versión de modelo que no sigue SemVer es rechazada."""
    from posture_risk.contracts.io_schemas import PredictionOutput, RiskLevel

    with pytest.raises(ValidationError):
        PredictionOutput(
            request_id="a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
            prediction=RiskLevel.BAJO,
            probabilities=[0.8, 0.15, 0.05],
            confidence=0.8,
            latency_ms=12.4,
            model_version="beta-experimental",   # no cumple pattern
            pipeline_stages=["preprocess"],
        )


# ─── Test 4: batch input rechaza IDs duplicados ─────────────────────────────

def test_batch_with_duplicate_request_ids_rejected(sample_features_valid):
    """Los request_id dentro de un batch deben ser únicos."""
    from posture_risk.contracts.io_schemas import (
        BatchPredictionInput, PredictionInput,
    )

    w = PredictionInput(
        request_id="a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
        features=sample_features_valid,
    )
    with pytest.raises(ValidationError) as exc_info:
        BatchPredictionInput(
            batch_id="test-batch",
            windows=[w, w],   # duplicado
        )
    assert "único" in str(exc_info.value).lower() or "unique" in str(exc_info.value).lower()
