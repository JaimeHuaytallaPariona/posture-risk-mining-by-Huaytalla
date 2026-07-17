"""
contracts/io_schemas.py
------------------------
Contratos formales de entrada/salida/error para el módulo de predicción
posture_risk.predict (CLI).

Sigue el protocolo del asesor (sesión 13, slide 9):
  "Entrada: esquema con tipos, rangos, obligatorios, defaults, ejemplo válido.
   Salida: predicción + campos de diagnóstico (probabilidades, latencia ms,
           versión del artefacto).
   Errores: formato claro (código, mensaje, hint)."

La validación con Pydantic protege el pipeline de inputs mal formados
antes de que lleguen al modelo. Un input inválido produce un ValidationError
con explicación clara del campo que falló.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Constantes del sistema ──────────────────────────────────────────────────

N_FEATURES         = 297                   # 27 canales × 11 features
N_CLASSES          = 3                     # bajo, medio, alto
FEATURE_MIN        = -1e6                  # rango permitido tras filtrado
FEATURE_MAX        = 1e6
MODEL_VERSION_RE   = r"^v\d+\.\d+\.\d+.*$" # ej. "v1.0.0", "v1.0.0-rc1"


class RiskLevel(str, Enum):
    """Niveles de riesgo postural (idénticos al mapeo del pipeline base)."""
    BAJO  = "bajo"
    MEDIO = "medio"
    ALTO  = "alto"


# ─── Entrada ─────────────────────────────────────────────────────────────────

class PredictionInput(BaseModel):
    """
    Entrada del módulo de predicción: una ventana de 200 ms con 297 features
    estadísticas ya extraídas (6 temporales + 5 espectrales × 27 canales IMU).

    El módulo NO extrae features desde señales crudas; asume que el vector
    de features ya fue calculado por el pipeline de ingestión aguas arriba.
    """
    request_id:    str = Field(
        ...,
        description="Identificador único de la solicitud (UUID recomendado)",
        min_length=1, max_length=128,
        examples=["a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d"],
    )
    features:      List[float] = Field(
        ...,
        description=f"Vector de {N_FEATURES} features estadísticas",
        min_length=N_FEATURES, max_length=N_FEATURES,
    )
    subject_hint:  Optional[int] = Field(
        default=None,
        description="ID del sujeto para trazabilidad (opcional, no afecta predicción)",
        ge=0, le=99999,
    )
    timestamp:     Optional[datetime] = Field(
        default=None,
        description="Momento de captura de la ventana (opcional)",
    )

    @field_validator("features")
    @classmethod
    def validate_feature_range(cls, v: List[float]) -> List[float]:
        """Rechaza vectores con NaN, Inf o valores fuera de rango razonable."""
        import math
        for i, x in enumerate(v):
            if math.isnan(x):
                raise ValueError(f"features[{i}] es NaN (no admitido)")
            if math.isinf(x):
                raise ValueError(f"features[{i}] es Inf (no admitido)")
            if not (FEATURE_MIN <= x <= FEATURE_MAX):
                raise ValueError(
                    f"features[{i}] fuera de rango: {x} ∉ [{FEATURE_MIN}, {FEATURE_MAX}]"
                )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
                "features":   [0.1] * N_FEATURES,
                "subject_hint": 101,
                "timestamp":  "2026-07-15T10:30:00Z",
            }
        }
    }


class BatchPredictionInput(BaseModel):
    """Entrada por lotes: múltiples ventanas en una sola invocación."""
    batch_id:  str = Field(..., min_length=1, max_length=128)
    windows:   List[PredictionInput] = Field(..., min_length=1, max_length=10000)

    @model_validator(mode="after")
    def check_unique_request_ids(self):
        ids = [w.request_id for w in self.windows]
        if len(set(ids)) != len(ids):
            raise ValueError("Los request_id dentro del batch deben ser únicos")
        return self


# ─── Salida ──────────────────────────────────────────────────────────────────

class PredictionOutput(BaseModel):
    """
    Salida del módulo: predicción + campos de diagnóstico requeridos
    por el asesor (probabilidades, latencia, versión).
    """
    request_id:     str
    prediction:     RiskLevel = Field(
        ..., description="Nivel de riesgo predicho por el modelo"
    )
    probabilities:  List[float] = Field(
        ...,
        description="Probabilidades calibradas por clase [bajo, medio, alto]",
        min_length=N_CLASSES, max_length=N_CLASSES,
    )
    confidence:     float = Field(
        ..., ge=0.0, le=1.0,
        description="Probabilidad de la clase predicha (máximo del vector)",
    )
    latency_ms:     float = Field(
        ..., ge=0.0,
        description="Latencia total de inferencia en milisegundos",
    )
    model_version:  str = Field(
        ..., pattern=MODEL_VERSION_RE,
        description="Versión del artefacto del modelo (SemVer)",
    )
    pipeline_stages: List[str] = Field(
        ...,
        description="Etapas del pipeline ejecutadas en orden",
    )
    served_at:      datetime = Field(
        default_factory=datetime.utcnow,
        description="Momento en que se produjo la respuesta (UTC)",
    )

    @field_validator("probabilities")
    @classmethod
    def check_probability_sum(cls, v: List[float]) -> List[float]:
        s = sum(v)
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"Probabilidades no suman 1.0 (suma={s:.4f})")
        for i, p in enumerate(v):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"probabilities[{i}]={p} fuera de [0, 1]")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id":    "a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
                "prediction":    "medio",
                "probabilities": [0.15, 0.72, 0.13],
                "confidence":    0.72,
                "latency_ms":    12.4,
                "model_version": "v1.0.0",
                "pipeline_stages": [
                    "preprocess", "scale", "infer_xgb", "calibrate", "threshold",
                ],
                "served_at": "2026-07-15T10:30:00.145Z",
            }
        }
    }


class BatchPredictionOutput(BaseModel):
    """Salida por lotes."""
    batch_id:      str
    n_processed:   int = Field(..., ge=0)
    n_errors:      int = Field(..., ge=0)
    total_latency_ms: float = Field(..., ge=0.0)
    predictions:   List[PredictionOutput]
    errors:        List["PredictionError"] = Field(default_factory=list)


# ─── Errores ─────────────────────────────────────────────────────────────────

class ErrorCode(str, Enum):
    """Códigos de error del módulo (estables entre versiones)."""
    INVALID_INPUT       = "E001_INVALID_INPUT"
    MODEL_NOT_LOADED    = "E002_MODEL_NOT_LOADED"
    INFERENCE_FAILED    = "E003_INFERENCE_FAILED"
    TIMEOUT             = "E004_TIMEOUT"
    PAYLOAD_TOO_LARGE   = "E005_PAYLOAD_TOO_LARGE"
    INTERNAL_ERROR      = "E999_INTERNAL_ERROR"


class PredictionError(BaseModel):
    """Formato de error del asesor: código + mensaje + hint."""
    request_id:  Optional[str] = Field(None, description="ID afectado si aplica")
    code:        ErrorCode
    message:     str = Field(..., description="Descripción del error")
    hint:        str = Field(..., description="Sugerencia de resolución")
    field:       Optional[str] = Field(None, description="Campo específico si aplica")
    timestamp:   datetime = Field(default_factory=datetime.utcnow)

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "a3f8c9d2-1e2f-4a5b-8c9d-0e1f2a3b4c5d",
                "code":       "E001_INVALID_INPUT",
                "message":    "features[42] es NaN (no admitido)",
                "hint":       "Verifique el pipeline de extracción; NaN típicamente "
                              "aparece por interpolación fallida en señal cruda",
                "field":      "features",
                "timestamp":  "2026-07-15T10:30:00Z",
            }
        }
    }


# Resolver forward references de BatchPredictionOutput
BatchPredictionOutput.model_rebuild()
