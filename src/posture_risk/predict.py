"""
posture_risk.predict  (Sprint 7)
---------------------------------
Módulo CLI de inferencia del sistema de detección postural.

Uso:
    python -m posture_risk.predict --in inputs.json --out predictions.json
    python -m posture_risk.predict --in inputs.jsonl --out predictions.jsonl --format jsonl

Contratos I/O: ver src/posture_risk/contracts/io_schemas.py
Observabilidad: ver src/posture_risk/observability.py

Este módulo NO extrae features desde señales crudas; asume que el vector
de 297 features ya fue calculado por el pipeline de ingestión aguas arriba.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger as log
from pydantic import ValidationError

from posture_risk.contracts.io_schemas import (
    BatchPredictionInput, BatchPredictionOutput,
    ErrorCode, PredictionError, PredictionInput, PredictionOutput,
    RiskLevel,
)


MODEL_VERSION       = "v1.0.0"
MAX_PAYLOAD_WINDOWS = int(os.getenv("MAX_PAYLOAD_WINDOWS", "10000"))
INFERENCE_TIMEOUT_S = float(os.getenv("INFERENCE_TIMEOUT_S", "5.0"))


# ─── Carga de artefactos del modelo ─────────────────────────────────────────

class ModelBundle:
    """Contenedor de todos los artefactos que necesita el pipeline."""

    def __init__(
        self,
        model_path:       Path,
        calibrators_path: Path,
        thresholds_path:  Path,
    ):
        import joblib
        log.info(f"Cargando modelo desde {model_path}")
        self.pipeline    = joblib.load(model_path)
        log.info(f"Cargando calibradores desde {calibrators_path}")
        self.calibrators = joblib.load(calibrators_path)
        log.info(f"Cargando umbrales desde {thresholds_path}")
        with open(thresholds_path) as f:
            self.thresholds = {int(k): float(v) for k, v in json.load(f).items()}
        self.model_version = MODEL_VERSION
        self.artifact_hash = self._compute_hash(model_path)
        log.info(
            f"ModelBundle listo: version={self.model_version} "
            f"hash={self.artifact_hash[:12]}..."
        )

    @staticmethod
    def _compute_hash(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


# ─── Pipeline por etapas ─────────────────────────────────────────────────────

def run_pipeline(
    features_np: np.ndarray,
    bundle:      ModelBundle,
) -> Tuple[int, np.ndarray, float]:
    """
    Ejecuta el pipeline completo sobre una ventana (o batch de ventanas).

    Retorna: (clase_predicha, probabilidades_calibradas, latencia_ms)
    """
    from posture_risk.analysis.mitigations import (
        apply_isotonic_calibration, apply_thresholds,
    )

    t0 = time.perf_counter()

    # 1. Preprocess (asegurar tipo)
    x = features_np.astype(np.float32)

    # 2. Escalado + inferencia (ya empaquetados en pipeline sklearn)
    probs_raw = bundle.pipeline.predict_proba(x)

    # 3. Calibración isotónica post-hoc
    probs_cal = apply_isotonic_calibration(probs_raw, bundle.calibrators)

    # 4. Aplicación de umbrales por clase
    preds = apply_thresholds(probs_cal, bundle.thresholds)

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    return int(preds[0]), probs_cal[0], latency_ms


def class_id_to_label(cls_id: int) -> RiskLevel:
    return {0: RiskLevel.BAJO, 1: RiskLevel.MEDIO, 2: RiskLevel.ALTO}[cls_id]


# ─── Predicción de una ventana ───────────────────────────────────────────────

def predict_single(
    inp:    PredictionInput,
    bundle: ModelBundle,
) -> PredictionOutput:
    """Predice el nivel de riesgo para una única ventana."""
    features_np = np.array(inp.features, dtype=np.float32).reshape(1, -1)
    cls_id, probs, latency_ms = run_pipeline(features_np, bundle)
    return PredictionOutput(
        request_id     = inp.request_id,
        prediction     = class_id_to_label(cls_id),
        probabilities  = [float(p) for p in probs],
        confidence     = float(np.max(probs)),
        latency_ms     = round(latency_ms, 3),
        model_version  = bundle.model_version,
        pipeline_stages = ["preprocess", "scale", "infer_xgb", "calibrate", "threshold"],
        served_at      = datetime.now(timezone.utc),
    )


def predict_batch(
    batch:  BatchPredictionInput,
    bundle: ModelBundle,
) -> BatchPredictionOutput:
    """Predice sobre un batch de ventanas, capturando errores por ventana."""
    predictions = []
    errors      = []
    t0 = time.perf_counter()

    for w in batch.windows:
        try:
            out = predict_single(w, bundle)
            predictions.append(out)
        except Exception as e:
            errors.append(PredictionError(
                request_id = w.request_id,
                code       = ErrorCode.INFERENCE_FAILED,
                message    = str(e),
                hint       = "Verifique el pipeline aguas arriba de esta ventana",
            ))

    total_latency_ms = (time.perf_counter() - t0) * 1000.0

    return BatchPredictionOutput(
        batch_id         = batch.batch_id,
        n_processed      = len(predictions),
        n_errors         = len(errors),
        total_latency_ms = round(total_latency_ms, 3),
        predictions      = predictions,
        errors           = errors,
    )


# ─── I/O de archivos ─────────────────────────────────────────────────────────

def load_input_json(path: Path) -> BatchPredictionInput:
    """Carga un archivo JSON con formato BatchPredictionInput."""
    with open(path) as f:
        raw = json.load(f)
    return BatchPredictionInput(**raw)


def load_input_jsonl(path: Path, batch_id: Optional[str] = None) -> BatchPredictionInput:
    """
    Carga un archivo JSONL donde cada línea es una PredictionInput individual.
    Se agrupa en un BatchPredictionInput con el batch_id dado (o UUID si no se provee).
    """
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            windows.append(PredictionInput(**data))
    return BatchPredictionInput(
        batch_id = batch_id or f"batch-{uuid.uuid4().hex[:12]}",
        windows  = windows,
    )


def write_output_json(output: BatchPredictionOutput, path: Path) -> None:
    with open(path, "w") as f:
        f.write(output.model_dump_json(indent=2))


def write_output_jsonl(output: BatchPredictionOutput, path: Path) -> None:
    with open(path, "w") as f:
        for pred in output.predictions:
            f.write(pred.model_dump_json() + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Módulo CLI de predicción de riesgo postural"
    )
    parser.add_argument("--in", dest="input_path", type=Path, required=True,
                        help="Archivo de entrada (JSON o JSONL)")
    parser.add_argument("--out", dest="output_path", type=Path, required=True,
                        help="Archivo de salida (JSON o JSONL)")
    parser.add_argument("--format", choices=["json", "jsonl"], default="json",
                        help="Formato de I/O (default: json)")
    parser.add_argument("--model", type=Path,
                        default=Path(os.getenv("MODEL_PATH", "models/best_xgb.pkl")),
                        help="Ruta al artefacto del modelo")
    parser.add_argument("--calibrators", type=Path,
                        default=Path(os.getenv("CALIBRATORS_PATH", "models/calibrators.pkl")),
                        help="Ruta a los calibradores isotónicos")
    parser.add_argument("--thresholds", type=Path,
                        default=Path(os.getenv("THRESHOLDS_PATH", "models/thresholds.json")),
                        help="Ruta al JSON de umbrales por clase")
    parser.add_argument("--metrics-csv", type=Path,
                        default=Path(os.getenv("METRICS_CSV", "logs/inference_metrics.csv")),
                        help="CSV donde persistir métricas de invocación")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"),
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Configurar logging
    log.remove()
    log.add(sys.stderr, level=args.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                   "{extra[request_id]} | {message}",
            filter=lambda r: r["extra"].setdefault("request_id", "system") or True)

    # Cargar modelo
    try:
        bundle = ModelBundle(args.model, args.calibrators, args.thresholds)
    except FileNotFoundError as e:
        err = PredictionError(
            code    = ErrorCode.MODEL_NOT_LOADED,
            message = f"Artefacto no encontrado: {e.filename}",
            hint    = "Verifique que MODEL_PATH, CALIBRATORS_PATH y THRESHOLDS_PATH "
                      "apunten a archivos existentes",
        )
        print(err.model_dump_json(indent=2), file=sys.stderr)
        sys.exit(2)

    # Cargar entrada
    try:
        if args.format == "json":
            batch = load_input_json(args.input_path)
        else:
            batch = load_input_jsonl(args.input_path)
    except ValidationError as e:
        err = PredictionError(
            code    = ErrorCode.INVALID_INPUT,
            message = str(e),
            hint    = "Revise el esquema de entrada en contracts/io_schemas.py",
            field   = "input_file",
        )
        print(err.model_dump_json(indent=2), file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        err = PredictionError(
            code    = ErrorCode.INVALID_INPUT,
            message = f"Archivo de entrada no encontrado: {args.input_path}",
            hint    = "Verifique la ruta --in",
        )
        print(err.model_dump_json(indent=2), file=sys.stderr)
        sys.exit(1)

    # Validar límite de tamaño
    if len(batch.windows) > MAX_PAYLOAD_WINDOWS:
        err = PredictionError(
            code    = ErrorCode.PAYLOAD_TOO_LARGE,
            message = f"Batch de {len(batch.windows)} excede MAX_PAYLOAD_WINDOWS={MAX_PAYLOAD_WINDOWS}",
            hint    = "Divida el batch en fragmentos más pequeños",
        )
        print(err.model_dump_json(indent=2), file=sys.stderr)
        sys.exit(1)

    # Predecir
    log.bind(request_id=batch.batch_id).info(
        f"Iniciando predicción de {len(batch.windows)} ventanas"
    )
    output = predict_batch(batch, bundle)

    # Métricas locales (observabilidad — sesión 13 slide 12)
    args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    write_metrics_csv(args.metrics_csv, output, bundle)

    # Escribir salida
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        write_output_json(output, args.output_path)
    else:
        write_output_jsonl(output, args.output_path)

    log.bind(request_id=batch.batch_id).info(
        f"Predicción completa: {output.n_processed} OK, "
        f"{output.n_errors} errores, "
        f"latencia total {output.total_latency_ms:.1f} ms"
    )


def write_metrics_csv(path: Path, output: BatchPredictionOutput,
                       bundle: ModelBundle) -> None:
    """Persiste una fila por invocación en CSV para trazabilidad local."""
    import csv
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow([
                "timestamp", "batch_id", "n_processed", "n_errors",
                "total_latency_ms", "avg_latency_ms",
                "model_version", "artifact_hash",
            ])
        avg_lat = np.mean([p.latency_ms for p in output.predictions]) \
                  if output.predictions else 0.0
        w.writerow([
            datetime.now(timezone.utc).isoformat(),
            output.batch_id, output.n_processed, output.n_errors,
            round(output.total_latency_ms, 3), round(avg_lat, 3),
            bundle.model_version, bundle.artifact_hash[:12],
        ])


if __name__ == "__main__":
    main()
