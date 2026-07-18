"""
latency_measurement.py  (Sprint 6)
-----------------------------------
Perfilado de latencia por etapa siguiendo el protocolo del asesor
(sesión 12, slide 6).

Mide la latencia de cada etapa del pipeline por separado:
  1. Preprocesamiento (extracción de features desde señal cruda)
  2. Escalado (StandardScaler)
  3. Inferencia del modelo
  4. Postproceso (calibración isotónica)
  5. Decisión final (aplicación de umbrales)

Reporta p50, p95, p99, media y std por etapa y total.
Además calcula throughput (predicciones por segundo).
"""

import gc
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger as log
from tqdm import tqdm


def measure_stage_latency(
    stage_fn:       Callable,
    stage_input:    np.ndarray,
    n_repeats:      int = 500,
    n_warmup:       int = 50,
    stage_name:     str = "stage",
) -> Dict:
    """
    Mide la latencia de una etapa de forma repetible.

    Protocolo:
      - Ejecutar n_warmup veces sin medir (para calentar caches, JIT, etc.)
      - Ejecutar n_repeats veces midiendo con time.perf_counter()
      - Reportar p50, p95, p99, media, std
    """
    # Warmup
    for _ in range(n_warmup):
        try:
            _ = stage_fn(stage_input)
        except Exception:
            pass

    # Medición
    times_ms = []
    errors   = 0
    for _ in range(n_repeats):
        try:
            t0 = time.perf_counter()
            _  = stage_fn(stage_input)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        except Exception as e:
            errors += 1

    if not times_ms:
        return {
            "stage":       stage_name,
            "n_repeats":   n_repeats,
            "n_errors":    errors,
            "error_rate":  100.0,
            "note":        "Todas las mediciones fallaron",
        }

    arr = np.array(times_ms)
    return {
        "stage":         stage_name,
        "n_repeats":     n_repeats,
        "n_errors":      errors,
        "error_rate_pct": 100.0 * errors / n_repeats,
        "p50_ms":        float(np.percentile(arr, 50)),
        "p95_ms":        float(np.percentile(arr, 95)),
        "p99_ms":        float(np.percentile(arr, 99)),
        "mean_ms":       float(np.mean(arr)),
        "std_ms":        float(np.std(arr)),
        "min_ms":        float(np.min(arr)),
        "max_ms":        float(np.max(arr)),
    }


def measure_pipeline_latency(
    stages:      Dict[str, Callable],
    input_sample: np.ndarray,
    n_repeats:   int = 500,
    n_warmup:    int = 50,
) -> Dict:
    """
    Mide la latencia de cada etapa Y la latencia total del pipeline.

    stages: dict {nombre_etapa: función} — las etapas se ejecutan en orden.
            La salida de cada etapa se pasa como input a la siguiente.
    """
    stage_results = {}
    current_input = input_sample

    for stage_name, stage_fn in stages.items():
        log.info(f"Midiendo etapa: {stage_name}")
        result = measure_stage_latency(
            stage_fn, current_input,
            n_repeats=n_repeats, n_warmup=n_warmup,
            stage_name=stage_name,
        )
        stage_results[stage_name] = result

        # Actualizar input con la salida de esta etapa
        try:
            current_input = stage_fn(current_input)
        except Exception:
            log.warning(f"Etapa {stage_name} produjo error al encadenar")

    # Medir el pipeline completo end-to-end
    def full_pipeline(x):
        for _, fn in stages.items():
            x = fn(x)
        return x

    log.info("Midiendo pipeline completo end-to-end")
    total_result = measure_stage_latency(
        full_pipeline, input_sample,
        n_repeats=n_repeats, n_warmup=n_warmup,
        stage_name="total",
    )

    return {
        "stages":     stage_results,
        "total":      total_result,
        "n_repeats":  n_repeats,
        "n_warmup":   n_warmup,
    }


def measure_throughput(
    pipeline_fn:  Callable,
    input_samples: List[np.ndarray],
    duration_s:   float = 10.0,
    n_warmup:     int   = 100,
) -> Dict:
    """
    Mide el throughput sostenido del pipeline durante `duration_s` segundos.

    Método: ejecuta el pipeline en un bucle sin pausas durante T segundos y
    cuenta cuántas predicciones se completaron. Reporta predicciones por segundo.
    """
    # Warmup
    for i in range(n_warmup):
        try:
            _ = pipeline_fn(input_samples[i % len(input_samples)])
        except Exception:
            pass

    # Medición sostenida
    n_completed = 0
    n_errors    = 0
    t_start = time.perf_counter()
    t_end   = t_start + duration_s

    idx = 0
    while time.perf_counter() < t_end:
        try:
            _ = pipeline_fn(input_samples[idx % len(input_samples)])
            n_completed += 1
        except Exception:
            n_errors += 1
        idx += 1

    actual_duration = time.perf_counter() - t_start
    throughput      = n_completed / actual_duration

    return {
        "duration_s":     actual_duration,
        "n_completed":    n_completed,
        "n_errors":       n_errors,
        "throughput_rps": throughput,
        "error_rate_pct": 100.0 * n_errors / (n_completed + n_errors) if (n_completed + n_errors) > 0 else 0.0,
    }


def format_latency_table(results: Dict, title: str = "Latencia") -> str:
    """Genera una tabla legible de resultados de latencia."""
    lines = []
    lines.append(f"\n{title}")
    lines.append("=" * 75)
    header = f"{'Etapa':<25} {'p50':>10} {'p95':>10} {'p99':>10} {'mean':>10} {'std':>8}"
    lines.append(header)
    lines.append("-" * 75)

    stages = results.get("stages", {})
    for stage_name, res in stages.items():
        if "p50_ms" not in res:
            continue
        line = (
            f"{stage_name:<25} "
            f"{res['p50_ms']:>7.3f} ms "
            f"{res['p95_ms']:>7.3f} ms "
            f"{res['p99_ms']:>7.3f} ms "
            f"{res['mean_ms']:>7.3f} ms "
            f"{res['std_ms']:>5.3f} ms"
        )
        lines.append(line)

    lines.append("-" * 75)
    total = results.get("total", {})
    if "p50_ms" in total:
        line = (
            f"{'TOTAL (end-to-end)':<25} "
            f"{total['p50_ms']:>7.3f} ms "
            f"{total['p95_ms']:>7.3f} ms "
            f"{total['p99_ms']:>7.3f} ms "
            f"{total['mean_ms']:>7.3f} ms "
            f"{total['std_ms']:>5.3f} ms"
        )
        lines.append(line)
    lines.append("=" * 75)
    return "\n".join(lines)
