"""
slo.py  (Sprint 6)
-------------------
Definición formal de los Service Level Objectives (SLOs) del sistema.

Sigue el concepto del asesor (sesión 12, slide 5):
  "SLO = el objetivo medible de calidad del servicio que el sistema IA
   debería cumplir. Serían los ítems individuales de un SLA."

Los SLOs se declaran ANTES de medir, no después de ver los resultados.
Cada SLO tiene: nombre, umbral, unidad, justificación técnica.
"""

from dataclasses import dataclass, field
from typing import Dict, List
import json
from pathlib import Path


@dataclass
class SLO:
    """Un Service Level Objective con umbral, unidad y justificación."""
    name:          str
    threshold:     float
    unit:          str
    comparison:    str   # "less_than", "greater_than", "less_or_equal", "greater_or_equal"
    justification: str
    critical:      bool = False   # SLO crítico bloquea despliegue si falla


# ─── SLOs adoptados para el proyecto ──────────────────────────────────────────

SLOs = {
    "latency_p95_ms": SLO(
        name          = "Latencia p95 por ventana",
        threshold     = 200.0,
        unit          = "ms",
        comparison    = "less_than",
        justification = (
            "Las ventanas de entrada son de 200 ms con 50% de solapamiento; "
            "el sistema debe producir una predicción por ventana en menos "
            "de 200 ms para sostener el streaming sin acumulación de cola. "
            "El p95 (peor caso realista) es más restrictivo que el p50 y "
            "asegura estabilidad temporal del sistema."
        ),
        critical      = True,
    ),
    "latency_p50_ms": SLO(
        name          = "Latencia p50 por ventana",
        threshold     = 100.0,
        unit          = "ms",
        comparison    = "less_than",
        justification = (
            "Mediana de latencia bajo 100 ms garantiza que la mayoría de "
            "predicciones son procesadas con margen amplio, dejando espacio "
            "para picos ocasionales."
        ),
        critical      = False,
    ),
    "throughput_req_per_sec": SLO(
        name          = "Throughput sostenido",
        threshold     = 5.0,
        unit          = "predicciones/segundo",
        comparison    = "greater_or_equal",
        justification = (
            "Con ventanas de 200 ms y 50% de solapamiento, el sistema debe "
            "procesar al menos 10 predicciones/segundo. Se adopta un umbral "
            "conservador de 5 pred/s (equivalente a ventanas de 200 ms sin "
            "solapamiento) como criterio mínimo de viabilidad."
        ),
        critical      = True,
    ),
    "error_rate_pct": SLO(
        name          = "Tasa de errores en inferencia",
        threshold     = 1.0,
        unit          = "%",
        comparison    = "less_than",
        justification = (
            "Errores de inferencia (excepciones, NaN, timeouts) deben ser "
            "menos del 1% de las predicciones. En un sistema ergonómico, "
            "una tasa alta de errores dejaría al operador sin alertas de "
            "riesgo, degradando la seguridad."
        ),
        critical      = True,
    ),
    "f1_macro_min": SLO(
        name          = "F1-Macro mínimo",
        threshold     = 0.80,
        unit          = "",
        comparison    = "greater_or_equal",
        justification = (
            "F1-Macro mínimo del sistema en validación LOSO no debe caer "
            "por debajo del baseline RF+2% ni por debajo del 0.80 absoluto. "
            "Umbral acordado con asesor."
        ),
        critical      = True,
    ),
}


# ─── Evaluación de compliance ────────────────────────────────────────────────

def evaluate_slo(slo: SLO, measured_value: float) -> Dict:
    """Evalúa si un valor medido cumple o no con el SLO."""
    if slo.comparison == "less_than":
        passed = measured_value < slo.threshold
    elif slo.comparison == "less_or_equal":
        passed = measured_value <= slo.threshold
    elif slo.comparison == "greater_than":
        passed = measured_value > slo.threshold
    elif slo.comparison == "greater_or_equal":
        passed = measured_value >= slo.threshold
    else:
        raise ValueError(f"Comparación desconocida: {slo.comparison}")

    return {
        "slo_name":       slo.name,
        "threshold":      slo.threshold,
        "unit":           slo.unit,
        "measured":       measured_value,
        "comparison":     slo.comparison,
        "passed":         bool(passed),
        "critical":       slo.critical,
        "justification":  slo.justification,
    }


def evaluate_all_slos(measurements: Dict[str, float]) -> Dict:
    """
    Evalúa todos los SLOs contra las mediciones tomadas.

    Retorna un diccionario con el reporte completo y el veredicto global.
    """
    results = {}
    for key, slo in SLOs.items():
        if key in measurements:
            results[key] = evaluate_slo(slo, measurements[key])
        else:
            results[key] = {
                "slo_name":  slo.name,
                "threshold": slo.threshold,
                "unit":      slo.unit,
                "measured":  None,
                "passed":    None,
                "critical":  slo.critical,
                "note":      "Medición no disponible",
            }

    # Veredicto: todos los críticos deben pasar
    critical_failing = [
        r for r in results.values()
        if r.get("critical") and r.get("passed") is False
    ]
    all_passing = all(
        r.get("passed") for r in results.values() if r.get("passed") is not None
    )

    return {
        "slos":               results,
        "all_passing":        all_passing,
        "critical_failing":   [r["slo_name"] for r in critical_failing],
        "deployment_ready":   len(critical_failing) == 0,
    }


def export_slo_report(report: Dict, output_path: Path) -> Path:
    """Guarda el reporte de SLOs en JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    return output_path
