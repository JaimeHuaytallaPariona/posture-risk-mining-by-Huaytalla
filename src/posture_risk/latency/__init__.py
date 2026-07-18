"""
Módulo de latencia y evaluación operativa — Sprint 6.
Sesión 12: Prueba con stakeholder y latencia.
"""

from posture_risk.latency.slo import (
    SLO, SLOs, evaluate_slo, evaluate_all_slos, export_slo_report,
)
from posture_risk.latency.latency_measurement import (
    measure_stage_latency, measure_pipeline_latency, measure_throughput,
    format_latency_table,
)
from posture_risk.latency.pruning import (
    build_pruned_pipeline, evaluate_pruning_variant, run_pruning_experiment,
)

__all__ = [
    # slo
    "SLO", "SLOs", "evaluate_slo", "evaluate_all_slos", "export_slo_report",
    # latency
    "measure_stage_latency", "measure_pipeline_latency", "measure_throughput",
    "format_latency_table",
    # pruning
    "build_pruned_pipeline", "evaluate_pruning_variant", "run_pruning_experiment",
]
