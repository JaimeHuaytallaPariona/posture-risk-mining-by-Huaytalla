"""
Módulo de experimentos A/B.

Implementa el protocolo del Ing. Glen Rodríguez para experimentos
controlados con un solo cambio por variante, mismo split/seed/datos,
y logging completo con trazabilidad.
"""

from posture_risk.experiments.ab_runner import (
    build_gradient_boosting,
    build_random_forest,
    cross_validate_pipeline,
    load_processed_h5,
    make_shared_splits,
    XGB_AVAILABLE, XGB_BACKEND,
)
from posture_risk.experiments.logger import (
    append_experiment, make_record, next_exp_id,
)
from posture_risk.experiments.reporting import (
    build_comparison_table, make_decision,
    plot_confusion_matrices_comparison, plot_pr_curves_comparison,
)

__all__ = [
    "build_random_forest", "build_gradient_boosting",
    "cross_validate_pipeline", "load_processed_h5", "make_shared_splits",
    "XGB_AVAILABLE", "XGB_BACKEND",
    "append_experiment", "make_record", "next_exp_id",
    "build_comparison_table", "make_decision",
    "plot_confusion_matrices_comparison", "plot_pr_curves_comparison",
]
