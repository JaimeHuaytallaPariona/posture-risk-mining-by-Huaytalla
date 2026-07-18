"""
Módulo de análisis de errores y slicing — Sprint 5.
Sesión 11 del Ing. Glen Rodríguez.
"""

from posture_risk.analysis.slicing import (
    bootstrap_ci, compute_slice_metrics,
    slice_by_subject, slice_by_class, slice_by_activity, slice_by_transition,
    compute_transition_mask, identify_problematic_slices,
    RISK_NAMES, ACTIVITY_NAMES,
)
from posture_risk.analysis.causes import (
    permutation_importance_slice, compare_importances, extract_error_examples,
)
from posture_risk.analysis.mitigations import (
    find_optimal_thresholds, apply_thresholds,
    fit_isotonic_calibrators, apply_isotonic_calibration,
    compare_before_after, validate_hypothesis,
)
from posture_risk.analysis.slice_reporting import (
    build_slice_summary, plot_slice_f1_with_ci,
    plot_slice_confusion_matrix, plot_before_after,
)

__all__ = [
    # slicing
    "bootstrap_ci", "compute_slice_metrics",
    "slice_by_subject", "slice_by_class", "slice_by_activity", "slice_by_transition",
    "compute_transition_mask", "identify_problematic_slices",
    "RISK_NAMES", "ACTIVITY_NAMES",
    # causes
    "permutation_importance_slice", "compare_importances", "extract_error_examples",
    # mitigations
    "find_optimal_thresholds", "apply_thresholds",
    "fit_isotonic_calibrators", "apply_isotonic_calibration",
    "compare_before_after", "validate_hypothesis",
    # reporting
    "build_slice_summary", "plot_slice_f1_with_ci",
    "plot_slice_confusion_matrix", "plot_before_after",
]
