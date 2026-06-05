"""
Módulo de experimentos — Sprint 3.
Añade hparam_tuning al conjunto de exportaciones públicas.
"""

# Sprint 1 & 2
from posture_risk.experiments.ab_runner import (
    build_gradient_boosting, build_random_forest,
    cross_validate_pipeline, get_feature_importances,
    load_processed_h5, make_shared_splits,
    XGB_AVAILABLE, XGB_BACKEND,
)
from posture_risk.experiments.logger import (
    append_experiment, make_record, next_exp_id, get_git_commit,
)
from posture_risk.experiments.reporting import (
    build_comparison_table, make_decision,
    plot_confusion_matrices_comparison, plot_pr_curves_comparison,
)
from posture_risk.experiments.validation import (
    run_full_checklist, compute_file_hash,
)
from posture_risk.experiments.diagnostics import (
    plot_feature_importance, plot_learning_curves, plot_calibration_curves,
)

# Sprint 3
from posture_risk.experiments.hparam_tuning import (
    load_search_config, run_study, evaluate_with_loso,
    save_best_model, save_best_config, build_topk_table,
)

__all__ = [
    "build_random_forest", "build_gradient_boosting",
    "cross_validate_pipeline", "get_feature_importances",
    "load_processed_h5", "make_shared_splits",
    "XGB_AVAILABLE", "XGB_BACKEND",
    "append_experiment", "make_record", "next_exp_id", "get_git_commit",
    "build_comparison_table", "make_decision",
    "plot_confusion_matrices_comparison", "plot_pr_curves_comparison",
    "run_full_checklist", "compute_file_hash",
    "plot_feature_importance", "plot_learning_curves", "plot_calibration_curves",
    "load_search_config", "run_study", "evaluate_with_loso",
    "save_best_model", "save_best_config", "build_topk_table",
]
