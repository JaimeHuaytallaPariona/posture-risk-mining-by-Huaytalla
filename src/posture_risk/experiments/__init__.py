"""
Módulo de experimentos A/B y HPO — Sprints 1, 2 y 3.
Expone todas las funciones públicas de los submódulos.
"""

from posture_risk.experiments.ab_runner import (
    build_gradient_boosting,
    build_random_forest,
    cross_validate_pipeline,
    get_feature_importances,
    load_processed_h5,
    make_shared_splits,
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
    check_split_strategy,
    check_fit_only_on_train,
    check_seeds_and_protocol,
    check_data_integrity,
    check_logs_completeness,
    compute_file_hash,
    run_full_checklist,
)
from posture_risk.experiments.diagnostics import (
    plot_feature_importance,
    plot_learning_curves,
    plot_calibration_curves,
)

# Sprint 3 — HPO
from posture_risk.experiments.hpo import (
    make_objective,
    create_study,
    run_study,
)
from posture_risk.experiments.hpo_reporting import (
    export_studies_to_csv,
    build_topk_table,
    plot_best_so_far,
    plot_param_importances,
    select_winner_and_backups,
    save_winner_config,
)

__all__ = [
    # ab_runner
    "build_random_forest", "build_gradient_boosting",
    "cross_validate_pipeline", "get_feature_importances",
    "load_processed_h5", "make_shared_splits",
    "XGB_AVAILABLE", "XGB_BACKEND",
    # logger
    "append_experiment", "make_record", "next_exp_id", "get_git_commit",
    # reporting
    "build_comparison_table", "make_decision",
    "plot_confusion_matrices_comparison", "plot_pr_curves_comparison",
    # validation
    "check_split_strategy", "check_fit_only_on_train",
    "check_seeds_and_protocol", "check_data_integrity",
    "check_logs_completeness", "compute_file_hash", "run_full_checklist",
    # diagnostics
    "plot_feature_importance", "plot_learning_curves", "plot_calibration_curves",
    # HPO
    "make_objective", "create_study", "run_study",
    "export_studies_to_csv", "build_topk_table",
    "plot_best_so_far", "plot_param_importances",
    "select_winner_and_backups", "save_winner_config",
]
