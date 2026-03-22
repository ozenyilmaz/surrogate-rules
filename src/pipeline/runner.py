"""
src/pipeline/runner.py
-----------------------
Orchestrates the full surrogate-rules pipeline for one (dataset, seed) pair.

Responsibility  : Coordinate data → forest → MIP → KPI → result.
Architecture    : src/pipeline/
Migration from  : generateproblem.py (computeAll) +
                  run_pipeline_resume.py (run_one_seed, summarize_cv)

Key improvements:
  - No print() calls; uses structured logging.
  - Returns typed dataclasses, not ad-hoc tuples.
  - CV and final-test logic unified under one runner function.
  - All steps use config-driven parameters.
  - Seed offset strategy is explicit and documented.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config.models import PipelineConfig
from src.data.loaders import load_dataset
from src.kpi.metrics import (
    FidelityMetrics,
    compute_expressive_power,
    compute_fidelity_metrics,
    get_coverage_and_predictions,
    hamming_loss_rule_model,
)
from src.optimization.solver import SolverResult, build_coverage_matrix, solve_rule_selection
from src.rules.forest import (
    ForestPaths,
    RulePath,
    check_path_representativeness,
    check_tree_representativeness,
    compute_forest_paths,
    compute_rule_stats,
    compute_surrogate_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────

@dataclass
class FoldResult:
    """Metrics for a single CV fold or the final holdout test."""
    surrogate_acc: float
    rf_acc: float
    precision: float
    recall: float
    f1: float
    mcc: float
    tp: int
    fp: int
    tn: int
    fn: int
    hamming: float
    disagreement: float
    overlap: float
    rule_count: int
    rcs: float
    fcs: float
    fie: float
    dice_mean: float
    dice_var: float
    rep_trees: float
    rep_paths: float

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SeedResult:
    """Aggregated result for a single seed: CV summary + final test."""
    seed: int
    cv_summary: Dict[str, float]   # mean + std of each FoldResult field
    final_test: FoldResult
    alpha: float
    depth: int
    lambda_: float


# ─────────────────────────────────────────────
# Single fold runner
# ─────────────────────────────────────────────

def _run_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    fold_seed: int,
) -> Optional[FoldResult]:
    """
    Execute the full pipeline for one train/test split.

    Returns None if the MIP produces no solution.
    """
    # 1. Forest + path extraction
    forest: ForestPaths = compute_forest_paths(
        X_train, y_train,
        n_estimators=cfg.forest.n_estimators,
        max_depth=cfg.forest.max_depth,
        seed=fold_seed,
    )

    # 2. Rule statistics
    stats = compute_rule_stats(
        X_train, y_train,
        paths=forest.paths,
        min_samples_fraction=cfg.solver.min_samples_per_rule,
    )
    if stats is None:
        logger.warning("No valid rules after pruning. Skipping fold.")
        return None

    L = len(stats.paths)
    n = X_train.shape[0]
    A = build_coverage_matrix(n, stats.samples)

    # 3. Solve MIP
    result: SolverResult = solve_rule_selection(
        n=n,
        A=A,
        loss=stats.loss,
        freq=stats.fusion,
        lambda_=cfg.solver.lambda_,
        alpha=cfg.solver.alpha,
        time_limit=cfg.solver.time_limit_seconds,
        output_flag=int(cfg.solver.is_verbose),
        backend=cfg.solver.backend,
    )

    if not result.solved or not result.selected_indices:
        logger.warning("MIP not solved or no rules selected.")
        return None

    sel = result.selected_indices
    sel_paths = [stats.paths[i] for i in sel]
    sel_labels = stats.labels[sel]
    sel_weights = stats.weights[sel]

    # 4. Surrogate accuracy
    surrogate_acc = compute_surrogate_score(X_test, y_test, sel_paths, sel_labels, sel_weights)
    rf_acc = accuracy_score(y_test, forest.clf.predict(X_test))

    # 5. Coverage + predictions
    coverage, y_pred = get_coverage_and_predictions(X_test, sel_paths, sel_labels)

    # 6. Classification metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[-1, 1]).ravel()
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    mcc = float(matthews_corrcoef(y_test, y_pred))

    # 7. Fidelity
    fidelity: FidelityMetrics = compute_fidelity_metrics(X_test, forest.clf, sel_paths, sel_labels)

    # 8. Expressiveness
    expr = compute_expressive_power(sel_paths)

    # 9. Representativeness
    rep_trees = check_tree_representativeness(sel_paths, forest.trees_noded)
    rep_paths = check_path_representativeness(sel_paths, forest.trees_pathed)

    return FoldResult(
        surrogate_acc=surrogate_acc,
        rf_acc=rf_acc,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
        hamming=hamming_loss_rule_model(y_test, coverage),
        disagreement=fidelity.disagreement_rate,
        overlap=fidelity.feature_overlap_f1,
        rule_count=len(sel_paths),
        rcs=expr.rule_condition_size,
        fcs=expr.feature_condition_size,
        fie=expr.feature_interaction_extent,
        dice_mean=expr.dice_mean,
        dice_var=expr.dice_var,
        rep_trees=rep_trees,
        rep_paths=rep_paths,
    )


# ─────────────────────────────────────────────
# CV summary
# ─────────────────────────────────────────────

def _summarize_cv(fold_results: List[FoldResult]) -> Dict[str, float]:
    """Compute mean ± std across CV folds for each metric."""
    df = pd.DataFrame([r.to_dict() for r in fold_results])
    summary = {}
    for col in df.columns:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std())
    return summary


# ─────────────────────────────────────────────
# Public: run one seed
# ─────────────────────────────────────────────

def run_seed(cfg: PipelineConfig, seed: int) -> Optional[SeedResult]:
    """
    Execute the full pipeline for a single seed:
        1. Load dataset
        2. Train/test split
        3. Stratified K-fold CV (optional)
        4. Final holdout test

    Args:
        cfg: Validated PipelineConfig.
        seed: Random seed for this run.

    Returns:
        SeedResult, or None if the pipeline fails for this seed.
    """
    logger.info("=== Seed %d | Dataset: %s ===", seed, cfg.dataset.name)
    np.random.seed(seed)

    # Load
    X, y = load_dataset(cfg.dataset.path)
    logger.info("Loaded %s: X=%s, y=%s", cfg.dataset.name, X.shape, y.shape)

    # Split
    stratify = y if cfg.preprocessing.stratify else None
    X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
        X, y,
        test_size=1 - cfg.preprocessing.train_test_split_ratio,
        random_state=seed,
        stratify=stratify,
    )

    cv_results: List[FoldResult] = []

    # CV
    if cfg.cv.enabled:
        skf = StratifiedKFold(
            n_splits=cfg.cv.n_splits,
            shuffle=cfg.cv.shuffle,
            random_state=seed,
        )
        for fold_idx, (tr, te) in enumerate(skf.split(X_train_full, y_train_full), 1):
            logger.info("  Fold %d/%d", fold_idx, cfg.cv.n_splits)
            fold_seed = seed + 10 * fold_idx  # deterministic, documented offset
            fold_result = _run_fold(
                X_train_full[tr], y_train_full[tr],
                X_train_full[te], y_train_full[te],
                cfg, fold_seed,
            )
            if fold_result is not None:
                cv_results.append(fold_result)

    cv_summary = _summarize_cv(cv_results) if cv_results else {}

    # Final holdout
    logger.info("  Final holdout test...")
    final_seed = seed + 999  # documented offset
    final_result = _run_fold(
        X_train_full, y_train_full,
        X_test_final, y_test_final,
        cfg, final_seed,
    )

    if final_result is None:
        logger.error("Final test failed for seed %d.", seed)
        return None

    return SeedResult(
        seed=seed,
        cv_summary=cv_summary,
        final_test=final_result,
        alpha=cfg.solver.alpha,
        depth=cfg.forest.max_depth,
        lambda_=cfg.solver.lambda_,
    )
