"""
src/kpi/metrics.py
------------------
KPI and fidelity measurement for the surrogate rule model.

Responsibility  : Compute all evaluation metrics comparing the surrogate
                  (rule-based) model to the original Random Forest.
Architecture    : src/kpi/
Migration from  : fidelitymeasure.py  +  metric helpers in run_pipeline_resume.py

Key improvements:
  - All functions typed and documented.
  - Returns structured dataclasses, not ad-hoc dicts.
  - Fidelity metrics, expressive-power metrics, and Dice metrics are grouped.
  - `get_rule_coverage_and_pred` is here (was duplicated in the pipeline script).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import chain
from typing import List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.rules.forest import RulePath, sample_satisfies_path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Coverage + predictions
# ─────────────────────────────────────────────

def get_coverage_and_predictions(
    X: np.ndarray,
    paths: List[RulePath],
    labels: np.ndarray,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    For each sample, find all rules that fire and compute a predicted label.

    Prediction logic:
        - If any rules fire → majority vote (sum of labels >= 0 → +1 else -1)
        - If no rules fire → -1 (conservative fallback)

    Args:
        X: Feature matrix (n × d).
        paths: Selected rule paths.
        labels: Per-rule label.

    Returns:
        Tuple of:
            coverage: List of fired rule indices per sample.
            predictions: Predicted label array.
    """
    n = X.shape[0]
    coverage: List[List[int]] = []
    predictions = np.zeros(n)

    for i in range(n):
        fired = [idx for idx, path in enumerate(paths) if sample_satisfies_path(X[i], path)]
        coverage.append(fired)
        if len(fired) == 0:
            predictions[i] = -1
        else:
            vote = sum(labels[j] for j in fired)
            predictions[i] = 1 if vote >= 0 else -1

    return coverage, predictions


# ─────────────────────────────────────────────
# Fidelity metrics
# ─────────────────────────────────────────────

@dataclass
class FidelityMetrics:
    """Metrics measuring how faithfully the surrogate mirrors the RF."""
    disagreement_rate: float   # fraction of samples where RF ≠ surrogate
    feature_overlap_f1: float  # F1-like overlap of features used


def compute_fidelity_metrics(
    X: np.ndarray,
    clf: RandomForestClassifier,
    paths: List[RulePath],
    labels: np.ndarray,
) -> FidelityMetrics:
    """
    Compute disagreement rate and feature overlap between the RF and the surrogate.

    Args:
        X: Evaluation feature matrix.
        clf: Trained RandomForestClassifier (original model).
        paths: Selected surrogate rule paths.
        labels: Per-rule majority label.

    Returns:
        FidelityMetrics dataclass.
    """
    disagreement = _classification_disagreement(X, clf, paths, labels)
    overlap = _feature_overlap_score(clf, paths)
    return FidelityMetrics(
        disagreement_rate=disagreement,
        feature_overlap_f1=overlap,
    )


def _classification_disagreement(
    X: np.ndarray,
    clf: RandomForestClassifier,
    paths: List[RulePath],
    labels: np.ndarray,
) -> float:
    """
    Fraction of samples for which RF prediction ≠ surrogate prediction.
    """
    n = X.shape[0]
    y_rf = clf.predict(X)
    y_rule = np.zeros(n)

    fallback = float(np.round(np.mean(labels))) if len(labels) > 0 else 0.0

    for i in range(n):
        matched = False
        for p, path in enumerate(paths):
            if sample_satisfies_path(X[i], path):
                y_rule[i] = labels[p]
                matched = True
                break
        if not matched:
            y_rule[i] = fallback

    return round(float(np.mean(y_rf != y_rule)), 4)


def _feature_overlap_score(
    clf: RandomForestClassifier,
    paths: List[RulePath],
) -> float:
    """
    F1-like harmonic mean of:
        precision = (rule_features ∩ top_RF_features) / rule_features
        recall    = (rule_features ∩ top_RF_features) / top_RF_features

    Top RF features = top 5% by importance.
    """
    importance = clf.feature_importances_
    threshold = np.percentile(importance, 95)
    top_rf = set(idx for idx, val in enumerate(importance) if val >= threshold)
    extracted = set(node[0] for node in chain.from_iterable(paths))

    if not extracted or not top_rf:
        return 0.0

    intersection = len(top_rf & extracted)
    precision = intersection / len(extracted)
    recall = intersection / len(top_rf)

    if (precision + recall) == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 4)


# ─────────────────────────────────────────────
# Rule complexity / expressiveness
# ─────────────────────────────────────────────

@dataclass
class ExpressivePowerMetrics:
    """Metrics characterising the complexity of the selected rule set."""
    rule_condition_size: float    # mean number of unique features per rule (RCS)
    feature_condition_size: float # mean feature set size (FCS)  — same as RCS here
    feature_interaction_extent: float  # number of unique feature pairs (FIE)
    dice_mean: float              # mean pairwise Sørensen–Dice between rule feature sets
    dice_var: float               # variance of pairwise Dice scores


def compute_expressive_power(paths: List[RulePath]) -> ExpressivePowerMetrics:
    """
    Compute expressiveness/complexity metrics for a set of rule paths.

    Args:
        paths: Selected rule paths.

    Returns:
        ExpressivePowerMetrics dataclass.
    """
    rule_lengths = []
    feature_sets = []
    pairs: set = set()

    for rule in paths:
        feats = [c[0] for c in rule]
        fs = set(feats)
        feature_sets.append(fs)
        rule_lengths.append(len(fs))
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                pairs.add(tuple(sorted([feats[i], feats[j]])))

    rcs = float(np.mean(rule_lengths)) if rule_lengths else 0.0
    fcs = float(np.mean([len(fs) for fs in feature_sets])) if feature_sets else 0.0
    fie = float(len(pairs))

    dice_mean, dice_var = _compute_dice_metrics(paths)

    return ExpressivePowerMetrics(
        rule_condition_size=rcs,
        feature_condition_size=fcs,
        feature_interaction_extent=fie,
        dice_mean=dice_mean,
        dice_var=dice_var,
    )


def _dice(a: set, b: set) -> float:
    inter = len(a & b)
    denom = len(a) + len(b)
    return 2 * inter / denom if denom > 0 else 0.0


def _compute_dice_metrics(paths: List[RulePath]) -> Tuple[float, float]:
    if len(paths) < 2:
        return 0.0, 0.0
    fsets = [set(c[0] for c in r) for r in paths]
    vals = [_dice(fsets[i], fsets[j])
            for i in range(len(fsets))
            for j in range(i + 1, len(fsets))]
    return float(np.mean(vals)), float(np.var(vals))


# ─────────────────────────────────────────────
# Hamming loss
# ─────────────────────────────────────────────

def hamming_loss_rule_model(
    y_true: np.ndarray,
    coverage: List[List[int]],
) -> float:
    """
    Multi-label–style Hamming loss for the rule model.

    For positive samples: penalises if no rule fires (missed).
    For negative samples: penalises for each rule that fires (false alarms).

    Args:
        y_true: True binary labels.
        coverage: Per-sample list of fired rule indices.

    Returns:
        Normalised Hamming loss in [0, 1].
    """
    total = 0
    for y, cov in zip(y_true, coverage):
        if y == 1:
            total += int(len(cov) == 0)
        else:
            total += len(cov)
    return total / len(y_true)
