"""
src/rules/forest.py
-------------------
Random Forest training and rule (path) extraction.

Responsibility  : Train a RandomForestClassifier, then walk every tree to
                  extract decision paths (sequences of split conditions).
Architecture    : src/rules/
Migration from  : warm_start.py (computePaths, computeLoss, computeScore,
                  computeSample, checkTrees, checkTreePaths, sorensenDice)

Key improvements:
  - All public functions typed and docstring'd.
  - `RulePath` type alias makes the data contract explicit.
  - `computePaths` no longer prints; caller controls logging.
  - `computeLoss` returns a typed dataclass instead of a 6-tuple.
  - Recursion in `computeSample` replaced with iteration to avoid
    deep-recursion issues on long paths.
  - `findParents` is now a private helper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import numpy as np
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────

# A single split condition: (feature_index, threshold, direction)
# direction: 'L' means feature <= threshold (go left), 'R' means feature > threshold
SplitCondition = Tuple[int, float, str]

# A rule path is an ordered sequence of split conditions from root → leaf
RulePath = List[SplitCondition]


# ─────────────────────────────────────────────
# Path extraction helpers
# ─────────────────────────────────────────────

def _find_parents(tree) -> np.ndarray:
    """
    Build a parent-pointer array for a sklearn tree structure.

    Convention (preserved from original):
        parent > 0  → arrived via left child  (feature <= threshold)
        parent < 0  → arrived via right child (feature >  threshold)
        parent == 0.1 / -0.1 → root (node 0) arrived via left / right

    Args:
        tree: sklearn DecisionTree internal ``tree_`` object.

    Returns:
        Float array of length n_nodes; NaN for the root.
    """
    n = len(tree.children_right)
    parents = np.full(n, np.nan)
    for i in range(n):
        for j in range(n):
            if tree.children_left[j] == i:
                parents[i] = 0.1 if j == 0 else j
            if tree.children_right[j] == i:
                parents[i] = -0.1 if j == 0 else -j
    return parents


def _extract_paths_from_tree(
    tree,
    max_depth: int,
) -> Tuple[List[RulePath], Set[SplitCondition]]:
    """
    Extract all valid leaf paths from a single sklearn decision tree.

    Args:
        tree: sklearn ``tree_`` object.
        max_depth: Maximum allowed path length.

    Returns:
        Tuple of (list of paths, set of all nodes seen in the tree).
    """
    leaf_nodes = np.where(tree.children_left == -1)[0].tolist()
    parents = _find_parents(tree)

    tree_paths: List[RulePath] = []
    tree_nodes: Set[SplitCondition] = set()

    for leaf in leaf_nodes:
        path: RulePath = []
        parent = parents[leaf]

        while not np.isnan(parent):
            node = round(abs(parent))
            direction = "L" if parent > 0 else "R"
            condition: SplitCondition = (tree.feature[node], tree.threshold[node], direction)
            path.append(condition)
            tree_nodes.add(condition)
            parent = parents[node]

        if 0 < len(path) <= max_depth:
            tree_paths.append(list(reversed(path)))

    return tree_paths, tree_nodes


# ─────────────────────────────────────────────
# Public: forest training + path extraction
# ─────────────────────────────────────────────

@dataclass
class ForestPaths:
    """Container for all extracted paths and the trained forest."""
    paths: List[RulePath]
    clf: RandomForestClassifier
    trees_pathed: List[List[RulePath]]     # per-tree list of paths
    trees_noded: List[Set[SplitCondition]] # per-tree set of nodes


def compute_forest_paths(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    max_depth: int,
    seed: int,
    clf: Optional[RandomForestClassifier] = None,
) -> ForestPaths:
    """
    Train a Random Forest (or reuse a provided one) and extract all
    decision paths up to ``max_depth``.

    Args:
        X: Feature matrix (n_samples × n_features).
        y: Label vector in {-1, +1}.
        n_estimators: Number of trees.
        max_depth: Maximum depth of each tree / max path length.
        seed: Random seed for reproducibility.
        clf: Optional pre-trained classifier; if given, paths are re-extracted
             but the forest is NOT retrained.

    Returns:
        ForestPaths dataclass.
    """
    if clf is None:
        logger.debug("Training RandomForest (n=%d, depth=%d, seed=%d)", n_estimators, max_depth, seed)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
        )
        clf.fit(X, y)
    else:
        logger.debug("Reusing provided RandomForest; re-extracting paths.")

    all_paths: List[RulePath] = []
    trees_pathed: List[List[RulePath]] = []
    trees_noded: List[Set[SplitCondition]] = []

    for estimator in clf.estimators_:
        tree_paths, tree_nodes = _extract_paths_from_tree(estimator.tree_, max_depth)
        all_paths.extend(tree_paths)
        trees_pathed.append(tree_paths)
        trees_noded.append(tree_nodes)

    return ForestPaths(
        paths=all_paths,
        clf=clf,
        trees_pathed=trees_pathed,
        trees_noded=trees_noded,
    )


# ─────────────────────────────────────────────
# Sample evaluation
# ─────────────────────────────────────────────

def sample_satisfies_path(x: np.ndarray, path: RulePath) -> bool:
    """
    Check whether a single sample satisfies all conditions in a rule path.

    Iterative version (avoids Python recursion limit on deep paths).

    Args:
        x: Feature vector of length n_features.
        path: Sequence of (feature, threshold, direction) conditions.

    Returns:
        True if all conditions are satisfied.
    """
    for (feature_idx, threshold, direction) in path:
        if direction == "L" and not (x[feature_idx] <= threshold):
            return False
        if direction == "R" and not (x[feature_idx] > threshold):
            return False
    return True


# ─────────────────────────────────────────────
# Loss computation
# ─────────────────────────────────────────────

@dataclass
class RuleStats:
    """Per-rule statistics computed from training data."""
    paths: List[RulePath]
    samples: List[List[int]]   # indices of covered training samples
    labels: np.ndarray         # majority class label per rule
    loss: np.ndarray           # normalised misclassification rate per rule
    weights: np.ndarray        # raw coverage count per rule
    fusion: List[float]        # normalised path-overlap score per rule


def compute_rule_stats(
    X: np.ndarray,
    y: np.ndarray,
    paths: List[RulePath],
    min_samples_fraction: Optional[float] = None,
) -> Optional[RuleStats]:
    """
    For each rule path, compute:
        - covered sample indices
        - majority-vote label
        - misclassification loss (normalised)
        - coverage weight
        - pairwise path-overlap fusion score

    Args:
        X: Training feature matrix.
        y: Training labels.
        paths: List of rule paths.
        min_samples_fraction: If set, drop paths covering fewer than
            ceil(min_samples_fraction × n) samples.

    Returns:
        RuleStats, or None if no paths survive pruning.
    """
    n = X.shape[0]
    logger.debug("Assigning %d samples to %d rules...", n, len(paths))

    # Assign samples to rules
    samples: List[List[int]] = []
    for path in paths:
        covered = [i for i in range(n) if sample_satisfies_path(X[i], path)]
        samples.append(covered)

    # Optional pruning
    if min_samples_fraction is not None:
        min_count = int(np.ceil(min_samples_fraction * n))
        filtered = [(p, s) for p, s in zip(paths, samples) if len(s) >= min_count]
        if not filtered:
            logger.warning("All paths pruned by min_samples_fraction=%.4f", min_samples_fraction)
            return None
        paths, samples = zip(*filtered)  # type: ignore[assignment]
        paths = list(paths)
        samples = list(samples)

    L = len(paths)
    loss = np.zeros(L)
    labels = np.zeros(L)
    weights = np.zeros(L)

    logger.debug("Computing loss and labels for %d rules...", L)
    for i, sample_idx in enumerate(samples):
        if not sample_idx:
            continue
        y_sub = y[sample_idx]
        rule_label = mode(y_sub, keepdims=False).mode
        labels[i] = rule_label
        loss[i] = 1.0 - accuracy_score(y_sub, np.full(len(y_sub), rule_label))
        weights[i] = len(y_sub)

    # Pairwise path-overlap (fusion) score
    fusion: List[float] = []
    for i, pi in enumerate(paths):
        overlap_sum = 0.0
        for j, pj in enumerate(paths):
            if i == j:
                continue
            shared = sum(1 for k in range(min(len(pi), len(pj))) if pi[k] == pj[k])
            overlap_sum += (2 * shared) / (len(pi) + len(pj))
        fusion.append(overlap_sum)

    if not fusion:
        return None

    def _normalise(arr):
        lo, hi = min(arr), max(arr)
        return arr if lo == hi else [(float(v) - lo) / (hi - lo) for v in arr]

    fusion_norm = _normalise(fusion)
    loss_norm = _normalise(loss.tolist())

    return RuleStats(
        paths=paths,
        samples=samples,
        labels=labels,
        loss=np.array(loss_norm),
        weights=weights,
        fusion=fusion_norm,
    )


# ─────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────

def compute_surrogate_score(
    X: np.ndarray,
    y: np.ndarray,
    paths: List[RulePath],
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Evaluate the surrogate rule model on (X, y).

    For each sample:
        - If multiple rules fire, use the label of the highest-weight rule.
        - If no rule fires, fall back to global majority label.

    Args:
        X: Feature matrix.
        y: True labels.
        paths: Selected rule paths.
        labels: Per-rule majority label.
        weights: Per-rule coverage weight.

    Returns:
        Classification accuracy in [0, 1].
    """
    n = X.shape[0]
    y_pred = np.zeros(n)
    fallback = float(np.round(np.mean(labels))) if len(labels) > 0 else 0.0

    for i in range(n):
        covered_idx = [p for p, path in enumerate(paths) if sample_satisfies_path(X[i], path)]
        if len(covered_idx) > 1:
            y_pred[i] = labels[covered_idx[np.argmax([weights[j] for j in covered_idx])]]
        elif len(covered_idx) == 1:
            y_pred[i] = labels[covered_idx[0]]
        else:
            y_pred[i] = fallback

    return float(accuracy_score(y, y_pred))


# ─────────────────────────────────────────────
# Representativeness
# ─────────────────────────────────────────────

def check_tree_representativeness(
    selected_paths: List[RulePath],
    trees_noded: List[Set[SplitCondition]],
) -> float:
    """
    Fraction of trees that share at least one node with the selected rules.
    """
    if not trees_noded:
        return 0.0
    selected_nodes = {node for path in selected_paths for node in path}
    covered = sum(1 for nodes in trees_noded if nodes & selected_nodes)
    return covered / len(trees_noded)


def check_path_representativeness(
    selected_paths: List[RulePath],
    trees_pathed: List[List[RulePath]],
) -> float:
    """
    Fraction of trees that contain at least one path identical to a selected rule.
    """
    if not trees_pathed:
        return 0.0
    selected_set = [tuple(map(tuple, p)) for p in selected_paths]
    covered = sum(
        1 for tree_paths in trees_pathed
        if any(tuple(map(tuple, tp)) in selected_set for tp in tree_paths)
    )
    return covered / len(trees_pathed)


def sorensen_dice(z1: List, z2: List) -> float:
    """Sørensen–Dice similarity between two lists (treated as sets)."""
    numerator = 2 * len(set(z1) & set(z2))
    denominator = len(z1) + len(z2)
    return round(numerator / denominator, 2) if denominator > 0 else 0.0
