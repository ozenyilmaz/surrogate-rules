"""
src/optimization/solver.py
--------------------------
Public API of the optimization layer.  Solver-agnostic.

This module is the ONLY import used by the rest of the pipeline.
Signature is identical to the previous gurobipy-only version so no
other file needs to change.

All solver-specific code lives in:
    src/optimization/backends/gurobi_backend.py
    src/optimization/backends/cbc_backend.py   (PuLP/CBC)
    src/optimization/backends/scip_backend.py
    src/optimization/registry.py               (auto-selection)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.optimization.backends.base import MIPProblem, SolverResult   # re-exported
from src.optimization.registry import get_backend

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Coverage matrix  (unchanged)
# ─────────────────────────────────────────────

def build_coverage_matrix(n: int, samples: List[List[int]]) -> np.ndarray:
    """
    Build the n × L binary coverage matrix A where A[i, j] = 1 iff
    sample i is covered by rule j.

    Args:
        n       : Number of training samples.
        samples : List of sample-index lists, one per rule.

    Returns:
        Binary numpy array of shape (n, L).
    """
    L = len(samples)
    A = np.zeros((n, L), dtype=float)
    for j, idx_list in enumerate(samples):
        A[idx_list, j] = 1.0
    return A


# ─────────────────────────────────────────────
# Main solver  (now backend-agnostic)
# ─────────────────────────────────────────────

def solve_rule_selection(
    n: int,
    A: np.ndarray,
    loss: np.ndarray,
    freq: List[float],
    lambda_: float = 0.5,
    alpha: float = 0.05,
    time_limit: int = 180,
    output_flag: int = 0,
    backend: str = "auto",
) -> SolverResult:
    """
    Solve the soft covering MIP for rule selection.

    Objective (maximise):
        Σ_j freq[j]·z[j]
        − (1 − λ) · Σ_j loss[j]·z[j]
        − α        · Σ_j z[j]

    Subject to:
        Σ_j A[i,j]·z[j] = 1   ∀ i   (exact cover)
        z[j] ∈ {0, 1}

    Args:
        n          : Number of training samples.
        A          : Coverage matrix (n × L).
        loss       : Per-rule normalised misclassification loss, length L.
        freq       : Per-rule fusion/overlap score, length L.
        lambda_    : Trade-off weight ∈ [0, 1].
        alpha      : Sparsity penalty ≥ 0.
        time_limit : Wall-clock budget in seconds.
        output_flag: 0 = silent, 1 = verbose solver output.
        backend    : 'gurobi' | 'cbc' | 'scip' | 'auto'.
                     'auto' tries Gurobi → SCIP → CBC.

    Returns:
        SolverResult with selected rule indices and metadata.
    """
    problem = MIPProblem(
        n=n,
        L=A.shape[1],
        A=A,
        loss=np.asarray(loss, dtype=float),
        freq=list(freq),
        lambda_=lambda_,
        alpha=alpha,
        time_limit=time_limit,
        verbose=bool(output_flag),
    )
    solver_backend = get_backend(backend)
    return solver_backend.solve(problem)


# ─────────────────────────────────────────────
# Hard CSP variant  (feasibility, no objective)
# ─────────────────────────────────────────────

def solve_exact_coverage(
    n: int,
    A: np.ndarray,
    time_limit: int = 180,
    output_flag: int = 0,
    backend: str = "auto",
) -> SolverResult:
    """
    Find any z ∈ {0,1}^L s.t. Σ_j A[i,j]·z[j] = 1 ∀i.
    Implemented as solve_rule_selection with zero coefficients.
    """
    L = A.shape[1]
    return solve_rule_selection(
        n=n, A=A,
        loss=np.zeros(L), freq=[0.0] * L,
        lambda_=1.0, alpha=0.0,
        time_limit=time_limit,
        output_flag=output_flag,
        backend=backend,
    )
