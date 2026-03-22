"""
src/optimization/backends/base.py
----------------------------------
Abstract base class for all MIP solver backends.

Every backend receives the same MIPProblem specification and must return
a SolverResult.  The pipeline never imports gurobipy, pulp, or pyscipopt
directly — it always goes through this interface.

Adding a new backend:
    1. Subclass AbstractSolverBackend.
    2. Implement solve().
    3. Register it in src/optimization/registry.py.
    4. Add its name to SolverConfig.backend in src/config/models.py.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────
# Problem specification  (solver-agnostic)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class MIPProblem:
    """
    Complete, solver-agnostic description of the rule-selection MIP.

    Objective (maximise):
        Σ_j  freq[j]·z[j]
        − (1 − lambda_) · Σ_j  loss[j]·z[j]
        − alpha          · Σ_j  z[j]

    Subject to:
        Σ_j  A[i,j]·z[j] = 1   ∀ i ∈ {0…n−1}   (exact cover)
        z[j] ∈ {0, 1}           ∀ j ∈ {0…L−1}

    Attributes:
        n         : Number of training samples.
        L         : Number of candidate rules.
        A         : Binary coverage matrix, shape (n, L).
                    A[i,j] = 1 iff rule j covers sample i.
        loss      : Per-rule normalised misclassification rate, length L.
        freq      : Per-rule fusion/overlap score (used as a reward), length L.
        lambda_   : Trade-off weight ∈ [0, 1].
                    λ=1 → pure coverage reward, λ=0 → pure loss minimisation.
        alpha     : Sparsity penalty ≥ 0.  Higher → fewer rules selected.
        time_limit: Wall-clock budget in seconds.
        verbose   : If True, the backend may emit solver output.
    """
    n: int
    L: int
    A: np.ndarray
    loss: np.ndarray
    freq: List[float]
    lambda_: float = 0.5
    alpha: float = 0.05
    time_limit: int = 180
    verbose: bool = False

    def objective_coefficients(self) -> np.ndarray:
        """
        Pre-compute the per-rule objective coefficient vector c, where
        the objective is  max  Σ_j c[j]·z[j].

        c[j] = freq[j] − (1−λ)·loss[j] − α

        This allows backends that accept a coefficient vector (e.g. scipy)
        to avoid reconstructing the objective from parts.
        """
        freq_arr = np.asarray(self.freq, dtype=float)
        loss_arr = np.asarray(self.loss, dtype=float)
        return freq_arr - (1.0 - self.lambda_) * loss_arr - self.alpha


# ─────────────────────────────────────────────
# Solve result  (identical to the old SolverResult)
# ─────────────────────────────────────────────

@dataclass
class SolverResult:
    """
    Outcome of a MIP solve.  Identical contract to the previous
    gurobipy-only SolverResult so the rest of the pipeline is unaffected.
    """
    selected_indices: List[int]
    objective_value: Optional[float]
    solve_time_seconds: float
    status: str          # 'OPTIMAL' | 'TIME_LIMIT' | 'INFEASIBLE' | 'ERROR'
    backend_name: str = "unknown"
    model_name: str = "SoftMIP"

    @property
    def n_selected(self) -> int:
        return len(self.selected_indices)

    @property
    def solved(self) -> bool:
        """True when a feasible (possibly sub-optimal) solution is available."""
        return self.status in ("OPTIMAL", "TIME_LIMIT")


# ─────────────────────────────────────────────
# Abstract backend
# ─────────────────────────────────────────────

class AbstractSolverBackend(ABC):
    """
    Contract every solver backend must satisfy.

    Subclasses must implement:
        - is_available() → bool
        - solve(problem)  → SolverResult
    """

    #: Short name used in config YAML and log messages.
    name: str = "abstract"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """
        Return True iff the backend's solver library is importable AND
        a valid licence exists (where applicable).

        This must never raise — catching ImportError / licence errors
        is the backend's responsibility.
        """

    @abstractmethod
    def solve(self, problem: MIPProblem) -> SolverResult:
        """
        Solve the given MIPProblem and return a SolverResult.

        Implementations must:
          - Respect problem.time_limit.
          - Respect problem.verbose.
          - Never raise on solver failure — return a SolverResult with
            status='ERROR' and an empty selected_indices list instead.
          - Log at DEBUG level; never print directly.
        """

    # ── Shared timing helper ──────────────────────────────────────────
    @staticmethod
    def _timed_solve(solve_fn) -> tuple[any, float]:
        """Run solve_fn(), return (result, elapsed_seconds)."""
        t0 = time.perf_counter()
        result = solve_fn()
        return result, time.perf_counter() - t0
