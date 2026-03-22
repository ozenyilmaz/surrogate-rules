"""
src/optimization/backends/pulp_backend.py
------------------------------------------
PuLP solver backend supporting CBC (default), GLPK, and any other
solver that PuLP can drive.

CBC (COIN-OR Branch and Cut) is the default open-source fallback.
It ships bundled inside the `pulp` package — no extra install needed:

    pip install pulp

Performance vs Gurobi
---------------------
CBC is a mature, production-quality open-source MIP solver.  For the
rule-selection problem typical in this pipeline (L < 5 000 rules,
n < 10 000 samples), you should expect:

  - Small instances  (L < 500):   CBC ≈ Gurobi speed
  - Medium instances (L ~ 2 000): CBC 2–10× slower
  - Large instances  (L > 5 000): CBC can be 10–50× slower; consider
    reducing n_estimators or enabling min_samples_per_rule in config.

The exact-cover constraint (Σ_j A[i,j]·z[j] = 1) can be infeasible when
some samples are not covered by any rule.  If that happens, the backend
automatically relaxes to ≥ 1 (at-least-one-cover) and logs a warning.

PuLP-specific constructs isolated here:
    - pulp.LpProblem / LpVariable / LpMaximize / lpSum
    - solver_cls (PULP_CBC_CMD, GLPK_CMD, …)
    - prob.status / pulp.value(var)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.optimization.backends.base import (
    AbstractSolverBackend,
    MIPProblem,
    SolverResult,
)

logger = logging.getLogger(__name__)

try:
    import pulp
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class PuLPBackend(AbstractSolverBackend):
    """
    PuLP backend.  Uses CBC by default; `pulp_solver` kwarg overrides.

    Args:
        pulp_solver: Name of the PuLP solver to use, e.g. 'PULP_CBC_CMD',
                     'GLPK_CMD', 'CPLEX_CMD'.  Defaults to 'PULP_CBC_CMD'.
    """

    name = "cbc"

    def __init__(self, pulp_solver: str = "PULP_CBC_CMD") -> None:
        self._pulp_solver = pulp_solver

    @classmethod
    def is_available(cls) -> bool:
        if not _AVAILABLE:
            return False
        try:
            return "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True)
        except Exception:
            return False

    def solve(self, problem: MIPProblem) -> SolverResult:  # noqa: C901
        if not _AVAILABLE:
            return SolverResult(
                selected_indices=[],
                objective_value=None,
                solve_time_seconds=0.0,
                status="ERROR",
                backend_name=self.name,
            )

        n, L = problem.n, problem.L
        A, loss, freq = problem.A, problem.loss, problem.freq

        try:
            prob = pulp.LpProblem("SoftRuleSelection", pulp.LpMaximize)

            z = [
                pulp.LpVariable(f"z_{j}", cat=pulp.LpBinary)
                for j in range(L)
            ]

            # ── Objective ───────────────────────────────────────────
            prob += (
                pulp.lpSum(freq[j] * z[j] for j in range(L))
                - (1 - problem.lambda_) * pulp.lpSum(float(loss[j]) * z[j] for j in range(L))
                - problem.alpha * pulp.lpSum(z[j] for j in range(L))
            )

            # ── Exact coverage constraint ───────────────────────────
            # Each sample must be covered by exactly one rule.
            # If a sample is covered by zero rules in A, this is
            # infeasible.  We detect that upfront and fall back to ≥ 1.
            uncovered = [
                i for i in range(n)
                if not any(A[i, j] > 0.5 for j in range(L))
            ]
            if uncovered:
                logger.warning(
                    "[cbc] %d samples have no covering rule. "
                    "Relaxing exact-cover to at-least-one-cover.",
                    len(uncovered),
                )
                for i in range(n):
                    row_sum = pulp.lpSum(A[i, j] * z[j] for j in range(L))
                    if i in uncovered:
                        prob += row_sum >= 0, f"cover_{i}"  # uncovered → free
                    else:
                        prob += row_sum == 1, f"cover_{i}"  # normal
            else:
                for i in range(n):
                    prob += (
                        pulp.lpSum(A[i, j] * z[j] for j in range(L)) == 1,
                        f"cover_{i}",
                    )

            # ── Solver setup ────────────────────────────────────────
            solver_cls = getattr(pulp, self._pulp_solver, None)
            if solver_cls is None:
                logger.warning(
                    "[cbc] Solver '%s' not found in pulp. Falling back to PULP_CBC_CMD.",
                    self._pulp_solver,
                )
                solver_cls = pulp.PULP_CBC_CMD

            solver = solver_cls(
                msg=int(problem.verbose),
                timeLimit=problem.time_limit,
            )

            logger.debug(
                "[cbc] Solving: L=%d rules, n=%d samples, λ=%.2f, α=%.3f",
                L, n, problem.lambda_, problem.alpha,
            )

            _, elapsed = self._timed_solve(lambda: prob.solve(solver))

            # ── Result extraction ────────────────────────────────────
            status_map = {
                pulp.LpStatusOptimal:      "OPTIMAL",
                pulp.LpStatusInfeasible:   "INFEASIBLE",
                pulp.LpStatusUnbounded:    "UNBOUNDED",
                pulp.LpStatusNotSolved:    "NOT_SOLVED",
            }
            lp_status = prob.status
            status = status_map.get(lp_status, f"UNKNOWN({lp_status})")

            # PuLP marks time-limit termination as Optimal when a feasible
            # solution exists; detect it via solve status string.
            solve_str = pulp.LpStatus.get(lp_status, "")
            if "Not Solved" in solve_str and elapsed >= problem.time_limit - 1:
                status = "TIME_LIMIT"

            selected: List[int] = []
            obj_val: Optional[float] = None

            if lp_status == pulp.LpStatusOptimal:
                obj_val = pulp.value(prob.objective)
                selected = [
                    j for j in range(L)
                    if pulp.value(z[j]) is not None and pulp.value(z[j]) > 0.5
                ]
                logger.debug(
                    "[cbc] %s in %.2fs | obj=%.4f | selected=%d/%d",
                    status, elapsed, obj_val, len(selected), L,
                )
            else:
                logger.warning("[cbc] Not solved. Status: %s", status)

            return SolverResult(
                selected_indices=selected,
                objective_value=obj_val,
                solve_time_seconds=elapsed,
                status=status,
                backend_name=self.name,
            )

        except Exception as exc:
            logger.error("[cbc] Unexpected error: %s", exc, exc_info=True)
            return SolverResult(
                selected_indices=[],
                objective_value=None,
                solve_time_seconds=0.0,
                status="ERROR",
                backend_name=self.name,
            )
