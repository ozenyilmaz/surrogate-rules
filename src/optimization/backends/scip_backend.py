"""
src/optimization/backends/scip_backend.py
------------------------------------------
SCIP solver backend via pyscipopt.

SCIP (Solving Constraint Integer Programs) is a world-class open-source
MIP solver developed at Zuse Institute Berlin.  It is free for academic
and research use and competitive with Gurobi on many problem classes.

Install:
    pip install pyscipopt

Performance vs CBC
------------------
SCIP is generally faster than CBC on hard MIP instances, especially
those with dense constraint matrices (which the covering constraint
can become when L is large).  For the rule-selection problem:

  - Small/medium instances: SCIP ≈ CBC; both solve to optimality quickly.
  - Large instances (L > 2 000): SCIP typically finds better bounds faster.
  - Licence: completely free for academic use; no registration needed.

SCIP-specific constructs isolated here:
    - pyscipopt.Model / addVar / addCons / quicksum / setObjective / optimize
    - model.getStatus / model.getVal / model.getObjVal
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
    import pyscipopt
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class SCIPBackend(AbstractSolverBackend):
    """SCIP MIP backend via pyscipopt."""

    name = "scip"

    @classmethod
    def is_available(cls) -> bool:
        if not _AVAILABLE:
            return False
        try:
            # Quick model to confirm SCIP is functional
            m = pyscipopt.Model()
            m.hideOutput(True)
            v = m.addVar(vtype="B")
            m.setObjective(v, "maximize")
            m.optimize()
            return True
        except Exception as exc:
            logger.debug("SCIP availability check failed: %s", exc)
            return False

    def solve(self, problem: MIPProblem) -> SolverResult:
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
            m = pyscipopt.Model("SoftRuleSelection")
            m.hideOutput(not problem.verbose)
            m.setParam("limits/time", float(problem.time_limit))

            z = [m.addVar(f"z_{j}", vtype="B") for j in range(L)]

            # ── Exact coverage constraint ───────────────────────────
            uncovered = [
                i for i in range(n)
                if not any(A[i, j] > 0.5 for j in range(L))
            ]
            if uncovered:
                logger.warning(
                    "[scip] %d samples have no covering rule. "
                    "Relaxing exact-cover to at-least-one-cover for those samples.",
                    len(uncovered),
                )

            uncovered_set = set(uncovered)
            for i in range(n):
                row = pyscipopt.quicksum(A[i, j] * z[j] for j in range(L))
                if i in uncovered_set:
                    m.addCons(row >= 0)
                else:
                    m.addCons(row == 1)

            # ── Objective ───────────────────────────────────────────
            obj = (
                pyscipopt.quicksum(freq[j] * z[j] for j in range(L))
                - (1 - problem.lambda_) * pyscipopt.quicksum(float(loss[j]) * z[j] for j in range(L))
                - problem.alpha * pyscipopt.quicksum(z[j] for j in range(L))
            )
            m.setObjective(obj, "maximize")

            logger.debug(
                "[scip] Solving: L=%d rules, n=%d samples, λ=%.2f, α=%.3f",
                L, n, problem.lambda_, problem.alpha,
            )

            _, elapsed = self._timed_solve(m.optimize)

            scip_status = m.getStatus()   # 'optimal', 'timelimit', 'infeasible', …

            status_map = {
                "optimal":    "OPTIMAL",
                "timelimit":  "TIME_LIMIT",
                "infeasible": "INFEASIBLE",
                "unbounded":  "UNBOUNDED",
            }
            status = status_map.get(scip_status, f"UNKNOWN({scip_status})")

            selected: List[int] = []
            obj_val: Optional[float] = None

            if scip_status in ("optimal", "timelimit"):
                try:
                    obj_val = m.getObjVal()
                    selected = [
                        j for j in range(L)
                        if m.getVal(z[j]) > 0.5
                    ]
                except Exception:
                    # getObjVal / getVal can raise if no solution was found
                    # before the time limit
                    pass

                if selected:
                    logger.debug(
                        "[scip] %s in %.2fs | obj=%.4f | selected=%d/%d",
                        status, elapsed, obj_val or 0.0, len(selected), L,
                    )
                else:
                    logger.warning("[scip] Solved but no variables extracted.")
                    status = "INFEASIBLE"
            else:
                logger.warning("[scip] Not solved. Status: %s", status)

            return SolverResult(
                selected_indices=selected,
                objective_value=obj_val,
                solve_time_seconds=elapsed,
                status=status,
                backend_name=self.name,
            )

        except Exception as exc:
            logger.error("[scip] Unexpected error: %s", exc, exc_info=True)
            return SolverResult(
                selected_indices=[],
                objective_value=None,
                solve_time_seconds=0.0,
                status="ERROR",
                backend_name=self.name,
            )
