"""
src/optimization/backends/gurobi_backend.py
--------------------------------------------
Gurobi solver backend.

This is the highest-performance option.  When a valid Gurobi licence is
present, it will be used by default (see registry.py).

Gurobi-specific constructs isolated here:
    - gurobipy.Model
    - gurobipy.GRB.BINARY / GRB.MAXIMIZE
    - gurobipy.quicksum
    - model.setParam / model.addVars / model.addConstrs / model.optimize
    - model.Status / z[j].X / model.ObjVal

Nothing outside this file imports gurobipy.
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

# ── Lazy import: only fails if backend is actually instantiated ───────────────
try:
    import gurobipy as gp
    from gurobipy import GRB
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False
except Exception:          # licence error at import time on some platforms
    _AVAILABLE = False


class GurobiBackend(AbstractSolverBackend):
    """Gurobi MIP backend via gurobipy."""

    name = "gurobi"

    @classmethod
    def is_available(cls) -> bool:
        if not _AVAILABLE:
            return False
        # Attempt a trivial model to catch expired/missing licence
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            m = gp.Model(env=env)
            m.addVar(vtype=GRB.BINARY)
            m.update()
            m.dispose()
            env.dispose()
            return True
        except Exception as exc:
            logger.debug("Gurobi licence check failed: %s", exc)
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
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", int(problem.verbose))
            env.start()

            m = gp.Model(env=env)
            m.setParam("TimeLimit", problem.time_limit)
            m.setParam("OutputFlag", int(problem.verbose))

            z = m.addVars(L, vtype=GRB.BINARY, name="z")

            # Exact coverage constraint
            m.addConstrs(
                (gp.quicksum(A[i, j] * z[j] for j in range(L)) == 1
                 for i in range(n)),
                name="cover",
            )

            # Objective: maximise freq reward − loss penalty − sparsity penalty
            m.setObjective(
                gp.quicksum(freq[j] * z[j] for j in range(L))
                - (1 - problem.lambda_) * gp.quicksum(float(loss[j]) * z[j] for j in range(L))
                - problem.alpha * gp.quicksum(z[j] for j in range(L)),
                sense=GRB.MAXIMIZE,
            )

            logger.debug(
                "[gurobi] Solving: L=%d rules, n=%d samples, λ=%.2f, α=%.3f",
                L, n, problem.lambda_, problem.alpha,
            )

            _, elapsed = self._timed_solve(m.optimize)

            _status_map = {
                GRB.OPTIMAL:    "OPTIMAL",
                GRB.TIME_LIMIT: "TIME_LIMIT",
                GRB.INFEASIBLE: "INFEASIBLE",
                GRB.INF_OR_UNBD: "INF_OR_UNBD",
            }
            status = _status_map.get(m.Status, f"UNKNOWN({m.Status})")
            selected: List[int] = []
            obj_val: Optional[float] = None

            if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT):
                obj_val = m.ObjVal
                selected = [j for j in range(L) if z[j].X > 0.5]
                logger.debug(
                    "[gurobi] %s in %.2fs | obj=%.4f | selected=%d/%d",
                    status, elapsed, obj_val, len(selected), L,
                )
            else:
                logger.warning("[gurobi] Not solved. Status: %s", status)

            m.dispose()
            env.dispose()

            return SolverResult(
                selected_indices=selected,
                objective_value=obj_val,
                solve_time_seconds=elapsed,
                status=status,
                backend_name=self.name,
            )

        except Exception as exc:
            logger.error("[gurobi] Unexpected error: %s", exc, exc_info=True)
            return SolverResult(
                selected_indices=[],
                objective_value=None,
                solve_time_seconds=0.0,
                status="ERROR",
                backend_name=self.name,
            )
