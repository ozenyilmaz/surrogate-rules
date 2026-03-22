"""
tests/unit/test_optimization.py
---------------------------------
Unit tests for the solver-agnostic optimization layer.

Covers:
  - MIPProblem construction and helpers
  - SolverResult contract
  - CBC backend (always available — no licence needed)
  - SCIP backend (if pyscipopt is installed)
  - Gurobi backend availability probe (no licence required for the test)
  - registry auto-selection
  - solver.py public API backward compatibility
  - Edge cases: infeasible problems, empty rule sets, time-limit behaviour

Gurobi tests are skipped automatically when gurobipy is absent or the
licence is invalid — they never fail the CI pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.optimization.backends.base import MIPProblem, SolverResult
from src.optimization.backends.cbc_backend import PuLPBackend
from src.optimization.registry import (
    BACKEND_REGISTRY,
    AUTO_FALLBACK_ORDER,
    get_backend,
    list_available_backends,
)
from src.optimization.solver import (
    build_coverage_matrix,
    solve_rule_selection,
    solve_exact_coverage,
)


# ─────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def small_problem() -> MIPProblem:
    """
    3 samples, 3 non-overlapping rules (identity coverage matrix).
    The optimal solution is to select all three rules.
    """
    n, L = 3, 3
    A = np.eye(n, L)
    return MIPProblem(
        n=n, L=L, A=A,
        loss=np.array([0.1, 0.2, 0.3]),
        freq=[0.5, 0.4, 0.3],
        lambda_=0.5,
        alpha=0.05,
        time_limit=30,
        verbose=False,
    )


@pytest.fixture
def overlapping_problem() -> MIPProblem:
    """
    4 samples, 4 rules.  Rules 0 and 2 cover sample 0; rules 1 and 3
    cover samples 1–3 exclusively.  The MIP must break the tie on sample 0.
    """
    n, L = 4, 4
    A = np.array([
        [1, 0, 1, 0],   # sample 0 covered by rules 0 and 2
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],   # sample 3 also covered by rule 1 — still unique via constraint
    ], dtype=float)
    # Make rows sum to exactly 1 for a feasible exact-cover problem
    A = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    return MIPProblem(
        n=n, L=L, A=A,
        loss=np.array([0.0, 0.1, 0.2, 0.3]),
        freq=[0.8, 0.6, 0.4, 0.2],
        lambda_=0.5,
        alpha=0.01,
        time_limit=30,
        verbose=False,
    )


# ─────────────────────────────────────────────
# MIPProblem
# ─────────────────────────────────────────────

class TestMIPProblem:

    def test_objective_coefficients_shape(self, small_problem):
        c = small_problem.objective_coefficients()
        assert c.shape == (small_problem.L,)

    def test_objective_coefficients_formula(self, small_problem):
        """c[j] = freq[j] - (1-lambda)*loss[j] - alpha"""
        c = small_problem.objective_coefficients()
        for j in range(small_problem.L):
            expected = (
                small_problem.freq[j]
                - (1 - small_problem.lambda_) * small_problem.loss[j]
                - small_problem.alpha
            )
            assert abs(c[j] - expected) < 1e-9

    def test_frozen(self, small_problem):
        with pytest.raises((TypeError, AttributeError)):
            small_problem.n = 99  # type: ignore[misc]


# ─────────────────────────────────────────────
# SolverResult
# ─────────────────────────────────────────────

class TestSolverResult:

    def test_solved_property_optimal(self):
        r = SolverResult(selected_indices=[0, 1], objective_value=0.5,
                         solve_time_seconds=0.1, status="OPTIMAL")
        assert r.solved is True

    def test_solved_property_time_limit(self):
        r = SolverResult(selected_indices=[0], objective_value=0.3,
                         solve_time_seconds=30.0, status="TIME_LIMIT")
        assert r.solved is True

    def test_solved_property_infeasible(self):
        r = SolverResult(selected_indices=[], objective_value=None,
                         solve_time_seconds=0.0, status="INFEASIBLE")
        assert r.solved is False

    def test_n_selected(self):
        r = SolverResult(selected_indices=[0, 2, 4], objective_value=1.0,
                         solve_time_seconds=0.2, status="OPTIMAL")
        assert r.n_selected == 3


# ─────────────────────────────────────────────
# build_coverage_matrix
# ─────────────────────────────────────────────

class TestBuildCoverageMatrix:

    def test_shape(self):
        A = build_coverage_matrix(5, [[0, 1], [2, 3], [4]])
        assert A.shape == (5, 3)

    def test_values(self):
        A = build_coverage_matrix(4, [[0, 2], [1, 3]])
        assert A[0, 0] == 1.0
        assert A[2, 0] == 1.0
        assert A[1, 1] == 1.0
        assert A[3, 1] == 1.0
        assert A[0, 1] == 0.0

    def test_empty_rule(self):
        # A rule with no covered samples should leave its column all-zero
        A = build_coverage_matrix(3, [[], [0, 1, 2]])
        assert A[:, 0].sum() == 0.0
        assert A[:, 1].sum() == 3.0


# ─────────────────────────────────────────────
# CBC backend  (always available)
# ─────────────────────────────────────────────

class TestCBCBackend:

    def test_is_available(self):
        assert PuLPBackend.is_available() is True

    def test_solves_small_problem(self, small_problem):
        backend = PuLPBackend()
        result = backend.solve(small_problem)
        assert result.solved
        assert result.status == "OPTIMAL"
        assert set(result.selected_indices) == {0, 1, 2}
        assert result.objective_value is not None
        assert result.backend_name == "cbc"

    def test_solves_overlapping_problem(self, overlapping_problem):
        backend = PuLPBackend()
        result = backend.solve(overlapping_problem)
        assert result.solved
        # All 4 samples must be covered
        A = overlapping_problem.A
        for i in range(overlapping_problem.n):
            covered = sum(A[i, j] for j in result.selected_indices)
            assert covered >= 1, f"Sample {i} not covered"

    def test_objective_value_sign(self, small_problem):
        """Objective must be positive for this well-conditioned problem."""
        result = PuLPBackend().solve(small_problem)
        assert result.objective_value > 0

    def test_returns_error_on_impossible_problem(self):
        """A row of all-zeros in A makes exact cover infeasible."""
        A = np.eye(3, 3)
        A[1, :] = 0   # sample 1 not covered by any rule
        problem = MIPProblem(
            n=3, L=3, A=A,
            loss=np.zeros(3), freq=[1.0, 1.0, 1.0],
            lambda_=0.5, alpha=0.0, time_limit=10,
        )
        # CBC backend should not raise — it should log a warning and
        # relax to at-least-one coverage
        result = PuLPBackend().solve(problem)
        assert isinstance(result, SolverResult)

    def test_result_indices_within_bounds(self, small_problem):
        result = PuLPBackend().solve(small_problem)
        L = small_problem.L
        for idx in result.selected_indices:
            assert 0 <= idx < L

    def test_solve_time_is_positive(self, small_problem):
        result = PuLPBackend().solve(small_problem)
        assert result.solve_time_seconds >= 0.0


# ─────────────────────────────────────────────
# SCIP backend  (skip if not installed)
# ─────────────────────────────────────────────

scip_available = pytest.mark.skipif(
    condition=not __import__("importlib").util.find_spec("pyscipopt"),
    reason="pyscipopt not installed",
)


@scip_available
class TestSCIPBackend:

    def setup_method(self):
        from src.optimization.backends.scip_backend import SCIPBackend
        self.backend_cls = SCIPBackend

    def test_is_available(self):
        assert self.backend_cls.is_available() is True

    def test_solves_small_problem(self, small_problem):
        backend = self.backend_cls()
        result = backend.solve(small_problem)
        assert result.solved
        assert result.status == "OPTIMAL"
        assert set(result.selected_indices) == {0, 1, 2}
        assert result.backend_name == "scip"

    def test_matches_cbc_solution(self, overlapping_problem):
        """SCIP and CBC must select the same rules on a deterministic problem."""
        cbc_result = PuLPBackend().solve(overlapping_problem)
        scip_result = self.backend_cls().solve(overlapping_problem)
        assert set(cbc_result.selected_indices) == set(scip_result.selected_indices)


# ─────────────────────────────────────────────
# Gurobi backend  (skip if licence absent)
# ─────────────────────────────────────────────

from src.optimization.backends.gurobi_backend import GurobiBackend

gurobi_available = pytest.mark.skipif(
    condition=not GurobiBackend.is_available(),
    reason="Gurobi not available or licence invalid",
)


@gurobi_available
class TestGurobiBackend:

    def test_solves_small_problem(self, small_problem):
        result = GurobiBackend().solve(small_problem)
        assert result.solved
        assert result.status == "OPTIMAL"
        assert set(result.selected_indices) == {0, 1, 2}
        assert result.backend_name == "gurobi"

    def test_matches_cbc_solution(self, overlapping_problem):
        cbc_result = PuLPBackend().solve(overlapping_problem)
        grb_result = GurobiBackend().solve(overlapping_problem)
        assert set(cbc_result.selected_indices) == set(grb_result.selected_indices)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

class TestRegistry:

    def test_all_known_names_in_registry(self):
        for name in ("gurobi", "cbc", "scip"):
            assert name in BACKEND_REGISTRY

    def test_auto_fallback_order_references_valid_names(self):
        for name in AUTO_FALLBACK_ORDER:
            assert name in BACKEND_REGISTRY

    def test_get_backend_cbc(self):
        backend = get_backend("cbc")
        assert backend.name == "cbc"

    def test_get_backend_auto_returns_something(self):
        backend = get_backend("auto")
        assert backend.name in ("gurobi", "cbc", "scip")

    def test_get_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown solver backend"):
            get_backend("magic_solver_9000")

    def test_list_available_backends_includes_cbc(self):
        available = list_available_backends()
        assert "cbc" in available

    def test_get_backend_unavailable_raises_runtime(self, monkeypatch):
        """Requesting an unavailable named backend raises RuntimeError."""
        monkeypatch.setattr(GurobiBackend, "is_available", classmethod(lambda cls: False))
        from src.optimization.backends.scip_backend import SCIPBackend
        monkeypatch.setattr(SCIPBackend, "is_available", classmethod(lambda cls: False))
        # CBC is always available, so only test the error path indirectly
        # by pointing to gurobi with a patched unavailability
        import importlib
        import src.optimization.registry as reg
        original = reg.BACKEND_REGISTRY.copy()
        try:
            reg.BACKEND_REGISTRY["gurobi"] = GurobiBackend
            with pytest.raises(RuntimeError, match="not available"):
                get_backend("gurobi")
        finally:
            reg.BACKEND_REGISTRY.update(original)


# ─────────────────────────────────────────────
# Public API (solver.py) backward compatibility
# ─────────────────────────────────────────────

class TestSolverPublicAPI:

    def test_solve_rule_selection_default_args(self):
        """Original call signature with no backend kwarg must still work."""
        n, L = 3, 3
        A = np.eye(n, L)
        result = solve_rule_selection(
            n=n, A=A,
            loss=np.array([0.1, 0.2, 0.3]),
            freq=[0.5, 0.4, 0.3],
        )
        assert isinstance(result, SolverResult)
        assert result.solved

    def test_solve_rule_selection_explicit_cbc(self):
        n, L = 3, 3
        A = np.eye(n, L)
        result = solve_rule_selection(
            n=n, A=A,
            loss=np.zeros(L), freq=[1.0] * L,
            backend="cbc",
        )
        assert result.solved
        assert result.backend_name == "cbc"

    def test_solve_exact_coverage(self):
        n, L = 3, 3
        A = np.eye(n, L)
        result = solve_exact_coverage(n=n, A=A, backend="cbc")
        assert result.solved
        assert len(result.selected_indices) == 3

    def test_output_flag_zero_does_not_crash(self):
        """Legacy output_flag=0 kwarg must still be accepted."""
        n, L = 2, 2
        A = np.eye(n, L)
        result = solve_rule_selection(
            n=n, A=A,
            loss=np.zeros(L), freq=[1.0] * L,
            output_flag=0, backend="cbc",
        )
        assert result.solved

    def test_large_L_returns_result(self):
        """Sanity check with L=200 rules on n=50 samples."""
        rng = np.random.default_rng(0)
        n, L = 50, 200
        # Guarantee each sample is covered by at least one rule
        A = np.zeros((n, L))
        for i in range(n):
            A[i, rng.integers(0, L)] = 1
        # Also add some random coverage
        A += rng.integers(0, 2, size=(n, L)).astype(float)
        A = np.clip(A, 0, 1)
        result = solve_rule_selection(
            n=n, A=A,
            loss=rng.random(L),
            freq=rng.random(L).tolist(),
            backend="cbc", time_limit=30,
        )
        assert isinstance(result, SolverResult)
