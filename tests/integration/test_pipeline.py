"""
tests/integration/test_pipeline.py
------------------------------------
Integration tests for the end-to-end pipeline.

These tests run the full pipeline on a tiny synthetic dataset so they do NOT
require a Gurobi licence or a running MongoDB instance.

Gurobi mock strategy
--------------------
The solver is mocked by monkeypatching `src.optimization.solver.solve_rule_selection`
to return a deterministic SolverResult.  This lets the rest of the pipeline
(forest, stats, KPI, etc.) be exercised with real computation.

MongoDB mock strategy
---------------------
Use mongomock (pip install mongomock) or simply disable storage in the test config.
The MongoRepository is tested in isolation in tests/unit/test_storage.py.

To run integration tests:
    pytest tests/integration/ -v
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config.models import (
    CVConfig,
    DatasetConfig,
    ForestConfig,
    PipelineConfig,
    PreprocessingConfig,
    RunConfig,
    SolverConfig,
    StorageConfig,
)
from src.optimization.solver import SolverResult
from src.pipeline.runner import run_seed


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_config(dataset_path: Path) -> PipelineConfig:
    return PipelineConfig(
        dataset=DatasetConfig(name="test", path=dataset_path),
        preprocessing=PreprocessingConfig(train_test_split_ratio=0.75, stratify=False),
        forest=ForestConfig(n_estimators=10, max_depth=2),
        solver=SolverConfig(**{"lambda": 0.5, "alpha": 0.05, "time_limit_seconds": 60}),
        cv=CVConfig(enabled=False),   # skip CV for speed
        storage=StorageConfig(enabled=False),
        run=RunConfig(num_seeds=1),
    )


def _make_arff(tmp_path: Path) -> Path:
    """Write a minimal 2-feature .arff file with 40 samples."""
    lines = [
        "@relation test",
        "@attribute f1 NUMERIC",
        "@attribute f2 NUMERIC",
        "@attribute class {e,p}",
        "@data",
    ]
    rng = np.random.default_rng(0)
    for i in range(40):
        f1, f2 = rng.random(2).round(3)
        cls = "p" if f1 + f2 > 1 else "e"
        lines.append(f"{f1},{f2},{cls}")
    p = tmp_path / "mushroom_test.arff"
    p.write_text("\n".join(lines))
    return p


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

class TestPipelineIntegration:

    @pytest.fixture
    def arff_path(self, tmp_path):
        return _make_arff(tmp_path)

    @pytest.fixture
    def cfg(self, arff_path):
        return _make_config(arff_path)

    def test_run_seed_returns_seed_result_with_solver_mock(self, cfg):
        """
        Full pipeline with Gurobi mocked: verifies all steps except the MIP solve
        execute correctly and return a valid SeedResult.
        """
        # We need to return a valid SolverResult pointing at real rule indices.
        # The mock intercepts after coverage matrix is built, so we return index 0.
        mock_result = SolverResult(
            selected_indices=[0],
            objective_value=0.5,
            solve_time_seconds=0.01,
            status="OPTIMAL",
        )

        with patch("src.pipeline.runner.solve_rule_selection", return_value=mock_result):
            result = run_seed(cfg, seed=42)

        assert result is not None
        assert result.seed == 42
        assert result.final_test.rule_count >= 1
        assert 0.0 <= result.final_test.surrogate_acc <= 1.0
        assert 0.0 <= result.final_test.rf_acc <= 1.0

    def test_run_seed_no_solution_returns_none(self, cfg):
        """If MIP returns no selected rules, run_seed should return None."""
        mock_result = SolverResult(
            selected_indices=[],
            objective_value=None,
            solve_time_seconds=0.0,
            status="INFEASIBLE",
        )

        with patch("src.pipeline.runner.solve_rule_selection", return_value=mock_result):
            result = run_seed(cfg, seed=42)

        assert result is None
