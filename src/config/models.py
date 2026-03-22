"""
src/config/models.py
--------------------
Pydantic-based configuration models for the entire pipeline.

Every section maps to a YAML block in configs/<experiment>.yaml.
Using pydantic v2 style (also compatible with v1 via model_validator).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class DatasetConfig(BaseModel):
    name: str = Field(..., description="Human-readable dataset identifier, e.g. 'mushroom'")
    path: Path = Field(..., description="Absolute or relative path to the .arff file")
    format: str = Field(default="arff", description="File format; 'arff' supported now, extensible")
    target_column: Optional[str] = Field(default=None, description="Target column name; None = auto-detect")

    @field_validator("path")
    @classmethod
    def path_must_exist(cls, v: Path) -> Path:
        if not Path(v).exists():
            raise ValueError(f"Dataset file not found: {v}")
        return v


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

class PreprocessingConfig(BaseModel):
    label_encoding: str = Field(
        default="minus_one_plus_one",
        description="Label encoding strategy: 'minus_one_plus_one' | 'zero_one'"
    )
    drop_na: bool = Field(default=True)
    train_test_split_ratio: float = Field(default=0.75, ge=0.1, le=0.99)
    stratify: bool = Field(default=True)


# ─────────────────────────────────────────────
# Forest (warm-start)
# ─────────────────────────────────────────────

class ForestConfig(BaseModel):
    n_estimators: int = Field(default=500, ge=1)
    max_depth: int = Field(default=3, ge=1, le=20)


# ─────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────

class SolverConfig(BaseModel):
    lambda_: float = Field(default=0.5, ge=0.0, le=1.0, alias="lambda")
    alpha: float = Field(default=0.05, ge=0.0, le=1.0)
    time_limit_seconds: int = Field(default=180, ge=1)
    backend: str = Field(
        default="auto",
        description=(
            "MIP solver backend: 'gurobi' | 'cbc' | 'scip' | 'auto'. "
            "'auto' tries Gurobi -> SCIP -> CBC in that order."
        ),
    )
    verbose: bool = Field(default=False, description="Show solver output (all backends).")
    output_flag: Optional[int] = Field(
        default=None,
        description="Deprecated. Use verbose: true/false instead.",
    )
    min_samples_per_rule: Optional[float] = Field(
        default=None,
        description="Minimum fraction of samples a rule must cover (Nmin). None = no pruning."
    )

    model_config = {"populate_by_name": True}

    @property
    def is_verbose(self) -> bool:
        if self.output_flag is not None:
            return bool(self.output_flag)
        return self.verbose


# ─────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────

class CVConfig(BaseModel):
    enabled: bool = Field(default=True)
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = Field(default=True)


# ─────────────────────────────────────────────
# MongoDB Storage
# ─────────────────────────────────────────────

class StorageConfig(BaseModel):
    enabled: bool = Field(default=False)
    uri: str = Field(default="mongodb://localhost:27017", description="MongoDB connection string")
    database: str = Field(default="results_db")
    collection: str = Field(default="results_full")
    skip_existing: bool = Field(default=True, description="Skip seeds already in DB")


# ─────────────────────────────────────────────
# Run / Experiment
# ─────────────────────────────────────────────

class RunConfig(BaseModel):
    experiment_name: str = Field(default="experiment", description="Used in output file naming")
    start_seed: int = Field(default=42)
    num_seeds: int = Field(default=1, ge=1)
    output_dir: Path = Field(default=Path("outputs"))
    log_level: str = Field(default="INFO", description="Python logging level")
    suppress_solver_output: bool = Field(default=True)

    @property
    def seeds(self) -> List[int]:
        return list(range(self.start_seed, self.start_seed + self.num_seeds))


# ─────────────────────────────────────────────
# Root config (composes all sections)
# ─────────────────────────────────────────────

class PipelineConfig(BaseModel):
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    forest: ForestConfig = Field(default_factory=ForestConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def sync_depth(self) -> "PipelineConfig":
        """Ensure forest.max_depth is consistently accessible from solver context."""
        return self
