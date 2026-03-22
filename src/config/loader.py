"""
src/config/loader.py
--------------------
Loads a YAML config file into a validated PipelineConfig instance.

Supports environment variable overrides for secrets (e.g. MONGO_URI).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.config.models import PipelineConfig


def _apply_env_overrides(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides for sensitive fields.

    Supported overrides:
        MONGO_URI  → storage.uri
    """
    mongo_uri = os.environ.get("MONGO_URI")
    if mongo_uri:
        raw.setdefault("storage", {})["uri"] = mongo_uri
    return raw


def load_config(config_path: str | Path) -> PipelineConfig:
    """
    Load and validate a pipeline configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A fully validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If the config is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    raw = _apply_env_overrides(raw)
    return PipelineConfig(**raw)


def save_config_snapshot(cfg: PipelineConfig, output_dir: Path) -> Path:
    """
    Serialise the resolved config to YAML and save it alongside run outputs.
    Guarantees reproducibility: every output folder has its config frozen next to it.

    Args:
        cfg: The validated PipelineConfig.
        output_dir: Directory to write the snapshot into.

    Returns:
        Path to the written snapshot file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "config_snapshot.yaml"

    # Pydantic v2: model_dump; v1: dict()
    try:
        raw = cfg.model_dump(mode="json")
    except AttributeError:
        raw = cfg.dict()

    with open(snapshot_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)

    return snapshot_path
