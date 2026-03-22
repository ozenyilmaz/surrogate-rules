"""
src/optimization/registry.py
-----------------------------
Central registry of all solver backends.

Responsibilities:
    - Map config strings ('gurobi', 'cbc', 'scip', 'auto') to backend classes.
    - Implement the 'auto' fallback chain: Gurobi → SCIP → CBC.
    - Provide get_backend() as the single factory function consumed by the pipeline.

Adding a new backend:
    1. Create src/optimization/backends/my_backend.py with MyBackend(AbstractSolverBackend).
    2. Import it here and add it to BACKEND_REGISTRY.
    3. Optionally insert it into AUTO_FALLBACK_ORDER.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Type

from src.optimization.backends.base import AbstractSolverBackend
from src.optimization.backends.cbc_backend import PuLPBackend
from src.optimization.backends.gurobi_backend import GurobiBackend
from src.optimization.backends.scip_backend import SCIPBackend

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────

BACKEND_REGISTRY: Dict[str, Type[AbstractSolverBackend]] = {
    "gurobi": GurobiBackend,
    "cbc":    PuLPBackend,
    "scip":   SCIPBackend,
}

# 'auto' tries each backend in this order, picks the first available one.
AUTO_FALLBACK_ORDER: List[str] = ["gurobi", "scip", "cbc"]


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def get_backend(name: str) -> AbstractSolverBackend:
    """
    Instantiate and return the requested solver backend.

    Args:
        name: One of 'gurobi', 'cbc', 'scip', or 'auto'.
              'auto' walks AUTO_FALLBACK_ORDER and returns the first
              available backend.

    Returns:
        An instantiated AbstractSolverBackend subclass.

    Raises:
        ValueError: If the name is unknown.
        RuntimeError: If name='auto' and no backend is available.
        RuntimeError: If a named backend is requested but unavailable.
    """
    name = name.lower().strip()

    if name == "auto":
        return _auto_select()

    if name not in BACKEND_REGISTRY:
        available = list(BACKEND_REGISTRY.keys()) + ["auto"]
        raise ValueError(
            f"Unknown solver backend '{name}'. "
            f"Available options: {available}"
        )

    backend_cls = BACKEND_REGISTRY[name]
    if not backend_cls.is_available():
        raise RuntimeError(
            f"Solver backend '{name}' is not available in this environment. "
            f"Check installation and licence.  "
            f"Use backend: auto in your config to enable fallback."
        )

    logger.info("Using solver backend: %s", name)
    return backend_cls()


def _auto_select() -> AbstractSolverBackend:
    """Walk the fallback chain and return the first available backend."""
    for name in AUTO_FALLBACK_ORDER:
        cls = BACKEND_REGISTRY.get(name)
        if cls and cls.is_available():
            logger.info("Auto-selected solver backend: %s", name)
            return cls()

    raise RuntimeError(
        "No MIP solver backend is available. "
        "Install at least one of: gurobipy, pyscipopt, pulp (CBC). "
        "  pip install pulp           # free, no licence needed\n"
        "  pip install pyscipopt      # free, academic licence\n"
        "  pip install gurobipy       # commercial, requires licence"
    )


def list_available_backends() -> List[str]:
    """Return names of all backends that are currently available."""
    return [name for name, cls in BACKEND_REGISTRY.items() if cls.is_available()]
