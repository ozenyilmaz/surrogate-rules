"""
src/storage/mongo.py
--------------------
MongoDB persistence layer.

Responsibility  : Save and query SeedResult documents.
Architecture    : src/storage/
Migration from  : seed_exists / save_result helpers in run_pipeline_resume.py

Key improvements:
  - MongoRepository class: one connection per experiment, not per call.
  - make_json_safe is a reusable utility here (was inline in the old script).
  - Context manager for safe connection lifecycle.
  - Typed interface; no raw dict manipulation in callers.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Dict

import numpy as np

from src.pipeline.runner import SeedResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# JSON serialisation helper
# ─────────────────────────────────────────────

def make_json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy scalars and arrays to native Python types
    so they can be stored in MongoDB / serialised to JSON.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ─────────────────────────────────────────────
# Repository
# ─────────────────────────────────────────────

class MongoRepository:
    """
    Thin repository for persisting pipeline SeedResults.

    Usage::

        repo = MongoRepository(uri, database, collection)
        with repo:
            if not repo.exists(dataset, seed):
                repo.save(dataset, seed_result)
    """

    def __init__(self, uri: str, database: str, collection: str) -> None:
        self._uri = uri
        self._database = database
        self._collection = collection
        self._client = None
        self._col = None

    def __enter__(self) -> "MongoRepository":
        try:
            from pymongo import MongoClient
            self._client = MongoClient(self._uri, serverSelectionTimeoutMS=5000)
            # Trigger a connection check
            self._client.server_info()
            self._col = self._client[self._database][self._collection]
            logger.debug("MongoDB connected: %s / %s", self._database, self._collection)
        except Exception as exc:
            raise ConnectionError(f"Cannot connect to MongoDB at {self._uri}: {exc}") from exc
        return self

    def __exit__(self, *_) -> None:
        if self._client:
            self._client.close()
            logger.debug("MongoDB connection closed.")

    def exists(self, dataset: str, seed: int) -> bool:
        """Return True if a result for (dataset, seed) already exists."""
        assert self._col is not None, "Call inside 'with' block."
        return self._col.count_documents({"dataset": dataset, "seed": int(seed)}) > 0

    def save(self, dataset: str, result: SeedResult) -> None:
        """
        Persist a SeedResult document.

        The document structure mirrors the SeedResult fields, extended with
        ``dataset`` for cross-dataset queries.
        """
        assert self._col is not None, "Call inside 'with' block."

        doc: Dict[str, Any] = {
            "dataset": dataset,
            "seed": result.seed,
            "alpha": result.alpha,
            "depth": result.depth,
            "lambda": result.lambda_,
            "cv_summary": result.cv_summary,
            "final_test": asdict(result.final_test),
        }

        doc = make_json_safe(doc)
        self._col.insert_one(doc)
        logger.debug("Saved result: dataset=%s seed=%d", dataset, result.seed)
