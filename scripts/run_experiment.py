"""
scripts/run_experiment.py
--------------------------
Thin CLI entry point.  All logic lives in src/.

Usage examples:
    python scripts/run_experiment.py --config configs/mushroom.yaml
    python scripts/run_experiment.py --config configs/mushroom.yaml --seeds 42 43 44
    MONGO_URI=mongodb://localhost:27017 python scripts/run_experiment.py --config configs/mushroom.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.loader import load_config, save_config_snapshot
from src.pipeline.runner import run_seed
from src.storage.mongo import MongoRepository
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the surrogate-rules experiment pipeline.")
    p.add_argument("--config", required=True, type=Path, help="Path to YAML config file.")
    p.add_argument(
        "--seeds", nargs="*", type=int, default=None,
        help="Override seeds from config. E.g. --seeds 42 43 44"
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and print plan without running."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load + validate config
    cfg = load_config(args.config)

    # Override seeds if provided
    seeds = args.seeds if args.seeds else cfg.run.seeds

    # Setup logging
    log_file = Path(cfg.run.output_dir) / "run.log"
    setup_logging(level=cfg.run.log_level, log_file=log_file)

    logger.info("Experiment   : %s", cfg.run.experiment_name)
    logger.info("Dataset      : %s (%s)", cfg.dataset.name, cfg.dataset.path)
    logger.info("Seeds        : %s", seeds)
    logger.info("Output dir   : %s", cfg.run.output_dir)

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    # Save config snapshot for reproducibility
    out_dir = Path(cfg.run.output_dir)
    snapshot = save_config_snapshot(cfg, out_dir)
    logger.info("Config snapshot saved: %s", snapshot)

    # Optionally open MongoDB connection
    repo = None
    if cfg.storage.enabled:
        repo = MongoRepository(cfg.storage.uri, cfg.storage.database, cfg.storage.collection)

    results = []

    def _run(seed: int) -> None:
        result = run_seed(cfg, seed)
        if result is None:
            logger.warning("Seed %d produced no result.", seed)
            return
        results.append(result)
        logger.info(
            "Seed %d done | surrogate_acc=%.4f | rf_acc=%.4f | rules=%d",
            seed,
            result.final_test.surrogate_acc,
            result.final_test.rf_acc,
            result.final_test.rule_count,
        )

    if repo is not None:
        with repo:
            for seed in seeds:
                if cfg.storage.skip_existing and repo.exists(cfg.dataset.name, seed):
                    logger.info("Seed %d already in DB — skipping.", seed)
                    continue
                _run(seed)
                if results:
                    repo.save(cfg.dataset.name, results[-1])
                    logger.info("Saved to MongoDB.")
    else:
        for seed in seeds:
            _run(seed)

    # Save CSV summary
    if results:
        import pandas as pd
        rows = [
            {"seed": r.seed, **r.final_test.to_dict()}
            for r in results
        ]
        csv_path = out_dir / f"{cfg.run.experiment_name}_results.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False, sep=";")
        logger.info("Results CSV: %s", csv_path)

    logger.info("Done. %d seeds completed.", len(results))


if __name__ == "__main__":
    main()
