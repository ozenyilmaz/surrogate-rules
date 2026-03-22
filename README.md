# surrogate-rules

MIP-based surrogate rule extraction from Random Forests.

A research pipeline that trains a Random Forest, extracts decision paths,
solves a Mixed Integer Program (Gurobi) to select an optimal rule subset,
and evaluates the surrogate model with a rich set of KPIs.

---

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Place your dataset
cp path/to/mushroom.arff data/

# 3. Run
python scripts/run_experiment.py --config configs/mushroom.yaml

# 4. With MongoDB persistence
MONGO_URI=mongodb://localhost:27017 \
python scripts/run_experiment.py --config configs/mushroom.yaml
```

---

## Project structure

```
surrogate-rules/
├── src/
│   ├── config/          # Pydantic config models + YAML loader
│   ├── data/            # Dataset loaders (extensible registry)
│   ├── features/        # (reserved for preprocessing transforms)
│   ├── optimization/    # Gurobi MIP solver
│   ├── rules/           # RF training, path extraction, surrogate scoring
│   ├── kpi/             # Fidelity, expressiveness, and classification metrics
│   ├── inference/       # (reserved for single-sample prediction/explanation)
│   ├── storage/         # MongoDB repository
│   ├── pipeline/        # Orchestration (seed loop, CV, holdout)
│   └── utils/           # Logging setup
├── scripts/
│   └── run_experiment.py   # CLI entry point
├── configs/
│   └── mushroom.yaml        # Example config
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   └── migration_guide.md
├── outputs/             # Run artifacts (auto-created)
├── data/                # Place .arff files here
├── requirements.txt
├── pyproject.toml
└── Dockerfile
```

---

## Running tests

```bash
pytest tests/unit/ -v          # Fast; no Gurobi or MongoDB needed
pytest tests/integration/ -v   # Gurobi mocked via unittest.mock
pytest --cov=src               # Coverage report
```

---

## Environment variables

| Variable | Purpose |
|---|---|
| `MONGO_URI` | Overrides `storage.uri` in config |
| `GRB_LICENSE_FILE` | Path to Gurobi licence file |

---

## Adding a new dataset

See `docs/migration_guide.md`.

---

## Solver backends

The pipeline supports three MIP solver backends. The `backend: auto` setting
(the default) selects the best available one at runtime.

### Priority order: `auto`

```
Gurobi  →  SCIP  →  CBC
```

### Backend comparison

| Backend | Install | Licence | Speed (large L) | Notes |
|---|---|---|---|---|
| `gurobi` | `pip install gurobipy` | Commercial | Fastest | Best for production |
| `scip` | `pip install pyscipopt` | Free (academic) | Fast | Competitive with Gurobi on research instances |
| `cbc` | `pip install pulp` | Free | Moderate | Bundled in `pulp` — zero-friction default |

### Configuring the backend

```yaml
# configs/mushroom.yaml
solver:
  backend: auto    # or: gurobi | scip | cbc
  verbose: false
  time_limit_seconds: 180
```

### Zero-licence quickstart

```bash
pip install pulp          # CBC bundled — no licence, no registration
python scripts/run_experiment.py --config configs/mushroom.yaml
# → Auto-selects CBC; pipeline runs end-to-end
```

### Performance guidance

For the rule-selection MIP in this pipeline (exact-cover with linear objective):

- **L < 500 rules**: all three backends are fast (< 5 s). No tuning needed.
- **L ~ 2 000**: CBC may take 10–30 s. SCIP typically solves in < 10 s.
- **L > 5 000**: Consider reducing `forest.n_estimators` or setting
  `solver.min_samples_per_rule: 0.01` to prune rare rules before solving.
- **Infeasible problems**: if some training samples are covered by zero rules
  (can happen with very small `n_estimators` or aggressive depth limits),
  the CBC and SCIP backends automatically relax the exact-cover constraint
  to at-least-one-cover for those samples and log a warning.
