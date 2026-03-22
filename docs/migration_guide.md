# Migration Guide: Script-based Prototype → `surrogate-rules` Package

## What moved where

| Original file | New location | Notes |
|---|---|---|
| `dataset.py` | `src/data/loaders.py` | Loader registry replaces if/elif chain. `register_loader()` for new datasets. |
| `dataset_mushroom.py` | deleted | Merged into `_load_mushroom()` inside registry. |
| `warm_start.py` | `src/rules/forest.py` | `computePaths` → `compute_forest_paths`. `computeLoss` → `compute_rule_stats`. `computeSample` → `sample_satisfies_path` (iterative). |
| `mip.py` | `src/optimization/solver.py` | `generateProblemSoft` → `solve_rule_selection`. Returns `SolverResult` dataclass. |
| `fidelitymeasure.py` | `src/kpi/metrics.py` | All fidelity + expressiveness metrics unified here. |
| `generateproblem.py` | `src/pipeline/runner.py` | `computeAll` split into `_run_fold` + `run_seed`. |
| `run_pipeline_resume.py` | `src/pipeline/runner.py` + `scripts/run_experiment.py` | Loop logic in script; seed/CV logic in runner. |
| `main.py` | `scripts/run_experiment.py` | Only the `computeResults_fixed_alpha` single-run mode. |

---

## Step-by-step migration

### 1 Install the package in editable mode

```bash
pip install -e ".[dev]"
```

### 2 Place datasets

Copy your `.arff` files into `data/`:

```
data/
  mushroom.arff
  banknote.arff
  ...
```

### 3 Create a config

Copy `configs/mushroom.yaml` and adjust `dataset.path`, `solver.alpha`, seeds, etc.

### 4 Run

```bash
python scripts/run_experiment.py --config configs/mushroom.yaml
```

With MongoDB:

```bash
MONGO_URI=mongodb://localhost:27017 \
python scripts/run_experiment.py --config configs/mushroom.yaml
```

Dry-run (validate config only):

```bash
python scripts/run_experiment.py --config configs/mushroom.yaml --dry-run
```

### 5 Run tests

```bash
pytest tests/unit/ -v                      # no Gurobi needed
pytest tests/integration/ -v              # Gurobi mocked
```

---

## Adding a new dataset

1. Write a loader function in `src/data/loaders.py`:

```python
def _load_my_dataset(df: pd.DataFrame) -> XYPair:
    ...
    return X, y
```

2. Register it:

```python
LOADER_REGISTRY["my_dataset"] = _load_my_dataset
```

3. Add `data/my_dataset.arff` and a config YAML.  Nothing else changes.

---

## Known issues fixed during migration

| Issue | Fix |
|---|---|
| `seed_exists` loop was unreachable (indented inside `seed_exists` function body) | Loop is now correctly in `scripts/run_experiment.py` |
| `computePaths` printed directly; could not be silenced without `redirect_stdout` | Now uses `logging.debug` |
| `generateProblemSoft` solved inside itself — model/z returned but already optimised | Solver returns `SolverResult`; caller does not re-optimise |
| `load_arff_dataset` in `run_pipeline_resume.py` expected 5-tuple but `dataset.py` returns 4-tuple | Unified: `load_dataset()` always returns `(X, y)` only; depth/lambda come from config |
| `dataset_mushroom.py` was a duplicate of `_load_mushroom` in `dataset.py` | Deleted; single implementation |
| Recursion in `computeSample` (could hit Python limit on deep trees) | Replaced with iterative loop in `sample_satisfies_path` |

---

## Migrating to the solver-agnostic optimization layer

### What changed

The optimization layer was refactored from a single `gurobipy`-only
`solver.py` into a layered backend system.

**Old call (still works unchanged):**
```python
from src.optimization.solver import solve_rule_selection
result = solve_rule_selection(n, A, loss, freq)
```

**New call with explicit backend:**
```python
result = solve_rule_selection(n, A, loss, freq, backend="cbc")
```

**Config change — old `output_flag` (deprecated, still accepted):**
```yaml
solver:
  output_flag: 0   # old
```

**Config change — new:**
```yaml
solver:
  backend: auto    # new — gurobi | cbc | scip | auto
  verbose: false   # new — replaces output_flag
```

### Files added

```
src/optimization/
├── registry.py                    ← get_backend() factory + auto-selection
└── backends/
    ├── base.py                    ← MIPProblem dataclass + AbstractSolverBackend
    ├── gurobi_backend.py          ← all gurobipy code isolated here
    ├── cbc_backend.py             ← PuLP/CBC
    └── scip_backend.py            ← pyscipopt/SCIP
```

### Mathematical equivalence

The MIP formulation is **identical** across all backends:

```
maximise   Σ_j  freq[j]·z[j]  −  (1−λ)·Σ_j  loss[j]·z[j]  −  α·Σ_j  z[j]
subject to Σ_j  A[i,j]·z[j] = 1   ∀ i
           z[j] ∈ {0, 1}
```

Verified numerically: CBC and SCIP produce the same `selected_indices` as
Gurobi on all tested instances.
