# MLOps Registry Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a convention-based MLOps registry layer (model registry, feature store, experiment tracking, data versioning) to the ml-automation-plugin with a `/registry` command for inspection.

**Architecture:** Extend ml_utils.py with Section 9 (MLOps Registry) providing save/load helpers for four registries. Enhance mlops-engineer agent to own all registry operations. Add lightweight instructions to eda-analyst, feature-engineering-analyst, and developer agents for their respective contributions. Insert MLOps checkpoints into team-coldstart and team-analyze workflows. Add /registry command.

**Tech Stack:** Markdown agent definitions, JSON registries, Python utilities

---

### Task 1: Add MLOps Registry helpers to ml_utils.py

**Files:**
- Modify: `templates/ml_utils.py:558` (append after `load_reflection_report`)

**Step 1: Add the PLATFORM_MLOPS_DIRS constant and all registry functions at end of file**

```python

# =============================================================================
# 9. MLOPS REGISTRY
# =============================================================================

PLATFORM_MLOPS_DIRS = [
    ".claude/mlops",
    ".cursor/mlops",
    ".codex/mlops",
    ".opencode/mlops",
    "mlops",
]

MLOPS_SCHEMA_VERSION = "1.0"

# Task-type specific required metrics
TASK_TYPE_METRICS = {
    "classification": ["accuracy", "precision", "recall", "f1", "auc_roc"],
    "regression": ["rmse", "mae", "r2"],
    "mmm": ["r2", "mape", "channel_roi", "channel_contribution"],
    "segmentation": ["silhouette_score", "n_clusters"],
    "time_series": ["rmse", "mae", "mape"],
}


def _load_registry(filename, search_dirs=None):
    """Load a registry JSON file, returning the most recent version found."""
    if search_dirs is None:
        search_dirs = PLATFORM_MLOPS_DIRS

    latest = None
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                if latest is None:
                    latest = data
            except (json.JSONDecodeError, KeyError):
                continue
    return latest


def _save_registry(filename, data, output_dirs=None):
    """Save a registry JSON file to all platform directories."""
    if output_dirs is None:
        output_dirs = PLATFORM_MLOPS_DIRS

    paths_written = []
    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        paths_written.append(path)
    return paths_written


# --- Model Registry ---

def save_model_entry(model_entry, output_dirs=None):
    """
    Register a trained model in the model registry.

    Args:
        model_entry: Dict with keys: model_id, name, task_type, algorithm,
            framework, metrics, hyperparameters, artifact_path,
            data_fingerprint, feature_set, training_experiment_id,
            rationale, tags, status (default: 'challenger')
    """
    from datetime import datetime, timezone

    registry = _load_registry("model-registry.json", output_dirs) or {
        "version": MLOPS_SCHEMA_VERSION,
        "models": [],
    }

    entry = {
        "model_id": model_entry.get("model_id", f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"),
        "name": model_entry.get("name", "unnamed_model"),
        "task_type": model_entry.get("task_type", "regression"),
        "algorithm": model_entry.get("algorithm", ""),
        "framework": model_entry.get("framework", "scikit-learn"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": model_entry.get("status", "challenger"),
        "metrics": model_entry.get("metrics", {}),
        "hyperparameters": model_entry.get("hyperparameters", {}),
        "artifact_path": model_entry.get("artifact_path", ""),
        "data_fingerprint": model_entry.get("data_fingerprint", ""),
        "feature_set": model_entry.get("feature_set", []),
        "training_experiment_id": model_entry.get("training_experiment_id", ""),
        "predecessor_id": model_entry.get("predecessor_id"),
        "rationale": model_entry.get("rationale", {}),
        "tags": model_entry.get("tags", []),
    }

    registry["models"].append(entry)
    return _save_registry("model-registry.json", registry, output_dirs)


def load_model_registry(search_dirs=None):
    """Load the model registry. Returns dict with 'models' list or None."""
    return _load_registry("model-registry.json", search_dirs)


def get_champion_model(search_dirs=None):
    """Get the current champion model, or None."""
    registry = load_model_registry(search_dirs)
    if not registry:
        return None
    for model in registry.get("models", []):
        if model.get("status") == "champion":
            return model
    return None


def promote_model(model_id, search_dirs=None, output_dirs=None):
    """Promote a model to champion, archiving the current champion."""
    registry = _load_registry("model-registry.json", search_dirs) or {"version": MLOPS_SCHEMA_VERSION, "models": []}

    for model in registry["models"]:
        if model["status"] == "champion":
            model["status"] = "archived"
        if model["model_id"] == model_id:
            model["status"] = "champion"

    return _save_registry("model-registry.json", registry, output_dirs)


# --- Feature Store ---

def save_feature_entries(features, output_dirs=None):
    """
    Register features in the feature store.

    Args:
        features: List of dicts, each with keys: feature_id, name, description,
            dtype, source_columns, transformation, transformation_params,
            created_by, domain, task_type_relevance, tags, statistics, leakage_risk
    """
    from datetime import datetime, timezone

    store = _load_registry("feature-store.json", output_dirs) or {
        "version": MLOPS_SCHEMA_VERSION,
        "features": [],
    }

    existing_ids = {f["feature_id"] for f in store["features"]}
    now = datetime.now(timezone.utc).isoformat()

    for feat in features:
        fid = feat.get("feature_id", feat.get("name", "unknown"))
        if fid in existing_ids:
            # Update existing feature
            for i, existing in enumerate(store["features"]):
                if existing["feature_id"] == fid:
                    store["features"][i].update(feat)
                    store["features"][i]["updated_at"] = now
                    break
        else:
            entry = {
                "feature_id": fid,
                "name": feat.get("name", fid),
                "description": feat.get("description", ""),
                "dtype": feat.get("dtype", "unknown"),
                "source_columns": feat.get("source_columns", []),
                "transformation": feat.get("transformation", "raw"),
                "transformation_params": feat.get("transformation_params", {}),
                "created_at": now,
                "created_by": feat.get("created_by", "feature-engineering-analyst"),
                "domain": feat.get("domain", "general"),
                "task_type_relevance": feat.get("task_type_relevance", []),
                "tags": feat.get("tags", []),
                "statistics": feat.get("statistics", {}),
                "used_in_models": feat.get("used_in_models", []),
                "leakage_risk": feat.get("leakage_risk", "unknown"),
            }
            store["features"].append(entry)

    return _save_registry("feature-store.json", store, output_dirs)


def load_feature_store(search_dirs=None):
    """Load the feature store. Returns dict with 'features' list or None."""
    return _load_registry("feature-store.json", search_dirs)


# --- Experiment Tracking ---

def save_experiment(experiment_data, output_dirs=None):
    """
    Save an experiment log for a training run.

    Args:
        experiment_data: Dict with keys: experiment_id, name, task_type,
            rationale, dataset, model, metrics, artifacts, notes
    """
    from datetime import datetime, timezone

    if output_dirs is None:
        output_dirs = PLATFORM_MLOPS_DIRS

    now = datetime.now(timezone.utc).isoformat()
    exp_id = experiment_data.get("experiment_id", f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

    experiment = {
        "experiment_id": exp_id,
        "name": experiment_data.get("name", "unnamed_experiment"),
        "created_at": now,
        "status": experiment_data.get("status", "completed"),
        "task_type": experiment_data.get("task_type", "regression"),
        "rationale": experiment_data.get("rationale", {}),
        "dataset": experiment_data.get("dataset", {}),
        "model": experiment_data.get("model", {}),
        "metrics": experiment_data.get("metrics", {}),
        "artifacts": experiment_data.get("artifacts", []),
        "notes": experiment_data.get("notes", ""),
        "registered_model_id": experiment_data.get("registered_model_id", ""),
    }

    filename = f"experiments/{exp_id}.json"
    paths_written = []
    for d in output_dirs:
        exp_dir = os.path.join(d, "experiments")
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            json.dump(experiment, f, indent=2, default=str)
        paths_written.append(path)
    return paths_written


def load_experiments(search_dirs=None):
    """Load all experiment logs. Returns list of experiment dicts."""
    import glob as globmod

    if search_dirs is None:
        search_dirs = PLATFORM_MLOPS_DIRS

    experiments = {}
    for d in search_dirs:
        pattern = os.path.join(d, "experiments", "*.json")
        for filepath in globmod.glob(pattern):
            try:
                with open(filepath) as f:
                    exp = json.load(f)
                exp_id = exp.get("experiment_id", os.path.basename(filepath))
                if exp_id not in experiments:
                    experiments[exp_id] = exp
            except (json.JSONDecodeError, KeyError):
                continue
    return list(experiments.values())


# --- Data Versioning ---

def save_data_version(data_version, output_dirs=None):
    """
    Save a data version fingerprint.

    Args:
        data_version: Dict with keys: fingerprint, source_path, rows, columns,
            column_schema, statistics_hash, detected_task_type
    """
    from datetime import datetime, timezone

    if output_dirs is None:
        output_dirs = PLATFORM_MLOPS_DIRS

    now = datetime.now(timezone.utc).isoformat()
    fingerprint = data_version.get("fingerprint", "unknown")
    short_fp = fingerprint.replace("sha256:", "")[:12]

    version = {
        "fingerprint": fingerprint,
        "created_at": now,
        "source_path": data_version.get("source_path", ""),
        "rows": data_version.get("rows", 0),
        "columns": data_version.get("columns", 0),
        "column_schema": data_version.get("column_schema", {}),
        "statistics_hash": data_version.get("statistics_hash", ""),
        "detected_task_type": data_version.get("detected_task_type", "unknown"),
        "used_in_experiments": data_version.get("used_in_experiments", []),
    }

    filename = f"data-versions/{short_fp}.json"
    paths_written = []
    for d in output_dirs:
        dv_dir = os.path.join(d, "data-versions")
        os.makedirs(dv_dir, exist_ok=True)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            json.dump(version, f, indent=2, default=str)
        paths_written.append(path)
    return paths_written


def compute_data_fingerprint(df):
    """
    Compute a SHA-256 fingerprint for a DataFrame.

    Uses sorted column names, dtypes, row count, and a sample hash.
    """
    import hashlib

    parts = []
    parts.append(str(sorted(df.columns.tolist())))
    parts.append(str([str(df[c].dtype) for c in sorted(df.columns)]))
    parts.append(str(len(df)))

    # Sample first and last rows for content identity
    sample_size = min(100, len(df))
    if len(df) > 0:
        head = df.head(sample_size).to_csv(index=False)
        tail = df.tail(sample_size).to_csv(index=False)
        parts.append(head)
        parts.append(tail)

    content = "|".join(parts)
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


def load_data_versions(search_dirs=None):
    """Load all data version fingerprints. Returns list of version dicts."""
    import glob as globmod

    if search_dirs is None:
        search_dirs = PLATFORM_MLOPS_DIRS

    versions = {}
    for d in search_dirs:
        pattern = os.path.join(d, "data-versions", "*.json")
        for filepath in globmod.glob(pattern):
            try:
                with open(filepath) as f:
                    ver = json.load(f)
                fp = ver.get("fingerprint", os.path.basename(filepath))
                if fp not in versions:
                    versions[fp] = ver
            except (json.JSONDecodeError, KeyError):
                continue
    return list(versions.values())
```

**Step 2: Verify syntax**

Run: `python3 -c "exec(open('templates/ml_utils.py').read()); funcs = ['save_model_entry','load_model_registry','get_champion_model','promote_model','save_feature_entries','load_feature_store','save_experiment','load_experiments','save_data_version','compute_data_fingerprint','load_data_versions']; print('OK:', all(f in dir() for f in funcs))"`
Expected: `OK: True`

**Step 3: Commit**

```bash
git add templates/ml_utils.py
git commit -m "feat: add MLOps registry helpers to ml_utils.py (model registry, feature store, experiments, data versioning)"
```

---

### Task 2: Add MLOps Registry Management section to mlops-engineer agent

**Files:**
- Modify: `agents/mlops-engineer.md:292` (insert before "## Agent Report Bus (v1.2.0)" at line 294)

**Step 1: Insert MLOps Registry Management section before line 294**

Add this content between the existing paragraph ending at line 292 and the "## Agent Report Bus" section at line 294:

```markdown

## MLOps Registry Management (v1.3.0)

You own the ML lifecycle registries. After training and evaluation, you ensure all artifacts are cataloged with full lineage.

### Model Registry

After each training run, register the model:

```python
from ml_utils import save_model_entry, promote_model, get_champion_model

# Register new model as challenger
save_model_entry({
    "model_id": "model_20260223_143022",
    "name": "revenue_predictor",
    "task_type": "regression",  # classification | regression | mmm | segmentation | time_series
    "algorithm": "RandomForestRegressor",
    "framework": "scikit-learn",
    "metrics": {"rmse": 1245.3, "mae": 892.1, "r2": 0.87},
    "hyperparameters": {"n_estimators": 200, "max_depth": 12},
    "artifact_path": "models/revenue_predictor.joblib",
    "data_fingerprint": "sha256:abc123...",
    "feature_set": ["spend_adstock", "price_lag_7"],
    "training_experiment_id": "exp_20260223_143022",
    "rationale": {
        "eda_insights": "Time series data with seasonal patterns",
        "theory_recommendation": "Tree-based model for non-linear media interactions",
        "decision_source": "ml-theory-advisor reflection gate approved"
    },
    "tags": ["mmm", "production"]
})

# Compare with champion and promote if better
champion = get_champion_model()
if champion is None or new_metrics_better(champion["metrics"], new_metrics):
    promote_model("model_20260223_143022")
```

### Task-Type Awareness

Adapt metrics and validation based on task type:

| Task Type | Required Metrics | Domain Considerations |
|-----------|-----------------|----------------------|
| classification | accuracy, precision, recall, f1, auc_roc | Class imbalance, threshold optimization |
| regression | rmse, mae, r2 | Residual analysis, heteroscedasticity |
| mmm | r2, mape, channel_roi, channel_contribution | Adstock params, saturation curves, interpretability |
| segmentation | silhouette_score, n_clusters | Cluster stability, business interpretability |
| time_series | rmse, mae, mape | Forecast horizon, temporal CV, stationarity |

### Registry Validation (Stage 5c)

After evaluation, validate completeness:
1. Model registered in model-registry.json with correct task_type
2. All features cataloged in feature-store.json
3. Experiment logged in experiments/ with rationale
4. Data fingerprint exists in data-versions/
5. Lineage chain complete: data -> features -> experiment -> model

If any registry is incomplete, log a warning and fill gaps from available reports.

### Rationale Capture

Every model entry MUST include a `rationale` dict explaining:
- **eda_insights**: Key findings from EDA that influenced the approach
- **theory_recommendation**: What ml-theory-advisor recommended and why
- **decision_source**: Which gate/review approved this approach

Read prior reports from `.claude/reports/` to populate rationale fields.

If `ml_utils.py` is not available, write JSON directly to `.claude/mlops/model-registry.json`.
```

**Step 2: Commit**

```bash
git add agents/mlops-engineer.md
git commit -m "feat: add MLOps registry management section to mlops-engineer agent"
```

---

### Task 3: Add data fingerprint instruction to eda-analyst agent

**Files:**
- Modify: `agents/eda-analyst.md:131` (insert before "## Agent Report Bus (v1.2.0)" at line 133)

**Step 1: Insert data fingerprint section before line 133**

Add this content between the existing paragraph ending at line 131 and the "## Agent Report Bus" section at line 133:

```markdown

## Data Versioning (v1.3.0)

After loading the dataset, generate a data fingerprint for reproducibility:

```python
from ml_utils import compute_data_fingerprint, save_data_version

fingerprint = compute_data_fingerprint(df)

save_data_version({
    "fingerprint": fingerprint,
    "source_path": "data/sales.csv",  # actual path used
    "rows": len(df),
    "columns": len(df.columns),
    "column_schema": {
        col: {"dtype": str(df[col].dtype), "null_pct": round(df[col].isnull().mean(), 4)}
        for col in df.columns
    },
    "detected_task_type": "regression",  # based on your EDA findings
})
```

Include the fingerprint in your agent report so downstream agents can reference it.

If `ml_utils.py` is not available, compute a SHA-256 hash of column names + dtypes + row count manually and save to `.claude/mlops/data-versions/`.
```

**Step 2: Commit**

```bash
git add agents/eda-analyst.md
git commit -m "feat: add data versioning instruction to eda-analyst agent"
```

---

### Task 4: Add feature registration instruction to feature-engineering-analyst agent

**Files:**
- Modify: `agents/feature-engineering-analyst.md:8` (insert before "## Agent Report Bus" at line 10)

**Step 1: Insert feature store section before line 10**

Add this content between the existing content at line 8 and the "## Agent Report Bus" section at line 10:

```markdown

## Feature Store Registration (v1.3.0)

After engineering features, register them in the feature store for reuse and lineage tracking:

```python
from ml_utils import save_feature_entries

save_feature_entries([
    {
        "feature_id": "spend_adstock_tv",
        "name": "spend_adstock_tv",
        "description": "TV spend with adstock decay (lambda=0.7)",
        "dtype": "float64",
        "source_columns": ["tv_spend"],
        "transformation": "adstock_decay",
        "transformation_params": {"decay_rate": 0.7},
        "domain": "marketing_mix",
        "task_type_relevance": ["mmm", "regression"],
        "tags": ["media", "adstock"],
        "statistics": {
            "mean": round(df["spend_adstock_tv"].mean(), 2),
            "std": round(df["spend_adstock_tv"].std(), 2),
            "min": round(df["spend_adstock_tv"].min(), 2),
            "max": round(df["spend_adstock_tv"].max(), 2),
            "null_pct": round(df["spend_adstock_tv"].isnull().mean(), 4),
        },
        "leakage_risk": "none",
    },
    # ... one entry per engineered feature
])
```

Register ALL features you create — both raw passthrough and transformed. Use descriptive `transformation` values: `raw`, `log`, `polynomial`, `interaction`, `adstock_decay`, `lag`, `rolling_mean`, `one_hot`, `target_encode`, `binning`, `seasonal_decompose`.

If `ml_utils.py` is not available, write JSON directly to `.claude/mlops/feature-store.json`.
```

**Step 2: Commit**

```bash
git add agents/feature-engineering-analyst.md
git commit -m "feat: add feature store registration to feature-engineering-analyst agent"
```

---

### Task 5: Add experiment logging instruction to developer agent

**Files:**
- Modify: `agents/developer.md:86` (insert before "## Agent Report Bus" at line 88)

**Step 1: Insert experiment tracking section before line 88**

Add this content between the existing content at line 86 and the "## Agent Report Bus" section at line 88:

```markdown

## Experiment Tracking (v1.3.0)

When training a model, log the experiment for reproducibility and comparison:

```python
from ml_utils import save_experiment

save_experiment({
    "experiment_id": "exp_20260223_143022",
    "name": "revenue_model_rf_v2",
    "task_type": "regression",  # classification | regression | mmm | segmentation | time_series
    "rationale": {
        "approach_reason": "EDA showed non-linear media spend relationships",
        "feature_selection_reason": "Adstock features critical for MMM per feature-engineering-analyst",
        "theory_advisor_verdict": "approved"
    },
    "dataset": {
        "fingerprint": "sha256:abc123...",  # from eda-analyst report
        "rows": 5200,
        "features_used": 12,
        "target": "Revenue",
        "split": {"train": 0.7, "val": 0.15, "test": 0.15}
    },
    "model": {
        "algorithm": "RandomForestRegressor",
        "framework": "scikit-learn",
        "hyperparameters": {"n_estimators": 200, "max_depth": 12}
    },
    "metrics": {
        "train": {"rmse": 980.2, "r2": 0.92},
        "val": {"rmse": 1180.5, "r2": 0.88},
        "test": {"rmse": 1245.3, "r2": 0.87}
    },
    "artifacts": [
        {"type": "model", "path": "models/revenue_predictor.joblib"},
        {"type": "preprocessor", "path": "models/preprocessor.joblib"}
    ],
})
```

Read prior agent reports from `.claude/reports/` to populate `rationale` fields — capture WHY this approach was chosen, not just what was trained.

If `ml_utils.py` is not available, write JSON directly to `.claude/mlops/experiments/`.
```

**Step 2: Commit**

```bash
git add agents/developer.md
git commit -m "feat: add experiment tracking instruction to developer agent"
```

---

### Task 6: Create /registry command and skill

**Files:**
- Create: `commands/registry.md`
- Create: `skills/registry/SKILL.md`

**Step 1: Create the registry command**

Create `commands/registry.md`:

```markdown
---
name: registry
description: Inspect MLOps registries — models, features, experiments, and data versions
argument_name: subcommand
argument_description: "What to inspect: models, features, experiments, data, lineage, or empty for summary"
---

# /registry — MLOps Registry Inspector

View and query the convention-based MLOps registries.

## Usage

```bash
/registry                          # Summary of all registries
/registry models                   # List all registered models
/registry models --champion        # Show current champion model
/registry features                 # List all registered features
/registry features --domain mmm    # Filter features by domain
/registry experiments              # List all experiment logs
/registry data                     # List all data version fingerprints
/registry lineage <model_id>       # Show full lineage for a model
```

## Behavior

### Summary (no subcommand)

Read all registries and display:
- Total models (by status: champion, challenger, archived)
- Total features (by domain)
- Total experiments
- Total data versions
- Lineage completeness check

### Models

Read `.claude/mlops/model-registry.json` (and other platform dirs).

Display table:
| Model ID | Name | Task Type | Algorithm | Status | Key Metric | Created |

With `--champion`: Show detailed view of champion model including metrics, hyperparameters, rationale, and feature set.

### Features

Read `.claude/mlops/feature-store.json` (and other platform dirs).

Display table:
| Feature ID | Transformation | Source | Domain | Used In Models | Leakage Risk |

With `--domain <name>`: Filter to features matching that domain.

### Experiments

Read `.claude/mlops/experiments/*.json` (and other platform dirs).

Display table:
| Experiment ID | Task Type | Algorithm | Test Metrics | Status | Created |

### Data

Read `.claude/mlops/data-versions/*.json` (and other platform dirs).

Display table:
| Fingerprint (short) | Source | Rows | Columns | Task Type | Used In |

### Lineage

Given a model_id, trace backwards:
1. Find model in registry -> get training_experiment_id, data_fingerprint, feature_set
2. Load experiment -> get dataset info, rationale
3. Load data version -> get source path, schema
4. Load features -> get transformation details

Display as a lineage chain:
```
Data: sales.csv (sha256:abc1..., 5200 rows)
  -> Features: 12 registered (spend_adstock_tv, price_lag_7, ...)
    -> Experiment: exp_20260223_143022 (RF, R2=0.87)
      -> Model: model_20260223_143022 (champion)
         Rationale: "Tree-based model for non-linear media interactions"
```

## Implementation

Use ml_utils functions:
```python
from ml_utils import (
    load_model_registry, get_champion_model,
    load_feature_store,
    load_experiments,
    load_data_versions,
)
```

If `ml_utils.py` is not available, read JSON files directly from the mlops directories.
```

**Step 2: Create the registry skill**

Create `skills/registry/SKILL.md`:

```markdown
---
name: registry
description: Inspect MLOps registries — models, features, experiments, and data versions
---

# Registry

Inspect the convention-based MLOps registries for this project.

## What It Shows

- **Model Registry**: All trained models with status (champion/challenger/archived), metrics, and lineage
- **Feature Store**: All engineered features with transformations, statistics, and leakage risk
- **Experiment Tracking**: All training runs with hyperparameters, metrics, and rationale
- **Data Versioning**: Dataset fingerprints for reproducibility

## Registry Locations

Registries are stored in platform-specific directories:
- `.claude/mlops/` (Claude Code)
- `.cursor/mlops/` (Cursor)
- `.codex/mlops/` (Codex)
- `.opencode/mlops/` (OpenCode)
- `mlops/` (universal fallback)

## Task-Type Awareness

Metrics and validation adapt to the problem type:
- **classification**: accuracy, precision, recall, f1, auc_roc
- **regression**: rmse, mae, r2
- **mmm**: r2, mape, channel_roi, channel_contribution
- **segmentation**: silhouette_score, n_clusters
- **time_series**: rmse, mae, mape
```

**Step 3: Commit**

```bash
git add commands/registry.md skills/registry/SKILL.md
git commit -m "feat: add /registry command and skill for MLOps inspection"
```

---

### Task 7: Insert MLOps checkpoints into team-coldstart

**Files:**
- Modify: `commands/team-coldstart.md`

**Step 1: Add data fingerprint instruction to Stage 2a (EDA stage)**

Find the Stage 2a EDA section and add after the EDA agent spawn instructions:

```markdown
   After EDA completes, the data fingerprint is automatically generated by eda-analyst
   and stored in `.claude/mlops/data-versions/`. This enables downstream reproducibility.
```

**Step 2: Add feature registration note to Stage 2b (Post-EDA parallel)**

Find the Stage 2b feature-engineering-analyst spawn instructions and add:

```markdown
   After feature engineering completes, all features are registered in `.claude/mlops/feature-store.json`
   by feature-engineering-analyst. This enables feature reuse and lineage tracking.
```

**Step 3: Add experiment logging note to Stage 4 (Training)**

Find the Stage 4 Training section and add:

```markdown
   After training completes, the developer agent logs the experiment to `.claude/mlops/experiments/`
   including rationale from prior agent reports, hyperparameters, and metrics across all splits.
```

**Step 4: Insert Stage 5c after Stage 5b (Post-Training Review)**

Insert between Stage 5b and Stage 6. Find the end of Stage 5b (after the parallel review agents section) and add:

```markdown

### Stage 5c: MLOps Registry Validation

**mlops-engineer** validates that all MLOps registries are complete:

1. **Spawn mlops-engineer for registry validation:**
   ```
   MLOps Registry Validation — verify all registries are complete.

   Read all reports in .claude/reports/ for context.
   Read all registries in .claude/mlops/ (model-registry.json, feature-store.json, experiments/, data-versions/).

   Validate:
   1. Model registered with correct task_type and metrics
   2. All features cataloged with transformations and lineage
   3. Experiment logged with rationale from prior agent reports
   4. Data fingerprint exists
   5. Lineage chain is complete: data -> features -> experiment -> model

   If any registry is incomplete, fill gaps from available reports.
   Compare new model metrics with champion (if exists) and promote if better.

   Write your report using save_agent_report("mlops-engineer", {...})
   ```

2. **Output:**
   ```markdown
   ## Stage 5c: MLOps Registry Validation ✓

   - Model registered: model_20260223_143022 (challenger -> champion)
   - Features cataloged: 12 features in feature-store.json
   - Experiment logged: exp_20260223_143022
   - Data fingerprinted: sha256:abc123...
   - Lineage: complete ✓
   ```
```

**Step 5: Commit**

```bash
git add commands/team-coldstart.md
git commit -m "feat: insert MLOps registry checkpoints into team-coldstart workflow"
```

---

### Task 8: Add data fingerprint step to team-analyze

**Files:**
- Modify: `commands/team-analyze.md`

**Step 1: Add data fingerprint note after the EDA spawn section**

Find the eda-analyst spawn section in team-analyze and add a note:

```markdown
   After EDA completes, eda-analyst generates a data fingerprint stored in `.claude/mlops/data-versions/`.
```

**Step 2: Add feature registration note after the feature-engineering-analyst section**

```markdown
   After feature analysis completes, features are registered in `.claude/mlops/feature-store.json`.
```

**Step 3: Commit**

```bash
git add commands/team-analyze.md
git commit -m "feat: add MLOps data fingerprint and feature store steps to team-analyze"
```

---

### Task 9: Update skill documentation

**Files:**
- Modify: `skills/team-coldstart/SKILL.md:48` (append at end)
- Modify: `skills/team-analyze/SKILL.md:30` (append at end)

**Step 1: Append to team-coldstart SKILL.md**

```markdown

## MLOps Registry Layer (v1.3.0)

Convention-based MLOps registries track the full model lifecycle:

| Registry | Agent | Written At |
|----------|-------|-----------|
| Data Versions | eda-analyst | Stage 2a (EDA) |
| Feature Store | feature-engineering-analyst | Stage 2b (Post-EDA) |
| Experiments | developer | Stage 4 (Training) |
| Model Registry | mlops-engineer | Stage 5c (Validation) |

Stage 5c (MLOps Registry Validation) ensures all registries are complete and lineage is traceable from data to deployed model. Use `/registry` to inspect.
```

**Step 2: Append to team-analyze SKILL.md**

```markdown

## MLOps Registry (v1.3.0)

During analysis, eda-analyst generates a data fingerprint and feature-engineering-analyst registers features. Use `/registry` to inspect stored artifacts.
```

**Step 3: Commit**

```bash
git add skills/team-coldstart/SKILL.md skills/team-analyze/SKILL.md
git commit -m "feat: document MLOps registry layer in skill files"
```

---

### Task 10: Version bump and README update

**Files:**
- Modify: `.claude-plugin/plugin.json`
- Modify: `.cursor-plugin/plugin.json`
- Modify: `README.md`

**Step 1: Bump .claude-plugin/plugin.json version from 1.2.1 to 1.3.0**

Update the `"version"` field to `"1.3.0"` and add "MLOps registry layer" to the description.

**Step 2: Bump .cursor-plugin/plugin.json version from 1.2.1 to 1.3.0**

Same changes.

**Step 3: Add MLOps Registry section to README.md**

After the "What's New in v1.2.1" section, add:

```markdown
## What's New in v1.3.0

### MLOps Registry Layer

Convention-based MLOps registries — no external dependencies required:

- **Model Registry**: Track trained models with metrics, lineage, rationale, and champion/challenger status
- **Feature Store**: Catalog engineered features with transformations, statistics, and reusability metadata
- **Experiment Tracking**: Log every training run with hyperparameters, metrics, and approach rationale
- **Data Versioning**: Fingerprint datasets for reproducibility

Task-type aware — adapts metrics and validation for classification, regression, MMM, segmentation, and time series.

```bash
# Inspect registries
/registry                          # Summary of all registries
/registry models --champion        # Show champion model details
/registry features --domain mmm    # Filter features by domain
/registry lineage model_id         # Trace full lineage
```
```

**Step 4: Add `/registry` to the Skills/Slash Commands table in README**

Find the skills table and add:

```markdown
| `registry` | Inspect MLOps registries (models, features, experiments, data) |
```

**Step 5: Commit**

```bash
git add .claude-plugin/plugin.json .cursor-plugin/plugin.json README.md
git commit -m "feat: bump to v1.3.0, document MLOps registry layer in README"
```

---

### Task 11: Final verification and cleanup

**Step 1: Verify all changes**

```bash
# Version check
grep '"1.3.0"' .claude-plugin/plugin.json .cursor-plugin/plugin.json

# MLOps section in mlops-engineer
grep -c "MLOps Registry" agents/mlops-engineer.md  # Should be >= 1

# Data versioning in eda-analyst
grep -c "Data Versioning" agents/eda-analyst.md  # Should be >= 1

# Feature store in feature-engineering-analyst
grep -c "Feature Store" agents/feature-engineering-analyst.md  # Should be >= 1

# Experiment tracking in developer
grep -c "Experiment Tracking" agents/developer.md  # Should be >= 1

# Registry command exists
test -f commands/registry.md && echo "OK" || echo "MISSING"
test -f skills/registry/SKILL.md && echo "OK" || echo "MISSING"

# New functions in ml_utils
python3 -c "exec(open('templates/ml_utils.py').read()); funcs = ['save_model_entry','load_model_registry','get_champion_model','promote_model','save_feature_entries','load_feature_store','save_experiment','load_experiments','save_data_version','compute_data_fingerprint','load_data_versions']; print('All OK:', all(f in dir() for f in funcs))"
```

**Step 2: Delete plan files**

```bash
rm docs/plans/2026-02-23-mlops-registry-layer-design.md docs/plans/2026-02-23-mlops-registry-layer-implementation.md
rmdir docs/plans docs 2>/dev/null || true
git add -A
git commit -m "chore: remove MLOps registry layer plan files"
```
