# V1.3.0 Design: Convention-Based MLOps Registry Layer

**Date:** 2026-02-23
**Version:** 1.3.0
**Status:** Approved

## Problem

The current workflow trains and deploys models but doesn't maintain a structured record of what was trained, which features were used, what data produced which model, or why architectural decisions were made. This makes it hard to:
- Compare model versions or roll back to a previous approach
- Reuse features across projects
- Reproduce experiments
- Understand why a particular model family was chosen
- Track the full lineage from raw data to deployed model

## Solution: Convention-Based MLOps Registry Layer

Add four local JSON registries — no external dependencies — following the same multi-platform pattern as the report bus. Enhance the `mlops-engineer` agent to own all registry operations. Add a `/registry` command for inspection.

### Design Principles
- **Convention-based**: Local JSON files, no MLflow/Feast/DVC required
- **Task-type aware**: Schemas adapt to classification, regression, MMM, segmentation, time series
- **Insight-driven**: Architecture decisions are captured with rationale linking to EDA insights and ml-theory-advisor recommendations
- **Multi-platform**: Same directory pattern as report bus

## 1. Storage Convention

```
.claude/mlops/          # Claude Code
.cursor/mlops/          # Cursor
.codex/mlops/           # Codex
.opencode/mlops/        # OpenCode
mlops/                  # Universal fallback

Within each:
+-- model-registry.json       # Catalog of all trained models
+-- feature-store.json        # Catalog of all engineered features
+-- experiments/
|   +-- {experiment-id}.json  # One file per training run
+-- data-versions/
    +-- {fingerprint-short}.json  # One file per dataset version
```

Platform directories constant in ml_utils.py:
```python
PLATFORM_MLOPS_DIRS = [
    ".claude/mlops",
    ".cursor/mlops",
    ".codex/mlops",
    ".opencode/mlops",
    "mlops",
]
```

## 2. Model Registry Schema

```json
{
  "version": "1.0",
  "models": [
    {
      "model_id": "model_20260223_143022",
      "name": "revenue_predictor",
      "task_type": "regression",
      "algorithm": "RandomForestRegressor",
      "framework": "scikit-learn",
      "created_at": "2026-02-23T14:30:22Z",
      "status": "champion",
      "metrics": {
        "rmse": 1245.3,
        "mae": 892.1,
        "r2": 0.87
      },
      "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 12
      },
      "artifact_path": "models/revenue_predictor.joblib",
      "data_fingerprint": "sha256:abc123...",
      "feature_set": ["spend_adstock", "price_lag_7", "seasonality_sin"],
      "training_experiment_id": "exp_20260223_143022",
      "predecessor_id": null,
      "rationale": {
        "eda_insights": "Time series data with strong seasonal patterns and media spend correlations",
        "theory_recommendation": "Random forest recommended for initial baseline; consider Bayesian methods for production MMM",
        "decision_source": "ml-theory-advisor reflection gate approved"
      },
      "tags": ["production", "mmm"]
    }
  ]
}
```

### Task-Type Specific Metrics

| Task Type | Required Metrics | Optional Metrics |
|-----------|-----------------|-----------------|
| classification | accuracy, precision, recall, f1, auc_roc | specificity, mcc, log_loss |
| regression | rmse, mae, r2 | mape, medae |
| mmm | r2, mape, channel_roi, channel_contribution | adstock_params, saturation_params |
| segmentation | silhouette_score, n_clusters | inertia, calinski_harabasz |
| time_series | rmse, mae, mape | smape, mase, forecast_horizon |

### Model Status Lifecycle

```
training -> completed -> challenger -> champion -> archived
                      -> failed
```

## 3. Feature Store Schema

```json
{
  "version": "1.0",
  "features": [
    {
      "feature_id": "spend_adstock_tv",
      "name": "spend_adstock_tv",
      "description": "TV spend with adstock decay transformation (lambda=0.7)",
      "dtype": "float64",
      "source_columns": ["tv_spend"],
      "transformation": "adstock_decay",
      "transformation_params": {"decay_rate": 0.7},
      "created_at": "2026-02-23T14:25:00Z",
      "created_by": "feature-engineering-analyst",
      "domain": "marketing_mix",
      "task_type_relevance": ["mmm", "regression"],
      "tags": ["media", "adstock"],
      "statistics": {
        "mean": 45230.5,
        "std": 12340.2,
        "min": 0.0,
        "max": 98450.0,
        "null_pct": 0.0
      },
      "used_in_models": ["model_20260223_143022"],
      "leakage_risk": "none"
    }
  ]
}
```

## 4. Experiment Tracking Schema

Each training run produces `experiments/{experiment-id}.json`:

```json
{
  "experiment_id": "exp_20260223_143022",
  "name": "revenue_model_rf_v2",
  "created_at": "2026-02-23T14:30:22Z",
  "status": "completed",
  "task_type": "regression",
  "rationale": {
    "approach_reason": "EDA showed linear relationships with media spend; tree-based model for non-linear interactions",
    "feature_selection_reason": "Feature-engineering-analyst identified adstock transformations as critical for MMM",
    "theory_advisor_verdict": "approved — appropriate model family for initial exploration"
  },
  "dataset": {
    "fingerprint": "sha256:abc123...",
    "rows": 5200,
    "features_used": 12,
    "target": "Revenue",
    "split": {"train": 0.7, "val": 0.15, "test": 0.15}
  },
  "model": {
    "algorithm": "RandomForestRegressor",
    "framework": "scikit-learn",
    "hyperparameters": {
      "n_estimators": 200,
      "max_depth": 12,
      "min_samples_split": 5
    }
  },
  "metrics": {
    "train": {"rmse": 980.2, "r2": 0.92},
    "val": {"rmse": 1180.5, "r2": 0.88},
    "test": {"rmse": 1245.3, "r2": 0.87}
  },
  "artifacts": [
    {"type": "model", "path": "models/revenue_predictor.joblib"},
    {"type": "preprocessor", "path": "models/preprocessor.joblib"},
    {"type": "feature_importance", "path": "reports/feature_importance.png"}
  ],
  "notes": "Increased n_estimators from 100 to 200, improved R2 by 0.03",
  "registered_model_id": "model_20260223_143022"
}
```

## 5. Data Versioning Schema

Each dataset version produces `data-versions/{fingerprint-short}.json`:

```json
{
  "fingerprint": "sha256:abc123def456...",
  "created_at": "2026-02-23T14:20:00Z",
  "source_path": "data/sales.csv",
  "rows": 5200,
  "columns": 24,
  "column_schema": {
    "Revenue": {"dtype": "float64", "null_pct": 0.0},
    "tv_spend": {"dtype": "float64", "null_pct": 0.02},
    "date": {"dtype": "datetime64", "null_pct": 0.0}
  },
  "statistics_hash": "sha256:789ghi...",
  "detected_task_type": "regression",
  "used_in_experiments": ["exp_20260223_143022"]
}
```

Fingerprint computation: SHA-256 of sorted column names + dtypes + row count + sample of first/last 100 rows.

## 6. /registry Command

New slash command for inspecting all MLOps artifacts:

```bash
/registry                      # Show summary of all registries
/registry models               # List all models with status
/registry models --champion    # Show current champion model
/registry features             # List all registered features
/registry features --domain mmm  # Filter features by domain
/registry experiments          # List all experiments
/registry data                 # List all data versions
/registry lineage model_id     # Show full lineage for a model
```

## 7. Agent Responsibilities

| Agent | MLOps Action |
|-------|-------------|
| eda-analyst | Writes data fingerprint via `save_data_version()` |
| feature-engineering-analyst | Registers features via `save_feature_entries()` |
| developer | Logs experiment via `save_experiment()` during training |
| mlops-engineer | Registers model via `save_model_entry()`, manages champion/challenger, validates completeness, cross-references all registries |
| ml-theory-advisor | Provides rationale that gets embedded in experiment + model entries |

## 8. Workflow Integration (team-coldstart)

```
Stage 1: Initialize
Stage 2a: EDA -> eda-analyst writes data fingerprint
Stage 2b: Post-EDA parallel -> feature-eng registers features
Stage 2c: Gate 1 (reflection) -> rationale captured
Stage 3: Preprocessing
Stage 3b: Gate 2 (reflection) -> rationale captured
Stage 4: Training -> developer logs experiment
Stage 4b: Gate 3 (reflection) -> rationale captured
Stage 5: Evaluation -> mlops-engineer registers model, sets champion/challenger
Stage 5b: Post-training review (existing)
NEW Stage 5c: MLOps Registry Validation
  -> mlops-engineer validates: model registered, features cataloged,
     experiment logged, data fingerprinted, lineage complete
... remaining stages ...
```

## 9. Files to Modify

| File | Change |
|------|--------|
| `templates/ml_utils.py` | Add Section 9: MLOps Registry helpers |
| `agents/mlops-engineer.md` | Add "MLOps Registry Management" section |
| `agents/eda-analyst.md` | Add data fingerprint instruction |
| `agents/feature-engineering-analyst.md` | Add feature registration instruction |
| `agents/developer.md` | Add experiment logging instruction |
| `commands/team-coldstart.md` | Add MLOps checkpoints + Stage 5c |
| `commands/team-analyze.md` | Add data fingerprint step |
| `commands/registry.md` | New /registry command |
| `skills/registry/SKILL.md` | New registry skill |
| `skills/team-coldstart/SKILL.md` | Document MLOps layer |
| `skills/team-analyze/SKILL.md` | Document data fingerprint |
| `.claude-plugin/plugin.json` | Bump to v1.3.0 |
| `.cursor-plugin/plugin.json` | Bump to v1.3.0 |
| `README.md` | Document MLOps layer |

No new agents, no external dependencies, no new hooks needed.

## 10. Backward Compatibility

- MLOps registries are additive — existing workflows still work without them
- Agents that don't produce MLOps artifacts just skip the registry step
- /registry command returns empty state if no artifacts exist yet
- All new ml_utils functions follow the same pattern as existing save_agent_report/load_agent_reports
