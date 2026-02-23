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
