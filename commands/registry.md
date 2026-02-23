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
