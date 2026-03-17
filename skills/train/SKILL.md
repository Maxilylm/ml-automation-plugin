---
name: train
description: Train a machine learning model with proper data splitting, preprocessing, and validation. Follows best practices to avoid data leakage and overfitting.
---

# Model Training

## When to Use
- Training ML models on prepared datasets
- Model selection and hyperparameter tuning
- Cross-validation and evaluation

## Key Principles

- Split data BEFORE preprocessing
- Use pipelines to prevent data leakage
- Cross-validate on training set only
- Evaluate ONCE on held-out test set
- Set `random_state` for reproducibility

## Workflow

1. **Data preparation** — Load preprocessed data, verify train/test split exists
2. **Build pipeline** — Combine preprocessor + model in sklearn Pipeline
3. **Baseline model** — Fit with defaults, get CV score as benchmark
4. **Hyperparameter tuning** — `GridSearchCV` or `RandomizedSearchCV` on train only
5. **Final evaluation** — Score ONCE on held-out test set
6. **Report** — Save metrics, model artifact, and experiment to registry

## Task-Type Model Selection

| Task Type | Start With | Then Try |
|-----------|-----------|----------|
| Binary classification | LogisticRegression, RandomForest | XGBoost, LightGBM |
| Multi-class | RandomForest, XGBoost | Neural network |
| Regression | LinearRegression, Ridge | RandomForest, XGBoost |
| Time series | ARIMA, Prophet | LightGBM with lag features |

## Report Bus Integration (v1.2.0)

Reads preprocessing and feature engineering reports. Writes training report:
```python
from ml_utils import save_agent_report
save_agent_report("trainer", {
    "status": "completed",
    "findings": {"best_model": "RandomForest", "cv_f1": 0.84, "test_f1": 0.81},
    "recommendations": [{"text": "Evaluate on full test set", "target_agent": "evaluator"}],
    "artifacts": ["models/model_v1.joblib"]
})
```

## Experiment Tracking (v1.3.0)

Log each training run with `save_experiment()`:
```python
from ml_utils import save_experiment
save_experiment("rf_baseline", {"model": "RandomForest", "cv_f1": 0.84, "params": {...}})
```

## Reflection Gate (v1.2.1)

After training, ml-theory-advisor validates model family choice, hyperparameter ranges, and validation methodology before evaluation proceeds.

## Full Specification

See `commands/train.md` for complete training workflows and model-specific templates.
