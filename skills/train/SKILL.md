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

| Task Type | Start With | Then Try | Advanced |
|-----------|-----------|----------|----------|
| Binary classification | LogisticRegression, RandomForest | XGBoost, LightGBM | StackingClassifier |
| Multi-class | RandomForest, XGBoost | Neural network | StackingClassifier |
| Regression | LinearRegression, Ridge | RandomForest, XGBoost | StackingRegressor |
| Time series | ARIMA, Prophet | LightGBM with lag features | — |

## Stacking Ensemble for Regression (v1.9.0)

When individual models plateau, use `StackingRegressor` to combine diverse base learners:

```python
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

estimators = [
    ("ridge", Ridge()),
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
]
# Optional: add XGBoost/LightGBM if installed
try:
    from xgboost import XGBRegressor
    estimators.append(("xgb", XGBRegressor(n_estimators=100, random_state=42)))
except ImportError:
    pass

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5, n_jobs=-1
)

# Tune the final estimator's alpha via GridSearchCV
param_grid = {"final_estimator__alpha": [0.01, 0.1, 1.0, 10.0]}
grid = GridSearchCV(stacking, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
grid.fit(X_train, y_train)
```

Use stacking when: baseline regression R2 < 0.90 and you have >500 training samples. Skip for very small datasets where the added complexity causes overfitting.

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
