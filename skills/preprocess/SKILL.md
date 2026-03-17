---
name: preprocess
description: Data processing pipeline creation for any tabular data. Handles missing values, encoding, scaling, and transformations. For ML tasks, ensures no data leakage.
---

# Data Preprocessing

## When to Use
- Preparing raw data for modeling or analysis
- Handling missing values, encoding categoricals, scaling numerics
- Building reproducible preprocessing pipelines

## Key Principles

- **Split before transform** — Always split data before fitting any preprocessor
- **Pipeline everything** — Use sklearn Pipeline to prevent leakage
- **Fit on train only** — Never fit transformers on test data

## Workflow

1. **Load and validate data** — Check schema, dtypes, shape
2. **Split into train/test FIRST** — `train_test_split(stratify=target)` before any transforms
3. **Build ColumnTransformer** — Numeric: impute + scale. Categorical: impute + encode
4. **Fit on training data only** — `pipeline.fit(X_train, y_train)`
5. **Transform both sets** — `X_train_t = pipeline.transform(X_train)`, same for test
6. **Save pipeline** — `joblib.dump(pipeline, "models/preprocessing_pipeline.joblib")`

## Common Transformers

| Column Type | Imputer | Transformer |
|-------------|---------|-------------|
| Numeric | `SimpleImputer(strategy="median")` | `StandardScaler()` or `RobustScaler()` |
| Categorical | `SimpleImputer(strategy="most_frequent")` | `OneHotEncoder(handle_unknown="ignore")` |
| Ordinal | `SimpleImputer(strategy="most_frequent")` | `OrdinalEncoder(categories=...)` |

## Report Bus Integration (v1.2.0)

Reads EDA report for column types and quality issues. Writes preprocessing report:
```python
from ml_utils import save_agent_report
save_agent_report("preprocessor", {
    "status": "completed",
    "findings": {"n_features_in": 12, "n_features_out": 28, "pipeline_steps": [...]},
    "artifacts": ["models/preprocessing_pipeline.joblib"]
})
```

## Reflection Gate (v1.2.1)

After preprocessing, ml-theory-advisor validates the pipeline design (encoding choices, scaling, data flow) before training proceeds.

## Full Specification

See `commands/preprocess.md` for complete pipeline templates and advanced transformer patterns.
