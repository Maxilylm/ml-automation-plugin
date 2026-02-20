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

- **Split before transform** - Always split data before fitting any preprocessor
- **Pipeline everything** - Use sklearn Pipeline to prevent leakage
- **Fit on train only** - Never fit transformers on test data

## Workflow

1. Load and validate data
2. Split into train/test FIRST
3. Build ColumnTransformer with appropriate transformers
4. Fit on training data only
5. Transform both train and test
6. Save pipeline for reproducibility
