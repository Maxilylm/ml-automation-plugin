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
- Set random_state for reproducibility

## Workflow

1. Data preparation and train/test split
2. Build preprocessing pipeline
3. Baseline model with cross-validation
4. Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
5. Final evaluation on test set
6. Report metrics and next steps
