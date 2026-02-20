---
name: train
description: "Train a machine learning model with proper data splitting, preprocessing, and validation. Follows best practices to avoid data leakage and overfitting."
user_invocable: true
---

# Model Training Skill

You are training a machine learning model following rigorous best practices.

## Training Workflow

### 1. Data Preparation
- Load and validate the dataset
- Separate features (X) from target (y)
- **CRITICAL**: Split data BEFORE any preprocessing
  ```python
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
  ```

### 2. Preprocessing Pipeline
- Create preprocessing steps using sklearn Pipeline
- Fit transformers ONLY on training data
- Apply to both train and test sets
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])
```

### 3. Model Selection
- Start with a simple baseline (e.g., LogisticRegression, DecisionTree)
- Use cross-validation on training set only
```python
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
```

### 4. Hyperparameter Tuning
- Use GridSearchCV or RandomizedSearchCV
- Tune on training/validation data only
- Never use test set for tuning decisions

### 5. Final Evaluation
- Train final model on full training set
- Evaluate ONCE on held-out test set
- Report appropriate metrics for the problem type

## Best Practices Enforced

- NO fitting on full data before splitting
- NO target leakage in features
- NO evaluation metrics on training data as final results
- ALWAYS use pipelines to prevent data leakage
- ALWAYS set random_state for reproducibility

## Output

Provide:
- Training code with clear documentation
- Cross-validation results
- Final test set performance
- Model summary and next steps

**IMPORTANT**: After training, invoke the `ml-theory-advisor` agent to review the pipeline for potential issues, and the `brutal-code-reviewer` agent to ensure code quality.
