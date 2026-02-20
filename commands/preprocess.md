---
name: preprocess
description: "Data processing pipeline creation for any tabular data. Handles missing values, encoding, scaling, and transformations. For ML tasks, ensures no data leakage."
user_invocable: true
aliases: ["process", "clean", "transform"]
---

# Data Processing Skill

You are creating a robust data processing pipeline. For ML tasks, this prevents data leakage. For analysis tasks, this ensures clean, consistent data.

## Preprocessing Workflow

### 1. Identify Column Types
```python
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target from feature lists
if target_col in numerical_cols:
    numerical_cols.remove(target_col)
if target_col in categorical_cols:
    categorical_cols.remove(target_col)
```

### 2. Missing Value Strategy
- **Numerical**: Median imputation (robust to outliers) or mean
- **Categorical**: Mode imputation or 'Unknown' category
- **Advanced**: KNN imputation, iterative imputation

```python
from sklearn.impute import SimpleImputer, KNNImputer

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
```

### 3. Encoding Categorical Variables
- **Nominal (no order)**: OneHotEncoder
- **Ordinal (has order)**: OrdinalEncoder with specified order
- **High cardinality**: TargetEncoder (careful of leakage!)

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# For nominal categories
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# For ordinal categories
ordinal_encoder = OrdinalEncoder(categories=[['low', 'medium', 'high']])
```

### 4. Scaling Numerical Features
- **StandardScaler**: When features should have zero mean, unit variance
- **MinMaxScaler**: When bounded range [0,1] is needed
- **RobustScaler**: When data has outliers

```python
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = StandardScaler()  # Most common choice
```

### 5. Build Complete Pipeline
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols)
])
```

### 6. Feature Engineering (Optional)
- Polynomial features for non-linear relationships
- Binning for continuous variables
- Interaction features
- Domain-specific transformations

## Data Leakage Prevention Checklist

- [ ] Split data BEFORE preprocessing
- [ ] Fit transformers on training data ONLY
- [ ] Transform (not fit_transform) on test data
- [ ] No target information in features
- [ ] No future information in features (time series)

## Output

Provide:
1. Column type identification
2. Missing value analysis and strategy
3. Complete preprocessing pipeline code
4. Instructions for fitting and transforming

**IMPORTANT**: After creating the preprocessing pipeline, invoke the `ml-theory-advisor` agent to verify no data leakage risks exist in the pipeline design.
