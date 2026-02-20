---
name: preprocess
description: "Data processing pipeline creation for any tabular data. Handles missing values, encoding, scaling, and transformations. For ML tasks, ensures no data leakage."
user_invocable: true
aliases: ["process", "clean", "transform"]
---

# Data Processing Skill

You are creating a robust data processing pipeline. For ML tasks, this prevents data leakage. For analysis tasks, this ensures clean, consistent data.

## Preprocessing Workflow

### 0. Initialize Reusable Utilities

Before writing boilerplate, copy the shared utilities into the project:

```python
import shutil, os
utils_src = os.path.expanduser("~/.config/opencode/ml-automation/templates/ml_utils.py")
if not os.path.exists("src/ml_utils.py"):
    os.makedirs("src", exist_ok=True)
    shutil.copy2(utils_src, "src/ml_utils.py")

from src.ml_utils import detect_column_types, build_preprocessor, safe_split, load_eda_report
```

Also check for prior EDA reports to inform preprocessing decisions:

```python
eda_report = load_eda_report()
if eda_report:
    print(f"Found EDA report: {eda_report['shape']['rows']} rows, {eda_report['shape']['cols']} cols")
    print(f"Quality issues: {len(eda_report.get('quality_issues', []))}")
    # Use this to inform imputation, scaling, and encoding choices
```

### 1. Identify Column Types
```python
# Use the reusable utility instead of manual detection
col_types = detect_column_types(df, target_col=target_col)
numerical_cols = col_types["numerical"]
categorical_cols = col_types["categorical"]
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
# Quick version using reusable utility:
preprocessor = build_preprocessor(numerical_cols, categorical_cols)

# Or customize if EDA revealed specific needs:
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

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
