---
name: team-analyze
description: "Run a quick analysis workflow on any dataset. Performs EDA, quality review, and provides recommendations. Works with any tabular data - CSV, Excel, database exports, etc."
user_invocable: true
aliases: ["team analyze", "analyze", "quick-analysis"]
---

# Team Analyze - Quick Analysis Workflow

You are running a quick analysis workflow that provides comprehensive data understanding. This is useful for:
- Initial data exploration on any dataset
- Data quality assessments
- Quick insights for stakeholders
- Feasibility assessments before larger projects
- Understanding unfamiliar data

## Overview

The `/team analyze` command runs analysis stages only:
1. **EDA** - Comprehensive exploratory data analysis
2. **Quality Review** - Data quality assessment and validation
3. **Recommendations** - Column/feature suggestions based on data type
4. **Summary Report** - Consolidated findings and recommendations

## Usage

```bash
# Analyze any dataset
/team analyze data/sales.csv
/team analyze exports/customers.xlsx
/team analyze db_export.csv

# With specific target variable (enables ML-focused recommendations)
/team analyze data/customers.csv --target Churn

# Focus on specific analysis
/team analyze data/data.csv --focus eda
/team analyze data/data.csv --focus quality
/team analyze data/data.csv --focus recommendations

# Generate visualizations
/team analyze data/data.csv --visualize

# Output format
/team analyze data/data.csv --format html
```

## Your Analysis Workflow

### Step 1: Data Validation

Before analysis, validate the data:

```python
import pandas as pd

# Load and validate
df = pd.read_csv(data_path)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Dtypes: {df.dtypes.value_counts().to_dict()}")
```

**Checks:**
- File exists and is readable
- Has at least 2 columns and 10 rows
- Columns have interpretable names
- No entirely empty columns

### Step 2: EDA (Invoke eda-analyst)

**Prompt for eda-analyst:**
```
Perform comprehensive exploratory data analysis on {data_path}.

Focus on:
1. Data Overview
   - Shape, columns, data types
   - First/last rows preview

2. Data Quality
   - Missing values by column
   - Duplicate rows
   - Constant columns

3. Statistical Summary
   - Numerical: mean, std, quartiles, outliers
   - Categorical: unique values, mode, frequency

4. Distribution Analysis
   - Histograms for numerical
   - Bar charts for categorical
   - Target distribution (if identified)

5. Correlation Analysis
   - Correlation matrix
   - Highly correlated pairs (>0.8)

Output a structured report with findings and recommendations.
```

**Expected Output:**
```markdown
### EDA Report

#### Data Overview
- Rows: 891
- Columns: 12
- Memory: 84 KB

#### Data Quality Issues
| Issue | Count | Columns |
|-------|-------|---------|
| Missing Values | 3 | Age, Cabin, Embarked |
| Duplicates | 0 | - |
| High Cardinality | 2 | Name, Ticket |

#### Key Statistics
| Feature | Type | Unique | Missing % | Notes |
|---------|------|--------|-----------|-------|
| Survived | int | 2 | 0% | Target (38% positive) |
| Pclass | int | 3 | 0% | Strong predictor |
| Sex | str | 2 | 0% | Strong predictor |
| Age | float | 88 | 19.9% | Needs imputation |

#### Recommendations
1. Impute Age using median by Pclass
2. Drop Cabin (77% missing)
3. Consider Name for title extraction
```

### Step 3: Leakage Review (Invoke ml-theory-advisor)

**Prompt for ml-theory-advisor:**
```
Review the dataset {data_path} for potential data leakage.

Check for:
1. Target Leakage
   - Features derived from target
   - Features only known after the event

2. Train-Test Leakage
   - Time-based features that could leak
   - ID-based relationships

3. Feature Leakage
   - Aggregations computed on full dataset
   - Future information encoded

4. Indirect Leakage
   - Proxy variables for target
   - Highly correlated features

Provide specific examples and remediation strategies.
```

**Expected Output:**
```markdown
### Leakage Assessment

#### Risk Level: LOW

#### Findings

| Feature | Risk | Type | Recommendation |
|---------|------|------|----------------|
| Fare | Low | Indirect | Monitor correlation |
| Cabin | None | - | Safe to use |
| Ticket | Low | ID-based | Consider dropping |

#### Safe Features
- Pclass, Sex, Age, SibSp, Parch, Embarked

#### Warnings
- None critical

#### Recommendations
1. Use proper cross-validation
2. Fit preprocessing only on training data
3. Monitor feature importance for anomalies
```

### Step 4: Feature Recommendations (Invoke feature-engineering-analyst)

**Prompt for feature-engineering-analyst:**
```
Analyze {data_path} and recommend feature engineering strategies.

Consider:
1. Domain Knowledge
   - What features make sense for this problem?
   - What interactions might be meaningful?

2. Missing Value Strategies
   - Imputation methods by feature type
   - Indicator variables for missingness

3. Encoding Strategies
   - Categorical encoding options
   - Binning for numerical features

4. Feature Creation
   - Combinations and interactions
   - Aggregations
   - Domain-specific transformations

5. Feature Selection
   - Candidates for removal
   - Importance-based selection

Provide specific, implementable recommendations.
```

**Expected Output:**
```markdown
### Feature Engineering Recommendations

#### New Features to Create

1. **family_size** = SibSp + Parch + 1
   - Rationale: Total family members aboard
   - Expected impact: Medium

2. **is_alone** = 1 if family_size == 1 else 0
   - Rationale: Solo travelers may have different survival rate
   - Expected impact: Low-Medium

3. **title** = extracted from Name
   - Rationale: Social status indicator
   - Expected impact: High

4. **age_group** = binned Age (child/adult/senior)
   - Rationale: Non-linear age effects
   - Expected impact: Medium

5. **fare_per_person** = Fare / family_size
   - Rationale: Actual individual fare
   - Expected impact: Medium

#### Encoding Recommendations

| Feature | Strategy | Rationale |
|---------|----------|-----------|
| Sex | Binary (0/1) | Only 2 values |
| Embarked | One-hot | 3 unordered categories |
| Pclass | Ordinal or one-hot | Could be either |
| Title | Target encoding | Many categories |

#### Features to Drop
- PassengerId (ID only)
- Name (after title extraction)
- Ticket (high cardinality, low value)
- Cabin (77% missing)
```

### Step 5: Summary Report

Compile findings into a comprehensive report:

```markdown
# Analysis Report: {dataset_name}

## Executive Summary
Brief overview of the dataset and key findings.

## Data Quality
Summary of quality issues and recommended actions.

## Target Variable
Distribution and characteristics of the prediction target.

## Key Features
Top predictive features based on analysis.

## Risk Assessment
Data leakage risks and mitigation strategies.

## Feature Engineering Plan
Prioritized list of features to create.

## Recommendations
1. Immediate actions
2. Modeling suggestions
3. Data collection improvements

## Next Steps
- Run `/team coldstart` for full pipeline
- Or address specific issues first
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | Auto-detect | Target variable name |
| `--focus` | all | Focus area (eda, leakage, features) |
| `--visualize` | false | Generate visualization files |
| `--output` | reports/ | Output directory for report |
| `--format` | markdown | Report format (md, html, pdf) |

## Agent Coordination (v1.2.0 — Report Bus)

This skill coordinates agents using the shared report bus:

1. **eda-analyst** - Data exploration (runs first, writes report to bus)
2. **ml-theory-advisor** - Leakage assessment (runs in PARALLEL after EDA)
3. **feature-engineering-analyst** - Feature recommendations (runs in PARALLEL after EDA)

### Parallel Execution

After EDA completes, spawn ml-theory-advisor AND feature-engineering-analyst concurrently using multiple Task tool calls in a single message. Both agents read the EDA report independently and write their own reports to the bus.

Each agent should:
- Read prior reports from `.claude/reports/` on startup
- Write their report using `save_agent_report()` on completion

### Status Check

After all agents complete, display workflow status:
```python
from ml_utils import get_workflow_status
status = get_workflow_status()
print(f"Completed: {len(status['completed'])}, Pending: {len(status['pending'])}")
for insight in status['insights']:
    print(f"  {insight['from']} → {insight['to']}: {insight['action']}")
```

## Output Files

Generated files:
- `reports/eda_report_{timestamp}.md` - EDA findings
- `reports/analysis_summary_{timestamp}.md` - Combined report
- `reports/figures/` - Visualizations (if --visualize)
