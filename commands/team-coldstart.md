---
name: team-coldstart
description: "Launch a full data workflow from raw data to deployed solution with interactive dashboard. Orchestrates multiple agents through analysis, processing, modeling, and deployment stages. Works with any tabular data."
user_invocable: true
aliases: ["team coldstart", "coldstart", "full-pipeline", "run-all"]
---

# Team Cold Start - Full Data Workflow Orchestration

You are launching a complete data workflow from raw data to deployed solution with an interactive Streamlit dashboard. This skill orchestrates multiple specialized agents through all stages of the data lifecycle.

## Overview

The `/team coldstart` command initiates an end-to-end data pipeline:
1. **Initialize** - Set up project, validate data, detect data type
2. **Analyze** - EDA, quality review, feature/column recommendations
3. **Process** - Build and validate data processing pipeline
4. **Model** - Train and compare models (if prediction task) OR generate insights (if analysis task)
5. **Evaluate** - Comprehensive evaluation with metrics and visualizations
6. **Dashboard** - Create interactive Streamlit dashboard with results
7. **Productionalize** - Create production-ready code and API
8. **Deploy** - Deploy to target environment (optional)
9. **Finalize** - Generate reports, merge PRs

## Usage

```bash
# Full pipeline with auto-detection
/team coldstart data/raw/sales.csv

# With specific target variable (ML task)
/team coldstart data/raw/customers.csv --target Churn

# Analysis-only mode (no ML, just insights + dashboard)
/team coldstart data/raw/survey.csv --mode analysis

# Skip deployment stage
/team coldstart data/raw/data.csv --no-deploy

# Deploy to specific target
/team coldstart data/raw/data.csv --deploy-to snowflake

# Custom dashboard title
/team coldstart data/raw/data.csv --dashboard-title "Sales Analytics"
```

## Your Orchestration Workflow

### Stage 1: Initialize

**Actions:**
1. Create feature branch: `feature/data-pipeline-{timestamp}`
2. Create task list for tracking progress
3. Validate input data exists and is readable
4. Auto-detect data type and task mode:
   - **ML Mode**: Target column specified or detectable (classification/regression)
   - **Analysis Mode**: No clear target, focus on insights and visualization
5. Set up project structure

**Tasks to Create:**
- [ ] Initialize project structure
- [ ] Validate input data quality
- [ ] Run exploratory data analysis
- [ ] Build data processing pipeline
- [ ] Train models OR generate insights
- [ ] Evaluate results
- [ ] Create Streamlit dashboard
- [ ] Create production code
- [ ] Deploy (if requested)
- [ ] Generate final report

**Output:**
```markdown
## Stage 1: Initialize ✓

- Branch: feature/data-pipeline-20240115-1030
- Data validated: data/raw/sales.csv (10000 rows, 25 columns)
- Detected mode: ML (target: Revenue) OR Analysis (no target)
- Quality check: PASSED (no critical issues)
```

### Stage 2: Analysis (Sequential → Parallel)

**Step 2a: Run EDA first** (generates structured report for downstream agents):

1. **eda-analyst** - Comprehensive EDA + Structured Report
   ```
   Perform thorough exploratory data analysis on {data_path}.
   Generate statistics, visualizations, and identify data quality issues.

   CRITICAL: Use ml_utils to save structured EDA report:
   - Copy ml_utils.py from ~/.config/opencode/ml-automation/templates/ml_utils.py to src/ml_utils.py
   - Call generate_eda_summary(df, target_col) and save_eda_report(summary)
   - Also save reports/eda_report.md with narrative findings

   Output: EDA report + .claude/eda_report.json (structured)
   ```

**Step 2b: Run these in parallel** (after EDA completes, so they can read its report):

2. **ml-theory-advisor** - Data quality and leakage review (if ML mode)
   ```
   Review the dataset {data_path} for potential data leakage risks.
   Check for features that may contain target information.
   Read .claude/eda_report.json for prior EDA context if available.
   Validate data splitting strategy.
   Output: Quality and leakage assessment report
   ```

3. **feature-engineering-analyst** - ML Feature Engineering
   ```
   Analyze {data_path} and engineer predictive features.
   READ .claude/eda_report.json first — use the EDA findings (distributions, correlations,
   quality issues, column types) to inform your feature engineering strategy.

   Recommend:
   - Feature transformations (log, binning, interactions, temporal)
   - Aggregation features
   - Encoding strategies
   - Features to drop (leakage, redundancy)
   Output: Feature engineering plan with implementation code
   ```

**Output:**
```markdown
## Stage 2: Analysis ✓

### EDA Summary
- Dataset: 10,000 rows x 25 columns
- Target: Revenue (continuous) OR None (analysis mode)
- Key columns: Category, Region, Date, Amount
- Missing values: 3 columns with >5% missing
- Recommendations: Handle nulls, normalize amounts

### Data Quality Assessment
- No critical leakage detected (ML mode)
- Data distribution: Normal for most numeric columns
- Outliers identified in 2 columns

### Column/Feature Recommendations
1. Create time-based features from Date
2. Aggregate by Category for insights
3. Normalize Amount by Region
```

### Stage 3: Data Processing

**Actions:**
1. Build data processing pipeline based on EDA findings
2. Handle missing values, encoding, scaling as needed
3. Invoke `ml-theory-advisor` to validate no leakage (if ML mode)
4. Generate unit tests for processing functions
5. Create PR for processing code
6. Invoke `brutal-code-reviewer` for code review
7. Merge PR after approval

**Code to generate:**
```python
# First, ensure ml_utils.py is in the project (copy from plugin templates if not present)
# src/ml_utils.py provides: load_data, detect_column_types, build_preprocessor, safe_split, etc.

# src/processing.py
from src.ml_utils import detect_column_types, build_preprocessor, load_eda_report

def process_data(df, target_col=None, mode='ml'):
    """Process raw data for modeling or analysis.

    Uses prior EDA report if available to inform preprocessing decisions.
    """
    # Load EDA context if available
    eda_report = load_eda_report()

    # Detect column types (or use EDA report's classification)
    if eda_report and "column_types" in eda_report:
        col_types = eda_report["column_types"]
    else:
        col_types = detect_column_types(df, target_col=target_col)

    # Build preprocessor
    preprocessor = build_preprocessor(col_types["numerical"], col_types["categorical"])

    return preprocessor, col_types
```

**Output:**
```markdown
## Stage 3: Data Processing ✓

- Pipeline created: src/processing.py
- Tests generated: tests/unit/test_processing.py
- Coverage: 92%
- PR #1: Merged ✓
- Quality check: PASSED
```

### Stage 4: Modeling / Insights Generation

**For ML Mode:**
1. Train baseline model (appropriate for task type)
2. Train advanced models (ensemble methods)
3. Invoke `ml-theory-advisor` to review methodology
4. Compare and select best model

**For Analysis Mode:**
1. Generate statistical summaries
2. Create aggregations and pivots
3. Identify key insights and patterns
4. Generate insight report

**Code to generate:**
```python
# src/model.py (ML mode)
def train_model(X, y, model_type='auto'):
    """Train and return best model for the task."""
    ...

# src/insights.py (Analysis mode)
def generate_insights(df, dimensions, metrics):
    """Generate key insights from data."""
    ...
```

**Output (ML Mode):**
```markdown
## Stage 4: Modeling ✓

### Models Trained
| Model | Metric 1 | Metric 2 | Metric 3 |
|-------|----------|----------|----------|
| Baseline | 0.79 | 0.76 | 0.74 |
| Advanced 1 | 0.84 | 0.82 | 0.80 |
| Advanced 2 | 0.85 | 0.83 | 0.81 |

Best model: Advanced 2
PR #2: Merged ✓
```

**Output (Analysis Mode):**
```markdown
## Stage 4: Insights ✓

### Key Findings
1. Revenue increased 23% YoY
2. Top 3 categories drive 65% of sales
3. Regional variance identified

Insights saved: reports/insights.md
PR #2: Merged ✓
```

### Stage 5: Evaluation

**For ML Mode:**
1. Run comprehensive model evaluation
2. Generate visualizations (confusion matrix, ROC, feature importance)
3. Invoke `ml-theory-advisor` to validate evaluation
4. Generate evaluation report

**For Analysis Mode:**
1. Validate insight accuracy
2. Generate supporting visualizations
3. Cross-check findings with data
4. Generate analysis report

**Output:**
```markdown
## Stage 5: Evaluation ✓

### Performance Summary (ML) / Validation Summary (Analysis)
- Key metrics validated
- Visualizations generated
- Cross-validation complete (ML)
- Insights verified (Analysis)

Methodology validated ✓
```

### Stage 6: Streamlit Dashboard

**Actions:**
1. Create interactive Streamlit dashboard with all results
2. Include visualizations from EDA and evaluation
3. Add interactive filters and controls
4. Include model predictions (ML) or data explorer (Analysis)
5. Generate dashboard code

**Dashboard Components:**

```python
# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="{Project} Dashboard", layout="wide")

# Sidebar filters
st.sidebar.header("Filters")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Analysis", "Results", "Predictions"])

with tab1:
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "{count}")
    col2.metric("Key Metric 1", "{value}")
    col3.metric("Key Metric 2", "{value}")
    col4.metric("Model Accuracy", "{accuracy}%")  # ML mode only

    # Overview charts
    st.plotly_chart(fig_overview)

with tab2:
    # EDA visualizations
    st.subheader("Data Distribution")
    st.plotly_chart(fig_distributions)

    st.subheader("Correlations")
    st.plotly_chart(fig_correlations)

with tab3:
    # Results (ML: model performance / Analysis: key insights)
    st.subheader("Key Findings")
    # Feature importance or insight cards

with tab4:
    # Interactive predictions (ML) or data explorer (Analysis)
    st.subheader("Make Predictions" if ml_mode else "Explore Data")
    # Input form for predictions or data filters
```

**Output:**
```markdown
## Stage 6: Streamlit Dashboard ✓

- Dashboard created: dashboard/app.py
- Components: Overview, Analysis, Results, Predictions
- Visualizations: 8 interactive charts
- Filters: Date range, Category, Region

### To Run Dashboard:
```bash
streamlit run dashboard/app.py
```

Dashboard URL: http://localhost:8501
PR #3: Merged ✓
```

### Stage 7: Productionalization

**Actions:**
1. Invoke `mlops-engineer` to create production code
2. Create FastAPI endpoint (predictions for ML, data API for analysis)
3. Create Dockerfile with dashboard included
4. Generate integration tests
5. Create PR for production code
6. Invoke `brutal-code-reviewer` for review
7. Merge after approval

**Files to generate:**
- `api/app.py` - FastAPI application
- `api/schemas.py` - Pydantic models
- `dashboard/app.py` - Streamlit dashboard
- `Dockerfile` - Container configuration
- `docker-compose.yml` - Service orchestration (API + Dashboard)

**Output:**
```markdown
## Stage 7: Productionalization ✓

- API created: api/app.py
- Dashboard created: dashboard/app.py
- Docker configured: Dockerfile
- Integration tests: 85% coverage
- PR #4: Merged ✓
- Production checklist: PASSED
```

### Stage 8: Deployment (Optional)

**If deployment requested:**

1. Determine target (local, snowflake, streamlit-sis, aws, gcp)
2. Invoke appropriate agent:
   - **Local**: `mlops-engineer` - Docker deployment
   - **Snowflake**: Deploy to Snowflake + Streamlit in Snowflake
   - **Cloud**: AWS/GCP deployment

3. Deploy both API and Dashboard
4. Verify health checks

**Output:**
```markdown
## Stage 8: Deployment ✓

- Target: local (or snowflake/aws/gcp)
- API deployed: http://localhost:8000
- Dashboard deployed: http://localhost:8501
- Health check: PASSED
```

### Stage 9: Finalization

**Actions:**
1. Generate comprehensive final report
2. Ensure all PRs merged to main
3. Update documentation
4. Close tasks
5. Output dashboard URL and access instructions

**Final Report:**
```markdown
## Project Complete: {Project Name}

### Summary
- Duration: {time}
- Mode: ML / Analysis
- PRs Merged: {count}
- Test Coverage: {percentage}%
- Key Metric: {value}

### Artifacts
- Data Pipeline: src/processing.py
- Model/Insights: src/model.py or src/insights.py
- API: http://localhost:8000
- Dashboard: http://localhost:8501

### Files Created
- src/processing.py
- src/model.py (ML) or src/insights.py (Analysis)
- api/app.py
- dashboard/app.py
- tests/unit/test_*.py
- tests/integration/test_api.py
- Dockerfile
- docker-compose.yml

### Dashboard Access
```bash
# Local development
streamlit run dashboard/app.py

# Docker
docker-compose up
# Dashboard: http://localhost:8501
# API: http://localhost:8000
```

### Next Steps
1. Review dashboard and customize visualizations
2. Share dashboard URL with stakeholders
3. Set up monitoring and alerts
4. Plan for data refresh schedule
```

## Error Handling

If any stage fails:
1. Log the error with context
2. Attempt automatic remediation if possible
3. Create an issue for manual intervention
4. Continue with remaining independent stages
5. Report partial completion status

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--target` | Auto-detect | Target variable name (triggers ML mode) |
| `--mode` | auto | Force mode: `ml` or `analysis` |
| `--no-deploy` | false | Skip deployment stage |
| `--deploy-to` | local | Deployment target (local, snowflake, aws, gcp) |
| `--dashboard-title` | Auto | Custom title for Streamlit dashboard |
| `--test-size` | 0.2 | Test split ratio (ML mode) |
| `--cv-folds` | 5 | Cross-validation folds (ML mode) |
| `--coverage-threshold` | 80 | Minimum test coverage |
| `--no-dashboard` | false | Skip dashboard creation |

## Dashboard Customization

The generated Streamlit dashboard includes:
- **Overview Tab**: KPI metrics, summary statistics
- **Analysis Tab**: EDA visualizations, distributions, correlations
- **Results Tab**: Model performance (ML) or key insights (Analysis)
- **Interactive Tab**: Predictions (ML) or data explorer (Analysis)

To customize after generation:
1. Edit `dashboard/app.py`
2. Add custom visualizations to `dashboard/components/`
3. Modify filters in sidebar
4. Run `streamlit run dashboard/app.py` to preview

## Agent Coordination

Throughout the workflow, agents are coordinated to:
- Track progress via task list
- Coordinate handoffs between stages
- Handle blockers and escalations
- Generate status updates

Invoke agents using the Task tool with appropriate prompts and context from previous stages.

## Output Structure

```
project/
├── src/
│   ├── processing.py      # Data processing pipeline
│   ├── model.py           # ML model (if ML mode)
│   └── insights.py        # Insights generation (if Analysis mode)
├── api/
│   ├── app.py             # FastAPI application
│   └── schemas.py         # Pydantic models
├── dashboard/
│   ├── app.py             # Streamlit dashboard
│   └── components/        # Reusable dashboard components
├── tests/
│   ├── unit/
│   └── integration/
├── reports/
│   ├── eda_report.md
│   └── final_report.md
├── models/                # Saved models (ML mode)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
