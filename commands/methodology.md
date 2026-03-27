---
name: methodology
description: "Generate a stakeholder-facing analytical approach document that outlines the end-to-end methodology, statistical methods, assumptions, decision points, and intermediate artefacts for reproducibility and review."
user_invocable: true
aliases: ["analytical-approach", "methods-doc", "governance-report"]
---

# Methodology Document Generator

You are generating a comprehensive analytical approach document for data science stakeholders. This document provides full transparency into the methods, assumptions, and decisions made during the analysis.

## Purpose

This document is for **data science stakeholders** who need to:
- Understand the end-to-end analytical approach
- Review statistical methods and assumptions
- Reproduce the analysis
- Audit decision points and intermediate artefacts

## How to Generate

### Step 1: Gather Context

Read all available sources of analytical context:

```python
from ml_utils import (
    load_agent_reports, load_eda_report, load_experiments,
    load_model_registry, load_feature_store, load_data_versions,
    load_lessons, load_trace_log, get_workflow_status,
)

reports = load_agent_reports()
eda = load_eda_report()
experiments = load_experiments()
registry = load_model_registry()
features = load_feature_store()
data_versions = load_data_versions()
lessons = load_lessons()
trace = load_trace_log()
status = get_workflow_status()
```

Also scan for:
- `plausibility_config.json` — domain expectations
- `reports/*` — any generated reports
- `models/` — model artifacts
- Stage plans: `stage_plan_*.json`
- Reflection reports: `*_reflection_*_report.json`

### Step 2: Generate Document

Write a Markdown document following this structure:

```markdown
# Analytical Approach Document

**Project**: {project_name}
**Generated**: {date}
**Data Fingerprint**: {fingerprint}
**Status**: {pipeline_status}

---

## 1. Objective
- Business question being addressed
- Success criteria defined
- Task type: {classification|regression|mmm|segmentation|time_series}

## 2. Data Description
- Source: {data_path}
- Shape: {rows} rows × {columns} columns
- Time range: {if applicable}
- Target variable: {target_col}
- Data fingerprint: {sha256 hash for reproducibility}
- Data quality summary from EDA

## 3. Exploratory Data Analysis Summary
- Key distributions and patterns
- Data quality issues identified
- High correlations flagged
- Recommendations from EDA agent

## 4. Feature Engineering
- Features created (from feature store)
- Transformation rationale
- Leakage risk assessment per feature
- Features dropped and why

## 5. Statistical Methods
- Model type and algorithm selected
- Why this algorithm (from experiment rationale)
- Preprocessing pipeline description
- Train/test split strategy (stratification, test size)
- Cross-validation approach
- Hyperparameter tuning method

## 6. Assumptions
- Distributional assumptions
- Independence assumptions
- Stationarity assumptions (if time series)
- Business domain assumptions (from plausibility config)
- Known limitations

## 7. Decision Points
- Key decisions made during analysis (from trace log)
- Reflection gate outcomes (approved/revised)
- Alternative approaches considered (from experiments)
- Why alternatives were rejected

## 8. Model Results
- Performance metrics (from model registry)
- Confusion matrix / residual analysis
- Feature importance ranking
- Business plausibility check results (if available)
- Model comparison table (all experiments)

## 9. Intermediate Artefacts
- List all generated artifacts with paths:
  - EDA report
  - Preprocessing pipeline
  - Model artifacts
  - Evaluation reports
  - Dashboard
  - Docker/API assets

## 10. Reproducibility Guide
- Environment requirements
- Data access instructions
- Step-by-step reproduction commands
- Random seeds used
- Package versions

## 11. Lessons Learned
- Issues encountered and resolutions
- Patterns discovered for future use

## 12. Appendix
- Full experiment log
- Detailed reflection gate reports
- Traceability log summary
```

### Step 3: Save Document

Save to `reports/methodology_{timestamp}.md` and also generate a JSON summary:

```python
from ml_utils import save_agent_report, log_trace_event

save_agent_report("methodology-generator", {
    "status": "completed",
    "findings": {
        "summary": "Analytical approach document generated",
        "details": {"sections": 12, "data_sources": [...], "decision_points": N}
    },
    "artifacts": ["reports/methodology_{timestamp}.md"],
})

log_trace_event({
    "event_type": "agent_complete",
    "agent": "methodology-generator",
    "action": "Generated analytical approach document",
    "outputs_summary": "reports/methodology_{timestamp}.md",
    "command": "methodology",
})
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output` | Output file path | `reports/methodology_{timestamp}.md` |
| `--format` | Output format (md, html, pdf) | md |
| `--include-trace` | Include full traceability log | false |
| `--include-appendix` | Include detailed appendix | true |
| `--project-name` | Override project name | auto-detected |

## When No Data Exists

If no reports or registry data are found, generate a template document with placeholders and instructions for what data would need to exist.
