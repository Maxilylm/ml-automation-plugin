---
name: team-analyze
description: Run a quick analysis workflow on any dataset. Performs EDA, quality review, and provides recommendations. Works with any tabular data - CSV, Excel, database exports, etc.
---

# Team Analyze

## When to Use
- Quick dataset analysis with automated quality review
- Getting recommendations for data preparation and modeling
- Running a coordinated analysis with multiple specialist agents

## Workflow

1. Data validation and loading
2. EDA (parallel with ML theory review and feature analysis)
3. Quality review and leakage detection
4. Feature engineering recommendations
5. Summary with actionable next steps

## Report Bus Integration (v1.2.0)

All agents write structured reports to the shared report bus. After EDA completes, spawn ml-theory-advisor and feature-engineering-analyst in parallel — they both read the EDA report independently.

Use `ml_utils.get_workflow_status()` after all agents complete to display a unified summary.
