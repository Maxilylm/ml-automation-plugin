---
name: eda
description: Perform exploratory data analysis on any dataset. Generates comprehensive statistics, visualizations, and insights about data quality, distributions, correlations, and patterns. Works with CSV, Excel, JSON, or any tabular data.
---

# Exploratory Data Analysis (EDA)

## When to Use
- Exploring a new dataset
- Investigating data quality issues
- Understanding variable relationships and patterns
- Preparing data summaries for stakeholders

## Workflow

1. **Data Overview** - Load dataset, display shape, columns, data types, first/last rows
2. **Data Quality Assessment** - Missing values, duplicates, type mismatches, constant columns
3. **Statistical Summary** - Descriptive stats for numerical, value counts for categorical, outlier detection
4. **Distribution Analysis** - Histograms, bar charts, box plots
5. **Correlation Analysis** - Correlation matrix, heatmap, highly correlated pairs (>0.8 or <-0.8)
6. **Target Variable Analysis** - Class distribution, feature importance vs target
7. **Key Findings Summary** - Data quality issues, feature engineering opportunities, modeling red flags

## Output Format

Provide findings in clear sections with code cells, markdown explanations, and visualizations.

After completing EDA, invoke the `ml-theory-advisor` agent to review findings for potential data leakage risks.
