---
name: eda
description: "Perform exploratory data analysis on any dataset. Generates comprehensive statistics, visualizations, and insights about data quality, distributions, correlations, and patterns. Works with CSV, Excel, JSON, or any tabular data."
user_invocable: true
aliases: ["explore", "analyze-data", "data-analysis"]
---

# Exploratory Data Analysis (EDA) Skill

You are performing comprehensive exploratory data analysis on any provided dataset. This works with any tabular data format.

## Your EDA Workflow

### 1. Data Overview
- Load the dataset and display shape, columns, and data types
- Show first and last few rows
- Identify the target variable (if classification/regression)

### 2. Data Quality Assessment
- Check for missing values (count and percentage per column)
- Identify duplicate rows
- Detect potential data type mismatches
- Find columns with constant or near-constant values

### 3. Statistical Summary
- Generate descriptive statistics for numerical columns
- Value counts and frequency distributions for categorical columns
- Identify outliers using IQR method or z-scores

### 4. Distribution Analysis
- Plot histograms for numerical features
- Bar charts for categorical features
- Box plots to visualize outliers

### 5. Correlation Analysis
- Compute correlation matrix for numerical features
- Visualize with heatmap
- Identify highly correlated feature pairs (>0.8 or <-0.8)

### 6. Target Variable Analysis (if applicable)
- Class distribution for classification
- Target distribution for regression
- Feature importance with respect to target

### 7. Key Findings Summary
- Summarize data quality issues
- Highlight potential feature engineering opportunities
- Note any red flags for modeling

## Output Format

Provide findings in clear sections with:
- Code cells for analysis
- Markdown cells explaining insights
- Visualizations where helpful

**IMPORTANT**: After completing EDA, proactively invoke the `ml-theory-advisor` agent to review findings for potential data leakage risks or modeling concerns.
