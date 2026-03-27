# Rigorous Evaluation Comparison Framework

## Problem Statement

We need a systematic way to compare the ml-automation plugin's output quality against:
1. Manual data science workflows (baseline)
2. Off-the-shelf tools (AutoML platforms, notebooks)
3. Previous plugin versions (regression testing)

## Evaluation Dimensions

### 1. Completeness Score (0-100)
Does the output cover all expected aspects?

| Aspect | Weight | Criteria |
|--------|--------|----------|
| Data Quality Assessment | 15% | Missing values, duplicates, outliers, type issues identified |
| Statistical Analysis | 15% | Distributions, correlations, target analysis performed |
| Feature Engineering | 15% | Relevant features created, leakage risks assessed |
| Model Selection | 15% | Multiple models compared, rationale documented |
| Evaluation Rigor | 15% | CV, multiple metrics, confusion matrix, calibration |
| Reproducibility | 10% | Data fingerprint, random seeds, package versions logged |
| Documentation | 10% | Methodology doc, decision trail, assumptions listed |
| Deployment Readiness | 5% | API, Docker, monitoring artifacts created |

### 2. Correctness Score (0-100)
Are the outputs technically correct?

| Check | Weight | Method |
|-------|--------|--------|
| No data leakage | 25% | Verify split-before-transform, no future data in train set |
| Metrics accurate | 20% | Re-compute metrics independently, compare |
| Feature importance valid | 15% | Cross-validate with permutation importance |
| Statistical tests appropriate | 15% | Verify test assumptions met |
| Code runs without errors | 15% | Execute generated code end-to-end |
| Predictions reasonable | 10% | Business plausibility checks pass |

### 3. Time-to-Insight (seconds)
Wall clock time from data upload to:
- First EDA summary
- First model trained
- Full pipeline complete
- Dashboard deployed

### 4. Reproducibility Score (0-100)
Can another data scientist reproduce the results?

| Check | Weight |
|-------|--------|
| Data version recorded | 20% |
| Random seeds documented | 20% |
| Package versions logged | 20% |
| Step-by-step instructions | 20% |
| Methodology document generated | 20% |

## Benchmark Datasets

Use standardized datasets where the "correct" analysis is known:

| Dataset | Task | Key Traps | Expected Outcome |
|---------|------|-----------|-----------------|
| Titanic | Classification | Missing values, derived features | ~85% accuracy, detect SibSp+Parch → FamilySize |
| Boston Housing | Regression | Multicollinearity, outliers | R² > 0.7, flag correlated features |
| Credit Card Fraud | Imbalanced | Severe class imbalance, leakage risk | Use AUPRC not accuracy, detect time-based leakage |
| Retail Sales | Time Series | Seasonality, trend | Decompose correctly, appropriate forecasting model |
| MMM Synthetic | MMM | Saturation, adstock | Recover true ROAS within 20% |

## Comparison Protocol

### Against Manual Baseline
1. Expert data scientist performs full analysis on benchmark dataset (gold standard)
2. Plugin runs `/team-coldstart` on same dataset
3. Score both on completeness, correctness, and reproducibility
4. Compare time-to-insight

### Against Off-the-Shelf Tools
1. Run same dataset through: H2O AutoML, TPOT, AutoGluon, DataRobot (if available)
2. Compare on all 4 dimensions
3. Note: off-the-shelf tools won't have reproducibility docs — that's a plugin advantage

### Version-over-Version Regression
1. Run benchmark suite on version N
2. Run same suite on version N+1
3. Flag any dimension that degraded

## Integration with Eval Framework

Add benchmark evals to `evals/`:

```json
{
  "skill_name": "benchmark",
  "evals": [
    {
      "id": 1,
      "name": "titanic-completeness",
      "prompt": "Run full pipeline on eda-workspace/test-datasets/churn_data.csv with target='churned'",
      "expected_output": "Complete pipeline with all stages, methodology doc, and dashboard",
      "assertions": [
        {"id": "bench-eda", "text": "EDA report generated with quality assessment", "type": "content_check"},
        {"id": "bench-features", "text": "Feature engineering with leakage assessment", "type": "content_check"},
        {"id": "bench-model", "text": "Multiple models compared", "type": "content_check"},
        {"id": "bench-metrics", "text": "CV metrics reported, not just test set", "type": "content_check"},
        {"id": "bench-methodology", "text": "Methodology document generated", "type": "file_check"},
        {"id": "bench-reproducibility", "text": "Data fingerprint and seeds recorded", "type": "content_check"},
        {"id": "bench-plausibility", "text": "Business plausibility checks run", "type": "content_check"}
      ]
    }
  ]
}
```

## Scoring Tool

```python
# evals/benchmark_scorer.py
def score_pipeline_output(output_dir, benchmark_config):
    """Score a completed pipeline against benchmark criteria."""
    scores = {
        "completeness": _score_completeness(output_dir),
        "correctness": _score_correctness(output_dir),
        "reproducibility": _score_reproducibility(output_dir),
        "time_to_insight": _measure_time(output_dir),
    }
    return scores
```

## Output Format

```markdown
# Benchmark Evaluation Report

## Dataset: Churn Prediction
## Plugin Version: 1.8.0
## Date: 2026-03-24

### Scores
| Dimension | Score | vs. Manual | vs. Previous |
|-----------|-------|------------|--------------|
| Completeness | 92/100 | +5 | +3 |
| Correctness | 88/100 | -2 | +1 |
| Reproducibility | 95/100 | +30 | +5 |
| Time-to-insight | 180s | -85% | -10% |

### Detailed Findings
...
```
