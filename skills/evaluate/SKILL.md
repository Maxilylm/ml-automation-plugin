---
name: evaluate
description: Comprehensive model evaluation with appropriate metrics, visualizations, and interpretation. Includes confusion matrix, ROC curves, feature importance, and error analysis.
---

# Model Evaluation

## When to Use
- Evaluating trained model performance
- Generating evaluation reports with visualizations
- Comparing models or analyzing errors

## Workflow

1. **Generate predictions** — On held-out test set (never train set)
2. **Compute metrics** — Classification: accuracy, precision, recall, F1, AUC-ROC. Regression: RMSE, MAE, R2
3. **Confusion matrix** — `sklearn.metrics.confusion_matrix` + classification report
4. **ROC curves** — Per-class ROC with AUC values
5. **Feature importance** — Model-specific (tree: `.feature_importances_`, linear: `.coef_`)
6. **Error analysis** — Examine worst predictions, identify patterns in misclassifications
7. **Summary** — Actionable recommendations (retrain, add features, adjust threshold)

## Task-Type Awareness

| Task Type | Key Metrics | Watch For |
|-----------|------------|-----------|
| Classification | F1, AUC-ROC, precision/recall | Class imbalance, threshold sensitivity |
| Regression | RMSE, MAE, R2 | Heteroscedasticity, outlier influence |
| Time Series | MAPE, RMSE, directional accuracy | Look-ahead bias, non-stationarity |

## Report Bus Integration (v1.2.0)

Read training report for model details. Write evaluation report:
```python
from ml_utils import save_agent_report
save_agent_report("evaluator", {
    "status": "completed",
    "findings": {"accuracy": 0.85, "f1": 0.81, "auc": 0.88},
    "recommendations": [{"text": "Consider threshold tuning", "target_agent": "developer"}],
    "artifacts": ["reports/evaluation_report.md", "reports/figures/roc_curve.png"]
})
```

## Full Specification

See `commands/evaluate.md` for complete evaluation workflows and output templates.
