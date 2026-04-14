---
name: evaluate
description: "Comprehensive model evaluation with appropriate metrics, visualizations, and interpretation. Includes confusion matrix, ROC curves, feature importance, and error analysis."
user_invocable: true
---

# Model Evaluation Skill

You are performing comprehensive model evaluation with proper methodology.

## Evaluation Workflow

### 1. Classification Metrics
For classification problems, compute and explain:
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
```

- **Accuracy**: Overall correctness (use with caution for imbalanced data)
- **Precision**: Of predicted positives, how many are correct
- **Recall**: Of actual positives, how many were found
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Model's ability to discriminate between classes

### 2. Confusion Matrix Analysis
- Visualize confusion matrix with seaborn heatmap
- Analyze false positives vs false negatives
- Discuss business implications of each error type

### 3. ROC and Precision-Recall Curves
```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
RocCurveDisplay.from_estimator(model, X_test, y_test)
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
```

### 4. Feature Importance
- For tree-based models: feature_importances_
- For linear models: coefficients

### 5. SHAP Analysis (v1.9.0)

Model-agnostic feature attribution using SHAP (SHapley Additive exPlanations):
```python
import shap

explainer = shap.Explainer(model, X_test)
shap_values = explainer(X_test)

# Summary plot — global feature importance with direction of effect
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("reports/figures/shap_summary.png", bbox_inches="tight", dpi=150)
plt.close()

# Dependence plots for top 3 features — interaction effects
for feat in top_features[:3]:
    shap.dependence_plot(feat, shap_values.values, X_test, show=False)
    plt.savefig(f"reports/figures/shap_dep_{feat}.png", bbox_inches="tight", dpi=150)
    plt.close()
```

If `shap` is not installed, install it: `pip install shap`. For tree models use `shap.TreeExplainer` (faster). For linear models use `shap.LinearExplainer`. Fall back to `shap.Explainer` (auto-selects) for any other model type.

### 6. Calibration Curve (v1.9.0) — Classification Only

Verify that predicted probabilities are trustworthy:
```python
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

CalibrationDisplay.from_estimator(model, X_test, y_test, n_bins=10, name=model_name)
plt.savefig("reports/figures/calibration_curve.png", bbox_inches="tight", dpi=150)
plt.close()
```

If the calibration curve deviates significantly from the diagonal, recommend recalibration:
```python
calibrated = CalibratedClassifierCV(model, cv=5, method="isotonic")
calibrated.fit(X_train, y_train)
```

### 7. Learning Curve (v1.9.0)

Diagnose bias vs variance by plotting performance against training set size:
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring=scoring_metric, n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
plt.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
plt.plot(train_sizes, val_scores.mean(axis=1), label="Validation score")
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
plt.xlabel("Training set size")
plt.ylabel(scoring_metric)
plt.title("Learning Curve")
plt.legend()
plt.savefig("reports/figures/learning_curve.png", bbox_inches="tight", dpi=150)
plt.close()
```

Interpretation:
- **Large gap** between train and val → overfitting (more data or regularization needed)
- **Both scores low** → underfitting (more complex model or better features needed)
- **Scores converge** → good fit, more data won't help much

### 8. Error Analysis
- Examine misclassified samples
- Look for patterns in errors
- Identify potential model weaknesses

### 9. Cross-Validation Stability
- Report mean and std of CV scores
- Check for high variance (potential overfitting)
- Compare train vs validation performance

## Metrics Selection Guide

| Problem Type | Primary Metrics | When to Use |
|-------------|-----------------|-------------|
| Balanced Classification | Accuracy, F1 | Equal class importance |
| Imbalanced Classification | F1, ROC-AUC, PR-AUC | Rare positive class |
| Cost-Sensitive | Custom weighted metrics | Different error costs |
| Regression | RMSE, MAE, R² | Continuous targets |

### Extension Hook Point: after-evaluation

Scan for extension agents with `hooks_into` containing "after-evaluation":
1. Use Glob: `.claude/plugins/*/agents/*.md`, `~/.claude/plugins/*/agents/*.md`
2. Read frontmatter — select agents with `extends: ml-automation` and `hooks_into` including "after-evaluation"
3. Spawn each matching agent with current report context
4. On failure: log warning, continue

## Output Format

Provide:
1. **Metrics Summary Table**: All relevant metrics in one place
2. **Visualizations**: Confusion matrix, ROC curve, feature importance, SHAP summary, calibration curve (classification), learning curve
3. **Interpretation**: What the metrics mean for this specific problem
4. **Recommendations**: Next steps based on evaluation results (including calibration advice and bias/variance diagnosis)

**IMPORTANT**: After evaluation, invoke the `ml-theory-advisor` agent to validate the evaluation methodology and interpret results in context.
