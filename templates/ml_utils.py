"""
ml_utils.py — Reusable ML utilities for the ml-automation plugin.

Agents and skills should reference this file instead of regenerating common code.
Copy this file into your project's src/ directory at pipeline initialization.

Usage:
    from ml_utils import load_data, detect_column_types, build_preprocessor, evaluate_model
"""

import json
import os
from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_data(path, **kwargs):
    """Load tabular data from CSV, Excel, JSON, or Parquet."""
    path = Path(path)
    loaders = {
        ".csv": pd.read_csv,
        ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".feather": pd.read_feather,
    }
    loader = loaders.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    return loader(path, **kwargs)


# =============================================================================
# 2. COLUMN TYPE DETECTION
# =============================================================================

def detect_column_types(df, target_col=None, id_threshold=0.95, cat_threshold=20):
    """
    Classify columns into numerical, categorical, datetime, text, and ID.

    Args:
        df: Input DataFrame
        target_col: Name of target column (excluded from feature lists)
        id_threshold: Uniqueness ratio above which a column is flagged as ID-like
        cat_threshold: Max unique values for a numeric column to be treated as categorical

    Returns:
        dict with keys: numerical, categorical, datetime, text, id_like, target
    """
    result = {
        "numerical": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "id_like": [],
        "target": target_col,
    }

    for col in df.columns:
        if col == target_col:
            continue

        dtype = df[col].dtype
        n_unique = df[col].nunique()
        uniqueness_ratio = n_unique / len(df) if len(df) > 0 else 0

        # ID-like detection
        if uniqueness_ratio > id_threshold and dtype == "object":
            result["id_like"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            result["datetime"].append(col)
        elif dtype == "object":
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > 50:
                result["text"].append(col)
            else:
                result["categorical"].append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if n_unique <= cat_threshold:
                result["categorical"].append(col)
            else:
                result["numerical"].append(col)
        else:
            result["categorical"].append(col)

    return result


# =============================================================================
# 3. PREPROCESSING PIPELINE BUILDER
# =============================================================================

def build_preprocessor(numerical_cols, categorical_cols):
    """
    Build a sklearn ColumnTransformer with standard preprocessing.

    Returns a fitted-ready ColumnTransformer. Fit on TRAINING data only.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numerical_cols:
        transformers.append(("num", num_pipeline, numerical_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers)


# =============================================================================
# 4. TRAIN-TEST SPLIT (LEAKAGE-SAFE)
# =============================================================================

def safe_split(df, target_col, test_size=0.2, random_state=42):
    """
    Split data into train/test BEFORE any preprocessing.

    For classification targets, uses stratified split.
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify = y if y.nunique() <= 20 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# 5. MODEL EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, problem_type="auto"):
    """
    Evaluate a model and return a metrics dictionary.

    Args:
        model: Fitted sklearn-compatible model
        X_test: Test features
        y_test: Test labels
        problem_type: 'classification', 'regression', or 'auto'

    Returns:
        dict of metric_name -> value
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score,
    )

    y_pred = model.predict(X_test)

    if problem_type == "auto":
        problem_type = "classification" if y_test.nunique() <= 20 else "regression"

    metrics = {}

    if problem_type == "classification":
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        avg = "binary" if y_test.nunique() == 2 else "weighted"
        metrics["precision"] = precision_score(y_test, y_pred, average=avg, zero_division=0)
        metrics["recall"] = recall_score(y_test, y_pred, average=avg, zero_division=0)
        metrics["f1"] = f1_score(y_test, y_pred, average=avg, zero_division=0)
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                if y_test.nunique() == 2:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    metrics["roc_auc"] = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
            except Exception:
                pass
    else:
        metrics["rmse"] = mean_squared_error(y_test, y_pred, squared=False)
        metrics["mae"] = mean_absolute_error(y_test, y_pred)
        metrics["r2"] = r2_score(y_test, y_pred)

    return metrics


# =============================================================================
# 6. EDA REPORT I/O
# =============================================================================

def save_eda_report(report_data, output_dir=".claude"):
    """
    Save structured EDA report as JSON for downstream agents.
    Also saves in the new agent report bus format for v1.2.0+ compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eda_report.json")
    with open(path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    bus_report = {
        "status": "completed",
        "findings": {
            "summary": f"EDA completed: {report_data.get('shape', {}).get('rows', '?')} rows, {report_data.get('shape', {}).get('cols', '?')} columns",
            "details": report_data,
        },
        "recommendations": [],
        "next_steps": ["Run feature engineering", "Run ML theory review"],
        "artifacts": [path],
        "depends_on": [],
        "enables": ["feature-engineering-analyst", "ml-theory-advisor", "frontend-ux-analyst"],
    }
    for issue in report_data.get("quality_issues", []):
        bus_report["recommendations"].append({
            "action": f"Address {issue.get('issue', 'unknown')} in column {issue.get('column', '?')}",
            "priority": issue.get("severity", "medium"),
            "target_agent": "feature-engineering-analyst",
        })

    save_agent_report("eda-analyst", bus_report)
    return path


def load_eda_report(search_dirs=None):
    """
    Load prior EDA report if it exists.
    Checks both legacy .claude/eda_report.json and new bus format.
    """
    if search_dirs is None:
        search_dirs = [".claude", "reports", ".claude/reports", ".cursor/reports"]

    for d in search_dirs:
        path = os.path.join(d, "eda_report.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    reports = load_agent_reports(search_dirs)
    eda = reports.get("eda-analyst")
    if eda:
        return eda.get("findings", {}).get("details", eda.get("findings", {}))

    return None


def generate_eda_summary(df, target_col=None):
    """
    Generate a structured EDA summary dict suitable for saving and passing to agents.
    """
    col_types = detect_column_types(df, target_col=target_col)

    missing = {
        col: {"count": int(df[col].isnull().sum()), "pct": round(df[col].isnull().mean() * 100, 2)}
        for col in df.columns if df[col].isnull().sum() > 0
    }

    num_stats = {}
    for col in col_types["numerical"]:
        num_stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "median": round(float(df[col].median()), 4),
            "std": round(float(df[col].std()), 4),
            "min": round(float(df[col].min()), 4),
            "max": round(float(df[col].max()), 4),
            "skew": round(float(df[col].skew()), 4),
        }

    cat_stats = {}
    for col in col_types["categorical"]:
        cat_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": df[col].value_counts().head(5).to_dict(),
        }

    # Correlation pairs > 0.8
    num_df = df[col_types["numerical"]]
    high_corr = []
    if len(num_df.columns) > 1:
        corr_matrix = num_df.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.8:
                    high_corr.append({
                        "col_a": corr_matrix.columns[i],
                        "col_b": corr_matrix.columns[j],
                        "correlation": round(float(val), 4),
                    })

    return {
        "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
        "column_types": col_types,
        "missing_values": missing,
        "numerical_stats": num_stats,
        "categorical_stats": cat_stats,
        "high_correlations": high_corr,
        "target": target_col,
        "quality_issues": _detect_quality_issues(df, col_types),
    }


def _detect_quality_issues(df, col_types):
    """Detect common data quality red flags."""
    issues = []

    for col in col_types["numerical"]:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if series.nunique() == 1:
            issues.append({"column": col, "issue": "constant_value", "severity": "high"})
        elif abs(series.skew()) > 3:
            issues.append({"column": col, "issue": "highly_skewed", "severity": "medium", "skew": round(float(series.skew()), 2)})

    for col in col_types["categorical"]:
        series = df[col].dropna()
        if series.nunique() == 1:
            issues.append({"column": col, "issue": "constant_value", "severity": "high"})
        if series.nunique() > 100:
            issues.append({"column": col, "issue": "high_cardinality", "severity": "medium", "n_unique": int(series.nunique())})

    for col in df.columns:
        pct_missing = df[col].isnull().mean()
        if pct_missing > 0.5:
            issues.append({"column": col, "issue": "majority_missing", "severity": "high", "pct": round(pct_missing * 100, 1)})

    return issues


# =============================================================================
# 7. AGENT REPORT BUS
# =============================================================================

REPORT_SCHEMA_VERSION = "1.2.0"

PLATFORM_REPORT_DIRS = [".claude/reports", ".cursor/reports", ".codex/reports", ".opencode/reports", "reports"]


def save_agent_report(agent_name, report_data, output_dirs=None):
    """
    Save a standardized agent report to all platform report directories.

    Args:
        agent_name: The agent identifier (e.g., 'eda-analyst')
        report_data: Dict with keys: findings, recommendations, next_steps, artifacts
        output_dirs: List of directories to write to (defaults to all platform dirs)
    """
    from datetime import datetime, timezone

    if output_dirs is None:
        output_dirs = PLATFORM_REPORT_DIRS

    report = {
        "agent": agent_name,
        "version": REPORT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": report_data.get("status", "completed"),
        "findings": report_data.get("findings", {}),
        "recommendations": report_data.get("recommendations", []),
        "next_steps": report_data.get("next_steps", []),
        "artifacts": report_data.get("artifacts", []),
        "depends_on": report_data.get("depends_on", []),
        "enables": report_data.get("enables", []),
    }

    filename = f"{agent_name}_report.json"
    paths_written = []

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        paths_written.append(path)

    return paths_written


def load_agent_reports(search_dirs=None):
    """
    Load all agent reports from report directories.

    Returns:
        dict of agent_name -> report_dict (most recent per agent)
    """
    import glob as globmod

    if search_dirs is None:
        search_dirs = PLATFORM_REPORT_DIRS

    reports = {}
    for d in search_dirs:
        pattern = os.path.join(d, "*_report.json")
        for filepath in globmod.glob(pattern):
            try:
                with open(filepath) as f:
                    report = json.load(f)
                agent = report.get("agent", os.path.basename(filepath).replace("_report.json", ""))
                if agent not in reports or report.get("timestamp", "") > reports[agent].get("timestamp", ""):
                    reports[agent] = report
            except (json.JSONDecodeError, KeyError):
                continue
    return reports


def get_workflow_status(search_dirs=None):
    """
    Get a summary of workflow status from agent reports.

    Returns:
        dict with keys: completed (list), pending (list), insights (list)
    """
    reports = load_agent_reports(search_dirs)

    workflow_agents = [
        "eda-analyst", "feature-engineering-analyst", "ml-theory-advisor",
        "frontend-ux-analyst", "developer", "brutal-code-reviewer",
        "pr-approver", "mlops-engineer", "orchestrator", "assigner",
    ]

    completed = []
    for agent_name, report in reports.items():
        summary = report.get("findings", {}).get("summary", "No summary")
        completed.append({"agent": agent_name, "summary": summary, "timestamp": report.get("timestamp", "")})

    completed_names = set(reports.keys())
    pending = [a for a in workflow_agents if a not in completed_names]

    insights = []
    for agent_name, report in reports.items():
        for rec in report.get("recommendations", []):
            if rec.get("target_agent") and rec.get("target_agent") in completed_names:
                insights.append({
                    "from": agent_name,
                    "to": rec["target_agent"],
                    "action": rec.get("action", ""),
                    "priority": rec.get("priority", "medium"),
                })

    return {"completed": completed, "pending": pending, "insights": insights}


# =============================================================================
# 8. REFLECTION REPORTS
# =============================================================================

REFLECTION_GATES = ["post-feature-engineering", "post-preprocessing", "post-training"]


def save_reflection_report(gate, report_data, output_dirs=None):
    """
    Save a reflection gate report from ml-theory-advisor.

    Args:
        gate: One of 'post-feature-engineering', 'post-preprocessing', 'post-training'
        report_data: Dict with keys: verdict, reasoning, corrections
        output_dirs: List of directories (defaults to PLATFORM_REPORT_DIRS)
    """
    from datetime import datetime, timezone

    if output_dirs is None:
        output_dirs = PLATFORM_REPORT_DIRS

    report = {
        "agent": "ml-theory-advisor",
        "version": REPORT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "findings": {
            "summary": f"Reflection gate: {gate}",
            "details": {
                "gate": gate,
                "verdict": report_data.get("verdict", "approved"),
                "reasoning": report_data.get("reasoning", ""),
                "corrections": report_data.get("corrections", []),
            },
        },
        "recommendations": report_data.get("corrections", []),
        "next_steps": [],
        "artifacts": [],
        "depends_on": [],
        "enables": [],
    }

    filename = f"ml-theory-advisor_reflection_{gate}_report.json"
    paths_written = []

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        paths_written.append(path)

    return paths_written


def load_reflection_report(gate, search_dirs=None):
    """
    Load the most recent reflection report for a specific gate.

    Args:
        gate: One of 'post-feature-engineering', 'post-preprocessing', 'post-training'
        search_dirs: Directories to search (defaults to PLATFORM_REPORT_DIRS)

    Returns:
        dict with verdict, reasoning, corrections — or None if not found
    """
    if search_dirs is None:
        search_dirs = PLATFORM_REPORT_DIRS

    filename = f"ml-theory-advisor_reflection_{gate}_report.json"
    latest = None

    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    report = json.load(f)
                if latest is None or report.get("timestamp", "") > latest.get("timestamp", ""):
                    latest = report
            except (json.JSONDecodeError, KeyError):
                continue

    if latest:
        return latest.get("findings", {}).get("details", {})
    return None
