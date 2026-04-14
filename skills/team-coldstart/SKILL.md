---
name: team-coldstart
description: Launch a full data workflow from raw data to deployed solution with interactive dashboard. Orchestrates multiple agents through analysis, processing, modeling, and deployment stages. Works with any tabular data.
---

# Team Cold Start

## When to Use
- Starting a new ML project from raw data
- Running the full pipeline: EDA -> preprocess -> train -> evaluate -> deploy
- Getting a deployed dashboard from scratch

## Workflow

1. Initialize project structure
2. Analyze data (parallel: EDA, ML theory, features)
3. Preprocess and engineer features
4. Train and tune models
5. Evaluate model performance
6. **Generate Streamlit dashboard** (v1.9.0) — Auto-generate `dashboard/app.py` with live inference, EDA visualizations, and model performance panels. Reads real values from report bus artifacts (never uses placeholder data). Includes `dashboard/requirements.txt` with pinned deps. Falls back to a minimal working dashboard if model artifacts are missing.
7. Deploy to target environment
8. Generate completion report

## Report Bus Integration (v1.2.0)

All agents in the workflow write structured reports to the shared report bus. Each agent:
1. Reads prior reports from `.claude/reports/` on startup
2. Writes its own report using `save_agent_report()` on completion

### Parallel Groups

- **Post-EDA**: feature-engineering-analyst + ml-theory-advisor + frontend-ux-analyst
- **Post-Training**: brutal-code-reviewer + ml-theory-advisor + frontend-ux-analyst

Use the Task tool with multiple calls in a single message to spawn parallel agents.

## Reflection Gates (v1.2.1)

Three reflection checkpoints validate outputs before the next stage proceeds:

| Gate | After | Before | Evaluates |
|------|-------|--------|-----------|
| Gate 1 | Feature Engineering | Preprocessing | Feature strategy, domain fit, leakage risk |
| Gate 2 | Preprocessing | Training | Pipeline design, encoding, data flow |
| Gate 3 | Training | Evaluation | Model family, hyperparameters, validation |

Each gate spawns ml-theory-advisor in reflection mode. If verdict is `revise`, the upstream agent re-runs with corrections (max `--max-reflect` iterations, default 2). Set `--max-reflect 0` to skip gates.

## MLOps Registry Layer (v1.3.0)

Convention-based MLOps registries track the full model lifecycle:

| Registry | Agent | Written At |
|----------|-------|-----------|
| Data Versions | eda-analyst | Stage 2a (EDA) |
| Feature Store | feature-engineering-analyst | Stage 2b (Post-EDA) |
| Experiments | developer | Stage 4 (Training) |
| Model Registry | mlops-engineer | Stage 5c (Validation) |

Stage 5c (MLOps Registry Validation) ensures all registries are complete and lineage is traceable from data to deployed model. Use `/registry` to inspect.

## Streamlit Dashboard Generation (v1.9.0)

Stage 6 generates a self-contained Streamlit dashboard at `dashboard/app.py`. The dashboard MUST:

1. **Read real data** — Load metrics, figures, and model artifacts from report bus outputs. Never hardcode placeholder values.
2. **Include these panels**:
   - Dataset overview (row count, column types, target distribution)
   - Key EDA visualizations (correlations, distributions)
   - Model performance metrics (from evaluation report)
   - Feature importance chart (from evaluation or SHAP analysis)
   - Live inference widget — load the trained model and let users input feature values for predictions
3. **Generate requirements.txt** — Pin streamlit, pandas, plotly, joblib, and any model-specific deps.
4. **Validate on generation** — Run syntax check and import validation before marking Stage 6 complete.
5. **Graceful degradation** — If model artifacts are missing (analysis-only mode), show EDA panels only.

Startup: `pip install -r dashboard/requirements.txt && streamlit run dashboard/app.py`

## Self-Check Loops (v1.4.0)

After each stage, run `validate_stage_output()` to verify outputs meet deterministic criteria. If validation fails, the stage re-runs with error context (max iterations configurable).

## Lessons Learned (v1.4.0)

Load lessons from previous workflow runs using `ml_utils.load_lessons()`. Pass relevant lessons to agents as context before each stage. After workflow completion, save new lessons for future runs.

## Full Specification

See `commands/team-coldstart.md` for the complete 9-stage workflow with parallel execution groups, reflection gate configuration, and stage-specific validation rules.
