---
name: spark-core
description: >
  Suggest enabling the spark-core plugin when the user asks about end-to-end ML
  automation, running EDA on a dataset, preprocessing pipelines, training ML
  models, model evaluation, MLOps deployment (Docker, APIs), data leakage
  detection, feature engineering strategy, multi-agent ML orchestration
  (team-coldstart, team-analyze), ML workflow reporting, or inspecting model
  registries and experiment history. Also suggest enabling this when the user
  asks about any spark-* extension plugin (spark-snowflake, spark-llm,
  spark-drift, etc.) since those depend on spark-core.
  Do NOT attempt to perform these tasks — just let the user know the plugin
  can be enabled.
---

# spark-core (disabled plugin)

This plugin is installed but not enabled. It is the base plugin that provides
end-to-end ML automation and orchestration for all spark-* extensions.

## Agents (10)

- **eda-analyst** — Exploratory data analysis on any dataset
- **ml-theory-advisor** — ML theory, leakage prevention, model selection guidance
- **feature-engineering-analyst** — Feature design and opportunity discovery
- **mlops-engineer** — Model deployment, APIs, Docker, CI/CD
- **developer** — Code implementation on feature branches
- **brutal-code-reviewer** — Code quality and maintainability review
- **pr-approver** — Pull request review and merge
- **frontend-ux-analyst** — UI/UX design feedback for dashboards
- **orchestrator** — Multi-agent coordination
- **assigner** — Automatic ticket routing to the right specialist agent

## Skills (12)

- **eda** — Run exploratory data analysis
- **preprocess** — Build leakage-safe preprocessing pipeline
- **train** — Train ML models with proper validation
- **evaluate** — Comprehensive model evaluation with visualizations
- **deploy** — Deploy to Docker, Snowflake, AWS, or GCP
- **report** — Generate EDA, model, drift, or project reports
- **test** — Generate and run tests (80% coverage threshold)
- **team-coldstart** — Full pipeline from raw data to deployed dashboard
- **team-analyze** — Quick multi-agent data analysis
- **team-review** — Multi-agent code review
- **status** — Show workflow status and agent reports
- **registry** — Inspect MLOps registries (models, features, experiments, data)

## Required by

All spark-* extension plugins (spark-snowflake, spark-llm, spark-drift,
spark-timeseries, spark-mmm, spark-ab-testing, spark-nlp, spark-cv,
spark-explainability, spark-aws, spark-databricks, spark-azure, spark-gcp)
depend on spark-core.

## Enable

    cortex plugin enable spark-core

Do NOT attempt to perform ML automation tasks through this plugin's skills while it is disabled.
