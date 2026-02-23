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
6. Build interactive dashboard
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
