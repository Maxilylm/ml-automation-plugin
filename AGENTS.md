# spark-core — Cortex Code Extension

End-to-end ML automation workflow. Orchestrates 10 specialized agents across EDA, preprocessing, training, evaluation, deployment, and reporting. This is the base plugin — all other spark extensions depend on it.

## Available Agents

| Agent | When to use |
|---|---|
| `eda-analyst` | User wants to explore a dataset, check distributions, correlations, data quality, or run any exploratory analysis |
| `ml-theory-advisor` | User asks about feature engineering strategy, data leakage prevention, model selection, or ML methodology |
| `feature-engineering-analyst` | User wants to discover or design features, transform variables, or identify feature opportunities |
| `mlops-engineer` | User wants to deploy a model, create an API, containerize with Docker, or set up CI/CD |
| `developer` | User asks to implement or fix code on a feature branch |
| `brutal-code-reviewer` | User wants a code quality or maintainability review |
| `pr-approver` | User wants to review and merge a pull request |
| `frontend-ux-analyst` | User wants feedback on a dashboard or UI design |
| `orchestrator` | User requests a multi-agent workflow or full pipeline run |
| `assigner` | Route an ambiguous request to the right specialist agent |

## Available Skills

| Skill | Trigger |
|---|---|
| `/eda` | "explore this dataset", "run EDA on", "check data quality", "what's in this file" |
| `/preprocess` | "preprocess the data", "build a preprocessing pipeline", "clean this data" |
| `/train` | "train a model", "fit a classifier", "build a regression model" |
| `/evaluate` | "evaluate the model", "show me metrics", "how good is the model" |
| `/deploy` | "deploy the model", "create an API", "dockerize", "deploy to production" |
| `/report` | "generate a report", "summarize results", "create an EDA report" |
| `/test` | "write tests", "generate unit tests", "check test coverage" |
| `/team-coldstart` | "full pipeline", "end to end from this CSV", "raw data to deployed model" |
| `/team-analyze` | "quick multi-agent analysis", "analyze this dataset with multiple agents" |
| `/team-review` | "multi-agent code review", "review this PR with multiple agents" |
| `/status` | "show workflow status", "what have the agents done", "show reports" |
| `/registry` | "show model registry", "list experiments", "show feature registry" |
| `/trace` | "show traceability log", "audit agent actions", "what did the agents do", "show workflow timeline" |

## Routing

- Data exploration / profiling → `eda-analyst`
- ML theory questions → `ml-theory-advisor`
- Feature design → `feature-engineering-analyst`
- Deployment / infrastructure → `mlops-engineer`
- Code implementation → `developer`
- Code review → `brutal-code-reviewer`
- PR merging → `pr-approver`
- Dashboard feedback → `frontend-ux-analyst`
- Multi-step orchestration → `orchestrator`
- Ambiguous routing → `assigner`
