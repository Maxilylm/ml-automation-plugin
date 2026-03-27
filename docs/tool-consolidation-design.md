# Tool Consolidation Architecture Design

## Problem Statement

Multiple use-cases (Bayesian MMM, generic ML, analysis-only, data engineering) are developed in separate repos. Common procedures (traceability, Kanban, reporting) and generic agents (EDA, reporting) are duplicated. Enhancements that are use-case-agnostic are expensive to propagate across repos.

## Current Architecture

```
ml-automation-plugin/          ← Generic ML automation
├── agents/ (10 agents)
├── commands/ (15 commands)
├── skills/ (15 skills)
└── templates/ml_utils.py

bayesian-mmm-plugin/           ← Hypothetical separate repo
├── agents/ (MMM-specific)
├── commands/ (MMM commands)
└── templates/mmm_utils.py
```

## Proposed Architecture: Mode Registry

Instead of separate repos, use a **single plugin with a mode registry** that activates the right capabilities based on the task type.

### Mode Registry (in plugin.json)

```json
{
  "modes": {
    "generic-ml": {
      "description": "Standard ML pipeline (classification, regression)",
      "agents": ["eda-analyst", "developer", "ml-theory-advisor", ...],
      "stages": ["init", "eda", "preprocessing", "training", "evaluation", "dashboard"],
      "task_types": ["classification", "regression"]
    },
    "bayesian-mmm": {
      "description": "Bayesian Media Mix Modeling",
      "agents": ["eda-analyst", "developer", "mmm-specialist", ...],
      "stages": ["init", "eda", "channel-analysis", "prior-elicitation", "modeling", "decomposition"],
      "task_types": ["mmm"],
      "extra_validations": ["saturation_check", "prior_plausibility", "roas_validation"]
    },
    "analysis-only": {
      "description": "EDA and reporting without model training",
      "agents": ["eda-analyst", "frontend-ux-analyst"],
      "stages": ["init", "eda", "reporting", "dashboard"],
      "task_types": ["analysis"]
    },
    "data-engineering": {
      "description": "Data pipeline and ETL workflows",
      "agents": ["developer", "mlops-engineer"],
      "stages": ["init", "ingestion", "transformation", "validation", "deployment"],
      "task_types": ["etl", "pipeline"]
    }
  }
}
```

### Shared Core (always available)

These components work across all modes:
- **Report Bus** — Agent communication
- **Traceability Log** — Audit trail
- **Lessons Learned** — Cross-run knowledge
- **MLOps Registry** — Model/feature/experiment tracking
- **Reflection Gates** — Pre/post-stage validation
- **Business Plausibility** — Domain validation
- **Methodology Document** — Governance reporting

### Mode-Specific Extensions

Each mode can register:
- Additional agents (e.g., `mmm-specialist`)
- Custom stage validators in `validate_stage_output()`
- Domain-specific plausibility rules
- Extra commands and skills

### Mode Detection

Auto-detect mode from data characteristics:
```python
def detect_mode(df, target_col=None, user_hint=None):
    if user_hint:
        return user_hint

    col_types = detect_column_types(df, target_col)

    # MMM signals: media spend columns, date column, KPI target
    media_keywords = ['spend', 'impressions', 'clicks', 'grp', 'cost']
    has_media = any(any(kw in c.lower() for kw in media_keywords) for c in df.columns)
    has_date = len(col_types['datetime']) > 0

    if has_media and has_date:
        return 'bayesian-mmm'
    elif target_col:
        return 'generic-ml'
    else:
        return 'analysis-only'
```

### Migration Path

1. **Phase 1** (current): Add mode registry to existing plugin, generic-ml mode as default
2. **Phase 2**: Add bayesian-mmm mode with MMM-specific agents and validators
3. **Phase 3**: Factor out shared core into a base plugin package
4. **Phase 4**: Each mode becomes an installable extension

### Benefits

- Single repo for common infrastructure
- Enhancements propagate automatically to all modes
- Consistent governance (traceability, methodology docs) across use-cases
- Mode-specific customization without code duplication
- Easy to add new modes without touching core

### Risks

- Plugin grows large — mitigate with lazy loading and mode-gated file discovery
- Mode detection may be wrong — always allow user override via `--mode` flag
- Mode-specific agents may conflict — namespace agents per mode if needed
