---
name: methodology
description: Generate a stakeholder-facing analytical approach document outlining end-to-end methodology, statistical methods, assumptions, decision points, and artefacts for reproducibility and review.
---

# Analytical Approach Document

## When to Use
- After completing a full analysis pipeline
- When stakeholders need a methods overview for review
- For governance documentation and audit requirements
- When reproducibility documentation is needed

## Workflow

1. **Gather Context** — Read all agent reports, EDA report, experiments, model registry, feature store, data versions, lessons learned, and traceability log
2. **Extract Decisions** — Identify key decision points from trace log and reflection gate outcomes
3. **Compile Methods** — Document statistical methods, preprocessing approach, model selection rationale from experiment tracking
4. **Document Assumptions** — List distributional, independence, and domain assumptions
5. **Generate Document** — Write structured Markdown with 12 sections (objective through appendix)
6. **Save Artefacts** — Save to `reports/methodology_{timestamp}.md` and write agent report

## Output Format

Structured Markdown document covering: objective, data description, EDA summary, feature engineering, statistical methods, assumptions, decision points, model results, intermediate artefacts, reproducibility guide, lessons learned, and appendix.

## Report Bus Integration (v1.8.0)

```python
from ml_utils import save_agent_report, log_trace_event
save_agent_report("methodology-generator", {
    "status": "completed",
    "findings": {"summary": "Methodology document generated", "details": {...}},
    "artifacts": ["reports/methodology_{timestamp}.md"]
})
```

## Full Specification

See `commands/methodology.md` for complete document structure and data sources.
