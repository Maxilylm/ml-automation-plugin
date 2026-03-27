---
name: feedback
description: Collect structured feedback from test users with severity, category, and context for system improvement.
---

# Feedback Collection

## When to Use
- Test users want to report issues or suggestions
- After completing a workflow and reviewing results
- When usability, performance, or quality issues are noticed

## Workflow

1. **Categorize** — Ask user for feedback type (bug, improvement, feature, usability, performance, quality)
2. **Gather Details** — Collect description, command involved, steps to reproduce, expected vs actual behavior
3. **Rate Severity** — Critical, high, medium, or low impact
4. **Save** — Record to report bus and traceability log
5. **Confirm** — Display summary of recorded feedback

## Output Format

Structured feedback saved as agent report with category, severity, title, description, and context.

## Report Bus Integration (v1.8.0)

```python
from ml_utils import save_agent_report
save_agent_report("feedback-{timestamp}", {
    "status": "completed",
    "findings": {"summary": "User feedback: {category}", "details": {...}},
})
```

## Full Specification

See `commands/feedback.md` for complete feedback form and viewing instructions.
