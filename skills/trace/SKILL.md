---
name: trace
description: View and filter the traceability log — a chronological audit trail of all agent actions, decisions, and stage transitions during workflow execution.
---

# Traceability Log

## When to Use
- Monitoring what agents did during a workflow
- Auditing decision-making and agent behavior
- Debugging unexpected workflow outcomes
- Reviewing stage transitions and timing

## Workflow

1. **Load Log** — Read traceability-log.jsonl from platform directories
2. **Apply Filters** — Filter by event type, agent, stage, command, or time range
3. **Format Output** — Display chronological timeline with icons and details
4. **Export** — Optionally export filtered log as JSON

## Output Format

Chronological event timeline with icons for event types (agent_spawn, agent_complete, decision, reflection, validation, error). Verbose mode includes inputs, outputs, and duration.

## Report Bus Integration (v1.8.0)

```python
from ml_utils import load_trace_log, format_trace_log
entries = load_trace_log(filters={"agent": "eda-analyst", "limit": 20})
print(format_trace_log(entries, verbose=True))
```

## Full Specification

See `commands/trace.md` for complete viewer options and event type reference.
