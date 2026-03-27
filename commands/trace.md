---
name: trace
description: "View and filter the traceability log — a chronological audit trail of all agent actions, decisions, and stage transitions during workflow execution."
user_invocable: true
aliases: ["traceability", "audit-log", "activity-log"]
---

# Traceability Log Viewer

You are displaying the traceability log — a chronological record of all processes, agents called, and descriptions of what each was asked to do and did.

## Purpose

The traceability log provides:
- **Auditability** — what the agent system did and why
- **Reproducibility** — step-by-step record for re-running analysis
- **Monitoring** — detect undesirable changes or unexpected agent behavior
- **Improvement** — input for assessing and improving the system

## How to Display

### Step 1: Load the Trace Log

```python
from ml_utils import load_trace_log, format_trace_log

# Load all events (newest first)
entries = load_trace_log()

# Or with filters:
entries = load_trace_log(filters={
    "event_type": "agent_complete",    # or list: ["agent_spawn", "agent_complete"]
    "agent": "eda-analyst",             # filter by specific agent
    "stage": "analysis",                # filter by pipeline stage
    "command": "team-coldstart",        # filter by originating command
    "since": "2024-01-15T00:00:00",    # events after this timestamp
    "limit": 50,                        # max entries
})

# Format for display
print(format_trace_log(entries, verbose=True))
```

### Step 2: Display Format

```
## Traceability Log

### Summary
- Total events: {count}
- Time range: {first_event} → {last_event}
- Agents involved: {unique_agents}
- Stages covered: {unique_stages}

### Event Timeline (newest first)

→ [2024-01-15 14:32:01] agent_spawn | eda-analyst | stage:analysis — Starting EDA on sales_data.csv
    IN:  data/raw/sales_data.csv (500 rows, 8 columns)

✓ [2024-01-15 14:33:45] agent_complete | eda-analyst | stage:analysis — EDA completed
    OUT: .claude/eda_report.json (3 quality issues, 2 high correlations)
    Duration: 104000ms

◆ [2024-01-15 14:33:46] decision | ml-theory-advisor | stage:analysis — Selected stratified split
    Details: 80/20 split, stratified on target due to class imbalance (38%)

◈ [2024-01-15 14:34:00] reflection | ml-theory-advisor | stage:post-feature-engineering — Approved
    Details: Feature set validated, no leakage risks identified

✔ [2024-01-15 14:34:15] validation | orchestrator | stage:preprocessing — Self-check passed
    Details: validate_stage_output('preprocessing') → passed

✗ [2024-01-15 14:35:00] error | developer | stage:training — Model training failed
    Details: ValueError: Target column has NaN values
```

### Step 3: Handle Flags

- `--agent <name>`: Show only events for a specific agent
- `--stage <name>`: Show only events for a specific pipeline stage
- `--type <event_type>`: Filter by event type (agent_spawn, agent_complete, decision, reflection, validation, error)
- `--since <timestamp>`: Show events after a timestamp
- `--limit <n>`: Show last N events
- `--verbose`: Include inputs/outputs/duration details
- `--export <path>`: Export filtered log as JSON

## Event Types

| Type | Icon | Description |
|------|------|-------------|
| `agent_spawn` | → | Agent was launched |
| `agent_complete` | ✓ | Agent finished successfully |
| `stage_start` | ▶ | Pipeline stage began |
| `stage_end` | ■ | Pipeline stage completed |
| `decision` | ◆ | A methodological decision was made |
| `reflection` | ◈ | Reflection gate evaluation |
| `validation` | ✔ | Self-check or validation ran |
| `error` | ✗ | An error occurred |
| `user_action` | ● | User-initiated action |

## Log Location

Traceability logs are stored as JSONL (one JSON object per line) in:
- `.claude/traceability-log.jsonl`
- `.cursor/traceability-log.jsonl`
- `.codex/traceability-log.jsonl`
- `.opencode/traceability-log.jsonl`

## When No Log Exists

If no traceability log is found:
```
## Traceability Log

No trace events recorded yet. Traceability logging is automatically enabled
when running workflow commands (/team-coldstart, /team-analyze, etc.).

To manually log an event:
from ml_utils import log_trace_event
log_trace_event({"event_type": "user_action", "action": "Started analysis"})
```
