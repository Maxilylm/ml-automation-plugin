---
name: status
description: "Show unified workflow status by reading all agent reports. Displays completed agents, pending work, and cross-agent insights."
---

# Workflow Status

Read and display the current state of all agent reports in the shared report bus.

## When to Use
- Checking what agents have completed work
- Understanding workflow progress and dependencies
- Viewing cross-agent recommendations and lessons learned

## Implementation

Scan all report directories for `*_report.json` files and present a unified view:

1. **Completed agents** — Name, status, key findings summary
2. **Pending agents** — What's waiting, what dependencies are unmet
3. **Cross-agent insights** — Recommendations with `target_agent` fields
4. **Lessons learned** — Load via `ml_utils.load_lessons()` if available

Use `ml_utils.get_workflow_status()` if available, otherwise scan directories manually.

## Flags

| Flag | Description |
|------|-------------|
| `--agent <name>` | Show details for a specific agent's report |
| `--pending` | Show only pending/incomplete work |
| `--insights` | Show only cross-agent recommendations |
| `--lessons` | Show only lessons learned entries |

## Output Format

```
## Workflow Status

| Agent | Status | Key Finding |
|-------|--------|------------|
| eda-analyst | completed | 500 rows, 12 features, 3 quality issues |
| ml-theory-advisor | completed | Approved feature strategy |
| developer | in_progress | Training pipeline (PR #4) |
| mlops-engineer | pending | Waiting for model artifact |
```

## Fallback

If `ml_utils` is not available, scan these directories:
- `.claude/reports/`, `.cursor/reports/`, `.codex/reports/`, `reports/`

Parse each `*_report.json` and extract `status`, `findings.summary`, and `recommendations`.

## Full Specification

See `commands/status.md` for complete output templates and flag handling.
