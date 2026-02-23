---
name: status
description: "Show unified workflow status by reading all agent reports. Displays completed agents, pending work, and cross-agent insights."
user_invocable: true
aliases: ["workflow-status", "agent-status", "report-status"]
---

# Workflow Status

You are displaying the current workflow status by reading all agent reports from the shared report bus.

## How to Display Status

### Step 1: Scan for Agent Reports

Look for `*_report.json` files in these directories (check all that exist):
- `.claude/reports/`
- `.cursor/reports/`
- `.codex/reports/`
- `.opencode/reports/`
- `reports/`

If the project has `ml_utils.py` available, use:

```python
from ml_utils import get_workflow_status, load_agent_reports

status = get_workflow_status()
reports = load_agent_reports()
```

Otherwise, manually scan the directories and read JSON files.

### Step 2: Display Status

Format the output as:

```
## Workflow Status

### Completed ({count}/{total})
✓ {agent-name} — {summary from report}
  Artifacts: {list of artifacts}

### In Progress ({count}/{total})
⟳ {agent-name} — {status details}

### Pending ({count}/{total})
○ {agent-name} — Ready to run | Blocked by: {dependencies}

### Cross-Agent Insights
- {from_agent} recommends: {action} → {target_agent} ({priority})
```

### Step 3: Handle Flags

- `--agent <name>`: Show detailed report for a specific agent (full findings, recommendations, artifacts)
- `--pending`: Show only pending/blocked agents
- `--insights`: Show only cross-agent recommendations

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--agent` | (none) | Show specific agent's full report |
| `--pending` | false | Show only pending items |
| `--insights` | false | Show only cross-agent insights |

## When No Reports Exist

If no reports are found, display:

```
## Workflow Status

No agent reports found. Run a workflow command to get started:
- /eda <data_path> — Start with exploratory data analysis
- /team analyze <data_path> — Quick multi-agent analysis
- /team coldstart <data_path> — Full pipeline from data to deployment
```
