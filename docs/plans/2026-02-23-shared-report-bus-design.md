# V1.2.0 Design: Shared Report Bus — Inter-Agent Communication Layer

**Date:** 2026-02-23
**Version:** 1.2.0
**Status:** Approved

## Problem

Agents in the ml-automation-plugin operate in isolation. The EDA analyst writes a report, but other agents don't systematically discover or build on it. The workflow is mostly sequential with no structured way for agents to communicate findings, recommendations, or next steps. This limits parallelization and creates redundant work.

## Solution: Shared Report Bus

A convention-based JSON report system where every agent writes a standardized report to a shared directory. Agents discover prior work by scanning for `*_report.json` files on startup. No central registry — pure file-based convention.

## 1. Report Schema

Every agent writes a JSON report with this structure:

```json
{
  "agent": "eda-analyst",
  "version": "1.2.0",
  "timestamp": "2026-02-23T14:30:00Z",
  "status": "completed",
  "findings": {
    "summary": "Short narrative of what was found",
    "details": {}
  },
  "recommendations": [
    {
      "action": "what to do",
      "priority": "high|medium|low",
      "target_agent": "feature-engineering-analyst"
    }
  ],
  "next_steps": ["Step 1", "Step 2"],
  "artifacts": ["reports/eda_report.md", "figures/distributions.png"],
  "depends_on": [],
  "enables": ["feature-engineering-analyst", "ml-theory-advisor"]
}
```

**File naming:** `{agent-name}_report.json` (e.g., `eda-analyst_report.json`)

## 2. Directory Structure Per Platform

| Platform   | Report Directory     |
|------------|---------------------|
| Claude Code | `.claude/reports/`  |
| Cursor     | `.cursor/reports/`   |
| Codex      | `.codex/reports/`    |
| OpenCode   | `.opencode/reports/` |
| Fallback   | `reports/`           |

All agents write to the platform-appropriate directory. The `reports/` fallback is always written for cross-platform compatibility.

## 3. Agent Modifications

### Read behavior (all agents)

On startup, every agent scans the reports directory for `*_report.json` files and reads any prior reports to inform its work.

### Write behavior (all agents)

On completion, every agent writes its report using the standard schema.

### Agent-specific changes

| Agent                        | Read                                | Write                              |
|------------------------------|-------------------------------------|-------------------------------------|
| eda-analyst                  | (first agent, nothing to read)      | `eda-analyst_report.json` + legacy `eda_report.json` |
| feature-engineering-analyst  | EDA report                          | `feature-engineering-analyst_report.json` |
| ml-theory-advisor            | ALL prior reports                   | `ml-theory-advisor_report.json`     |
| mlops-engineer               | Evaluation + preprocessing reports  | `mlops-engineer_report.json`        |
| developer                    | Relevant agent reports for context  | `developer_report.json`             |
| brutal-code-reviewer         | Prior reports to validate alignment | `brutal-code-reviewer_report.json`  |
| pr-approver                  | Workflow state for PR context       | `pr-approver_report.json`           |
| frontend-ux-analyst          | All reports for data/model context  | `frontend-ux-analyst_report.json`   |
| orchestrator                 | ALL reports for workflow state       | `orchestrator_report.json`          |
| assigner                     | Reports for smarter assignments     | `assigner_report.json`              |

## 4. New Utility Functions in ml_utils.py

```python
def save_agent_report(agent_name, report_data, output_dirs=None):
    """Save standardized agent report to all platform directories."""

def load_agent_reports(search_dirs=None):
    """Load all agent reports. Returns dict of agent_name -> report."""

def get_workflow_status(search_dirs=None):
    """Summary of which agents have run and what's pending."""
```

## 5. Parallelization Model

### Dependency graph

```
                    ┌─────────────┐
                    │ EDA Analyst  │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              v            v            v
    ┌─────────────┐ ┌───────────┐ ┌──────────────┐
    │ Feature Eng │ │ ML Theory │ │ Frontend UX  │
    └──────┬──────┘ └─────┬─────┘ └──────┬───────┘
           │              │              │
           └──────────────┼──────────────┘
                          v
                  ┌───────────────┐
                  │  Preprocessor  │
                  └───────┬───────┘
                          v
                  ┌───────────────┐
                  │   Training    │
                  └───────┬───────┘
                          v
              ┌───────────┼───────────┐
              v           v           v
    ┌─────────────┐ ┌──────────┐ ┌─────────────┐
    │  Evaluator  │ │ ML Theory│ │ Code Review │
    └──────┬──────┘ └────┬─────┘ └──────┬──────┘
           │             │              │
           └─────────────┼──────────────┘
                         v
                 ┌───────────────┐
                 │   Deployment   │
                 └───────────────┘
```

### Parallel groups

1. **Post-EDA:** feature-engineering-analyst + ml-theory-advisor + frontend-ux-analyst
2. **Post-Training:** evaluator + ml-theory-advisor + brutal-code-reviewer

### Implementation

Team commands (`team-coldstart`, `team-analyze`) spawn agents in parallel groups using the Task tool. Each group waits for dependencies before spawning.

## 6. New /status Command

### Usage

```bash
/status                  # Full workflow status
/status --agent eda      # Specific agent report
/status --pending        # Pending/blocked items only
```

### Output

```
## Workflow Status

### Completed (3/6)
 eda-analyst         - 10,000 rows, 25 columns, 3 quality issues
 feature-engineering - 12 features recommended, 3 interaction terms
 ml-theory-advisor   - No leakage detected, stratified split recommended

### Pending (3/6)
 preprocessing       - Ready to run
 training            - Blocked by: preprocessing
 evaluation          - Blocked by: training

### Cross-Agent Insights
- Feature eng recommends log-transform on Amount -> aligns with EDA skewness
- ML theory flags high cardinality in Region -> feature eng suggests target encoding
```

### New files

- `commands/status.md` — Slash command definition
- `skills/status/status.md` — Skill implementation

## 7. Platform Compatibility

### Files to update

| File | Changes |
|------|---------|
| `.claude-plugin/plugin.json` | Version 1.2.0, updated description |
| `.cursor-plugin/plugin.json` | Version 1.2.0, updated description |
| `.opencode/plugins/ml-automation.js` | Add `status` tool, update report paths |
| `.codex/INSTALL.md` | Version 1.2.0, document /status |
| `hooks/hooks.json` | Add post-agent report validation hook |
| `hooks/cursor-hooks.json` | Mirror hook changes |
| `README.md` | Document v1.2.0 features |

### New hook

Post-agent report validation (`hooks/post-agent-report.sh`):
- Triggered on `SubagentStop` for all agents
- Validates report was written with correct schema
- Warns if report is missing or malformed

### Backward compatibility

- `eda_report.json` (legacy) still written alongside `eda-analyst_report.json`
- `load_eda_report()` checks both old and new paths
- Agents gracefully handle missing reports directory (create on first write)

## 8. Summary of New/Modified Files

### New files
- `commands/status.md`
- `skills/status/status.md`
- `hooks/post-agent-report.sh`

### Modified files
- `templates/ml_utils.py` (3 new functions)
- All 10 agents in `agents/` (add read/write report behavior)
- `commands/team-coldstart.md` (parallel spawning)
- `commands/team-analyze.md` (parallel spawning)
- `.claude-plugin/plugin.json` (version bump)
- `.cursor-plugin/plugin.json` (version bump)
- `.opencode/plugins/ml-automation.js` (add status tool)
- `.codex/INSTALL.md` (version bump)
- `hooks/hooks.json` (new hook)
- `hooks/cursor-hooks.json` (new hook)
- `README.md` (document v1.2.0)
