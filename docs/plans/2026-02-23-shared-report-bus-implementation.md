# Shared Report Bus v1.2.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a convention-based JSON report system enabling all 10 agents to share findings, recommendations, and next steps — with parallel execution groups and a new /status command.

**Architecture:** Every agent reads `*_report.json` files from a shared reports directory on startup and writes its own report on completion. Team commands spawn agents in parallel groups based on a dependency graph. A new `/status` command provides unified workflow visibility.

**Tech Stack:** JSON reports, Bash hooks, Markdown agent/command/skill definitions, JavaScript (OpenCode plugin)

---

### Task 1: Add report utility functions to ml_utils.py

**Files:**
- Modify: `templates/ml_utils.py:206-240` (after existing EDA report functions)

**Step 1: Add the three new utility functions after the existing EDA functions**

Add after line 240 (end of `generate_eda_summary`):

```python
# =============================================================================
# 7. AGENT REPORT BUS
# =============================================================================

REPORT_SCHEMA_VERSION = "1.2.0"

PLATFORM_REPORT_DIRS = [".claude/reports", ".cursor/reports", ".codex/reports", ".opencode/reports", "reports"]


def save_agent_report(agent_name, report_data, output_dirs=None):
    """
    Save a standardized agent report to all platform report directories.

    Args:
        agent_name: The agent identifier (e.g., 'eda-analyst')
        report_data: Dict with keys: findings, recommendations, next_steps, artifacts
        output_dirs: List of directories to write to (defaults to all platform dirs)
    """
    from datetime import datetime, timezone

    if output_dirs is None:
        output_dirs = PLATFORM_REPORT_DIRS

    report = {
        "agent": agent_name,
        "version": REPORT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": report_data.get("status", "completed"),
        "findings": report_data.get("findings", {}),
        "recommendations": report_data.get("recommendations", []),
        "next_steps": report_data.get("next_steps", []),
        "artifacts": report_data.get("artifacts", []),
        "depends_on": report_data.get("depends_on", []),
        "enables": report_data.get("enables", []),
    }

    filename = f"{agent_name}_report.json"
    paths_written = []

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        paths_written.append(path)

    return paths_written


def load_agent_reports(search_dirs=None):
    """
    Load all agent reports from report directories.

    Returns:
        dict of agent_name -> report_dict (most recent per agent)
    """
    import glob as globmod

    if search_dirs is None:
        search_dirs = PLATFORM_REPORT_DIRS

    reports = {}
    for d in search_dirs:
        pattern = os.path.join(d, "*_report.json")
        for filepath in globmod.glob(pattern):
            try:
                with open(filepath) as f:
                    report = json.load(f)
                agent = report.get("agent", os.path.basename(filepath).replace("_report.json", ""))
                # Keep most recent by timestamp
                if agent not in reports or report.get("timestamp", "") > reports[agent].get("timestamp", ""):
                    reports[agent] = report
            except (json.JSONDecodeError, KeyError):
                continue
    return reports


def get_workflow_status(search_dirs=None):
    """
    Get a summary of workflow status from agent reports.

    Returns:
        dict with keys: completed (list), pending (list), insights (list)
    """
    reports = load_agent_reports(search_dirs)

    # All known workflow agents in execution order
    workflow_agents = [
        "eda-analyst", "feature-engineering-analyst", "ml-theory-advisor",
        "frontend-ux-analyst", "developer", "brutal-code-reviewer",
        "pr-approver", "mlops-engineer", "orchestrator", "assigner",
    ]

    completed = []
    for agent_name, report in reports.items():
        summary = report.get("findings", {}).get("summary", "No summary")
        completed.append({"agent": agent_name, "summary": summary, "timestamp": report.get("timestamp", "")})

    completed_names = set(reports.keys())
    pending = [a for a in workflow_agents if a not in completed_names]

    # Cross-agent insights: collect recommendations targeting other agents
    insights = []
    for agent_name, report in reports.items():
        for rec in report.get("recommendations", []):
            if rec.get("target_agent") and rec.get("target_agent") in completed_names:
                insights.append({
                    "from": agent_name,
                    "to": rec["target_agent"],
                    "action": rec.get("action", ""),
                    "priority": rec.get("priority", "medium"),
                })

    return {"completed": completed, "pending": pending, "insights": insights}
```

**Step 2: Update `save_eda_report` to also write bus-format report**

Modify the existing `save_eda_report` function at line 210 to add backward-compat writing:

```python
def save_eda_report(report_data, output_dir=".claude"):
    """
    Save structured EDA report as JSON for downstream agents.
    Also saves in the new agent report bus format for v1.2.0+ compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Legacy path
    path = os.path.join(output_dir, "eda_report.json")
    with open(path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    # Also save as bus-format report
    bus_report = {
        "status": "completed",
        "findings": {
            "summary": f"EDA completed: {report_data.get('shape', {}).get('rows', '?')} rows, {report_data.get('shape', {}).get('cols', '?')} columns",
            "details": report_data,
        },
        "recommendations": [],
        "next_steps": ["Run feature engineering", "Run ML theory review"],
        "artifacts": [path],
        "depends_on": [],
        "enables": ["feature-engineering-analyst", "ml-theory-advisor", "frontend-ux-analyst"],
    }
    # Add recommendations from quality issues
    for issue in report_data.get("quality_issues", []):
        bus_report["recommendations"].append({
            "action": f"Address {issue.get('issue', 'unknown')} in column {issue.get('column', '?')}",
            "priority": issue.get("severity", "medium"),
            "target_agent": "feature-engineering-analyst",
        })

    save_agent_report("eda-analyst", bus_report)
    return path
```

**Step 3: Update `load_eda_report` to also check bus format**

Modify `load_eda_report` at line 225:

```python
def load_eda_report(search_dirs=None):
    """
    Load prior EDA report if it exists.
    Checks both legacy .claude/eda_report.json and new bus format.
    """
    if search_dirs is None:
        search_dirs = [".claude", "reports", ".claude/reports", ".cursor/reports"]

    # Check legacy paths first
    for d in search_dirs:
        path = os.path.join(d, "eda_report.json")
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)

    # Check bus format
    reports = load_agent_reports(search_dirs)
    eda = reports.get("eda-analyst")
    if eda:
        return eda.get("findings", {}).get("details", eda.get("findings", {}))

    return None
```

**Step 4: Commit**

```bash
git add templates/ml_utils.py
git commit -m "feat: add agent report bus utilities to ml_utils.py

Adds save_agent_report(), load_agent_reports(), get_workflow_status()
for inter-agent communication. Updates save_eda_report() and
load_eda_report() for backward compatibility with bus format."
```

---

### Task 2: Update all 10 agent system prompts with report read/write behavior

**Files:**
- Modify: `agents/eda-analyst.md`
- Modify: `agents/feature-engineering-analyst.md`
- Modify: `agents/ml-theory-advisor.md`
- Modify: `agents/mlops-engineer.md`
- Modify: `agents/developer.md`
- Modify: `agents/brutal-code-reviewer.md`
- Modify: `agents/pr-approver.md`
- Modify: `agents/frontend-ux-analyst.md`
- Modify: `agents/orchestrator.md`
- Modify: `agents/assigner.md`

**Step 1: Add report bus section to eda-analyst.md**

Add before the final line (line 131) of `agents/eda-analyst.md`:

```markdown
## Agent Report Bus (v1.2.0)

### On Completion — Write Report

When you complete your analysis, write a structured report using the project's `ml_utils.py`:

```python
from ml_utils import save_agent_report

save_agent_report("eda-analyst", {
    "status": "completed",
    "findings": {
        "summary": "Brief narrative of EDA findings",
        "details": {
            "shape": {"rows": N, "cols": M},
            "quality_issues": [...],
            "key_patterns": [...]
        }
    },
    "recommendations": [
        {"action": "description", "priority": "high|medium|low", "target_agent": "feature-engineering-analyst"}
    ],
    "next_steps": ["Run feature engineering", "Run ML theory review"],
    "artifacts": ["reports/eda_report.md"],
    "enables": ["feature-engineering-analyst", "ml-theory-advisor", "frontend-ux-analyst"]
})
```

If `ml_utils.py` is not available in the project, write the report JSON directly to `.claude/reports/eda-analyst_report.json` (and `reports/eda-analyst_report.json`) using the schema above.
```

**Step 2: Add report bus section to feature-engineering-analyst.md**

Replace the existing "Prior Context: Check for EDA Reports" section (lines 10-17) with an expanded version that includes bus reading:

```markdown
## Prior Context: Check for Agent Reports (v1.2.0)

**ALWAYS** start by scanning for prior agent reports:
1. Look for `*_report.json` files in `.claude/reports/`, `reports/`, or equivalent platform directories
2. Specifically look for `eda-analyst_report.json` — contains structured EDA summary with column stats, correlations, quality issues
3. Also check legacy paths: `.claude/eda_report.json`, `reports/eda_report.md`
4. If found, use these insights to inform your feature engineering strategy
5. If not found, proceed with your own data exploration

### On Completion — Write Report

When finished, write your report:

```python
from ml_utils import save_agent_report

save_agent_report("feature-engineering-analyst", {
    "status": "completed",
    "findings": {
        "summary": "Brief narrative of feature engineering recommendations",
        "details": {"features_recommended": [...], "features_to_drop": [...]}
    },
    "recommendations": [
        {"action": "description", "priority": "high", "target_agent": "developer"}
    ],
    "next_steps": ["Implement feature pipeline", "Run preprocessing"],
    "artifacts": ["reports/feature_engineering_plan.md"],
    "depends_on": ["eda-analyst"],
    "enables": ["developer", "mlops-engineer"]
})
```

If `ml_utils.py` is not available, write JSON directly to `.claude/reports/feature-engineering-analyst_report.json` and `reports/feature-engineering-analyst_report.json`.
```

**Step 3: Add report bus section to ml-theory-advisor.md**

Add before the final line (line 141) of `agents/ml-theory-advisor.md`:

```markdown
## Agent Report Bus (v1.2.0)

### On Startup — Read All Prior Reports

Before starting your review, scan for ALL prior agent reports:
1. Look for `*_report.json` files in `.claude/reports/`, `reports/`, or equivalent platform directories
2. Read every report found — your role is to cross-validate findings across agents
3. Flag any cross-agent inconsistencies (e.g., feature eng recommends a feature that has leakage risk)

### On Completion — Write Report

```python
from ml_utils import save_agent_report

save_agent_report("ml-theory-advisor", {
    "status": "completed",
    "findings": {
        "summary": "Brief narrative of theory review findings",
        "details": {"leakage_risks": [...], "methodology_issues": [...], "recommendations": [...]}
    },
    "recommendations": [
        {"action": "description", "priority": "high", "target_agent": "developer"}
    ],
    "next_steps": ["Proceed with preprocessing", "Address identified risks"],
    "artifacts": ["reports/ml_theory_review.md"],
    "depends_on": ["eda-analyst"],
    "enables": ["developer", "mlops-engineer"]
})
```

If `ml_utils.py` is not available, write JSON directly to `.claude/reports/ml-theory-advisor_report.json` and `reports/ml-theory-advisor_report.json`.
```

**Step 4: Add report bus section to mlops-engineer.md**

Add before the final line (line 292) of `agents/mlops-engineer.md`:

```markdown
## Agent Report Bus (v1.2.0)

### On Startup — Read Prior Reports

Before deployment decisions, scan for prior agent reports:
1. Look for `*_report.json` in `.claude/reports/`, `reports/`
2. Key reports to read: evaluation results, preprocessing pipeline details, ML theory advice
3. Use these to inform deployment configuration (model type, preprocessing steps, monitoring needs)

### On Completion — Write Report

```python
from ml_utils import save_agent_report

save_agent_report("mlops-engineer", {
    "status": "completed",
    "findings": {
        "summary": "Deployment summary",
        "details": {"deployment_target": "...", "endpoints": [...], "monitoring": {...}}
    },
    "recommendations": [],
    "next_steps": ["Monitor model performance", "Set up alerting"],
    "artifacts": ["Dockerfile", "docker-compose.yml", "api/app.py"],
    "depends_on": ["developer", "brutal-code-reviewer"]
})
```
```

**Step 5: Add report bus section to developer.md**

Add before "## When Stuck" (line 89) of `agents/developer.md`:

```markdown
## Agent Report Bus (v1.2.0)

### On Startup — Read Relevant Reports

Before implementing, check for agent reports that provide context:
1. Scan `.claude/reports/` and `reports/` for `*_report.json` files
2. Read reports from analysis agents (EDA, feature engineering, ML theory) for implementation guidance
3. Follow their recommendations when implementing features

### On Completion — Write Report

After creating your PR, write a brief report:

```python
from ml_utils import save_agent_report

save_agent_report("developer", {
    "status": "completed",
    "findings": {
        "summary": "Brief description of what was implemented",
        "details": {"files_changed": [...], "pr_number": N}
    },
    "recommendations": [
        {"action": "Review PR #N", "priority": "high", "target_agent": "pr-approver"}
    ],
    "next_steps": ["Code review", "Merge PR"],
    "artifacts": ["src/..."],
    "depends_on": ["feature-engineering-analyst", "ml-theory-advisor"]
})
```

If `ml_utils.py` is not available, write JSON directly to `.claude/reports/developer_report.json` and `reports/developer_report.json`.

```

**Step 6: Add report bus section to brutal-code-reviewer.md**

Add before the final line (line 115) of `agents/brutal-code-reviewer.md`:

```markdown
## Agent Report Bus (v1.2.0)

### On Startup — Read Prior Reports

Before reviewing, scan for prior agent reports:
1. Look for `*_report.json` in `.claude/reports/`, `reports/`
2. Check if code changes align with recommendations from analysis agents
3. Flag deviations from recommended approaches

### On Completion — Write Report

```python
from ml_utils import save_agent_report

save_agent_report("brutal-code-reviewer", {
    "status": "completed",
    "findings": {
        "summary": "Code review summary",
        "details": {"critical_issues": [...], "improvements": [...], "suggestions": [...]}
    },
    "recommendations": [
        {"action": "Fix issue X", "priority": "high", "target_agent": "developer"}
    ],
    "next_steps": ["Address critical issues", "Re-review after fixes"],
    "artifacts": [],
    "depends_on": ["developer"]
})
```
```

**Step 7: Add report bus section to pr-approver.md**

Add before "## When to Reject" (line 67) of `agents/pr-approver.md`:

```markdown
## Agent Report Bus (v1.2.0)

### On Startup — Read Workflow State

Before reviewing a PR, check for prior agent reports:
1. Scan `.claude/reports/` for `*_report.json` files
2. Understand what agents have run and what they recommended
3. Verify the PR addresses recommendations from analysis agents

### On Completion — Write Report

```python
from ml_utils import save_agent_report

save_agent_report("pr-approver", {
    "status": "completed",
    "findings": {
        "summary": "PR review decision",
        "details": {"pr_number": N, "decision": "approved|changes_requested", "comments": [...]}
    },
    "recommendations": [],
    "next_steps": ["Merge completed" or "Address requested changes"],
    "artifacts": [],
    "depends_on": ["developer", "brutal-code-reviewer"]
})
```
```

**Step 8: Add report bus section to frontend-ux-analyst.md**

Add before the final line (line 132) of `agents/frontend-ux-analyst.md`:

```markdown
## Agent Report Bus (v1.2.0)

### On Startup — Read Prior Reports

Before analysis, scan for prior agent reports:
1. Look for `*_report.json` in `.claude/reports/`, `reports/`
2. Read EDA and model reports to understand data context for dashboard design
3. Use evaluation metrics to inform visualization priorities

### On Completion — Write Report

```python
from ml_utils import save_agent_report

save_agent_report("frontend-ux-analyst", {
    "status": "completed",
    "findings": {
        "summary": "UX analysis summary",
        "details": {"strengths": [...], "critical_issues": [...], "enhancements": [...]}
    },
    "recommendations": [
        {"action": "Fix UX issue X", "priority": "high", "target_agent": "developer"}
    ],
    "next_steps": ["Implement UX fixes", "Re-review after changes"],
    "artifacts": [],
    "depends_on": ["eda-analyst"],
    "enables": ["developer"]
})
```
```

**Step 9: Update orchestrator.md to use report bus for workflow state**

Add after "## Best Practices" section (after line 103) of `agents/orchestrator.md`:

```markdown
## Agent Report Bus (v1.2.0)

### Read Workflow State from Reports

Before delegating, scan for all agent reports to understand workflow state:
1. Look for `*_report.json` in `.claude/reports/`, `reports/`
2. Determine which agents have completed their work
3. Identify what's pending and what can run in parallel

### Parallel Execution Groups

Based on agent reports, spawn agents in parallel groups:

**Post-EDA Group** (after eda-analyst completes):
- `feature-engineering-analyst`
- `ml-theory-advisor`
- `frontend-ux-analyst`

**Post-Training Review Group** (after training/evaluation):
- `brutal-code-reviewer`
- `ml-theory-advisor`
- `frontend-ux-analyst`

### On Completion — Write Report

After coordinating a workflow, write a summary report:

```python
from ml_utils import save_agent_report

save_agent_report("orchestrator", {
    "status": "completed",
    "findings": {
        "summary": "Workflow orchestration summary",
        "details": {"agents_spawned": [...], "parallel_groups": [...], "total_duration": "..."}
    },
    "recommendations": [],
    "next_steps": ["Review final outputs"],
    "artifacts": []
})
```
```

**Step 10: Update assigner.md to read reports for smarter assignments**

Add after "## When Uncertain" (after line 83) of `agents/assigner.md`:

```markdown
## Agent Report Bus (v1.2.0)

### Read Reports for Smarter Assignment

Before assigning tickets, scan for agent reports:
1. Look for `*_report.json` in `.claude/reports/`, `reports/`
2. Check which agents have already run — avoid assigning duplicate work
3. Use recommendations with `target_agent` fields to inform assignments

### On Completion — Write Report

```python
from ml_utils import save_agent_report

save_agent_report("assigner", {
    "status": "completed",
    "findings": {
        "summary": "Assignment summary",
        "details": {"tickets_assigned": [...]}
    },
    "recommendations": [],
    "next_steps": [],
    "artifacts": []
})
```
```

**Step 11: Commit all agent changes**

```bash
git add agents/
git commit -m "feat: add report bus read/write behavior to all 10 agents

Each agent now reads prior *_report.json files on startup and writes
its own standardized report on completion. Enables inter-agent
communication and workflow state tracking."
```

---

### Task 3: Create /status command and skill

**Files:**
- Create: `commands/status.md`
- Create: `skills/status/SKILL.md`

**Step 1: Create the /status command definition**

Create `commands/status.md`:

```markdown
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
```

**Step 2: Create the /status skill**

Create `skills/status/SKILL.md`:

```markdown
---
name: status
description: "Show unified workflow status by reading all agent reports. Displays completed agents, pending work, and cross-agent insights."
---

# Workflow Status Skill

Read and display the current state of all agent reports in the shared report bus.

## Implementation

Scan all report directories for `*_report.json` files and present a unified view of:
1. Which agents have completed work (with summaries)
2. Which agents are pending (with dependency information)
3. Cross-agent insights (recommendations targeting other agents)

Use `ml_utils.get_workflow_status()` if available, otherwise scan directories manually.
```

**Step 3: Commit**

```bash
git add commands/status.md skills/status/SKILL.md
git commit -m "feat: add /status command for unified workflow visibility

New slash command that reads all agent reports and displays workflow
state: completed agents, pending work, cross-agent insights."
```

---

### Task 4: Update team-coldstart.md with parallel agent spawning

**Files:**
- Modify: `commands/team-coldstart.md`
- Modify: `skills/team-coldstart/SKILL.md`

**Step 1: Update team-coldstart.md Stage 2 for explicit parallelization**

Replace the "Stage 2: Analysis" section (lines 82-144) with:

```markdown
### Stage 2: Analysis (Sequential → Parallel with Report Bus)

**Step 2a: Run EDA first** (generates report for downstream agents):

1. **eda-analyst** - Comprehensive EDA + Report Bus Output
   ```
   Perform thorough exploratory data analysis on {data_path}.
   Generate statistics, visualizations, and identify data quality issues.

   CRITICAL: Use ml_utils to save structured EDA report:
   - Copy ml_utils.py from the plugin templates if not present in src/
   - Call generate_eda_summary(df, target_col) and save_eda_report(summary)
   - Also save reports/eda_report.md with narrative findings
   - The save_eda_report function will automatically write to the report bus

   Output: EDA report + agent bus report (eda-analyst_report.json)
   ```

**Step 2b: Run these in PARALLEL** (after EDA completes — they all read the EDA report independently):

Spawn ALL THREE agents concurrently using multiple Task tool calls in a single message:

2. **ml-theory-advisor** - Data quality and leakage review (if ML mode)
   ```
   Review the dataset {data_path} for potential data leakage risks.
   Check for features that may contain target information.
   FIRST: Read .claude/reports/eda-analyst_report.json (or .claude/eda_report.json) for prior EDA context.
   Validate data splitting strategy.
   WHEN DONE: Write your report using save_agent_report("ml-theory-advisor", {...})
   Output: Quality and leakage assessment report + agent bus report
   ```

3. **feature-engineering-analyst** - ML Feature Engineering
   ```
   Analyze {data_path} and engineer predictive features.
   FIRST: Read .claude/reports/eda-analyst_report.json (or .claude/eda_report.json) — use the EDA findings
   to inform your feature engineering strategy.
   WHEN DONE: Write your report using save_agent_report("feature-engineering-analyst", {...})
   Output: Feature engineering plan + agent bus report
   ```

4. **frontend-ux-analyst** - Dashboard Planning
   ```
   Review the EDA findings for {data_path} to plan dashboard visualizations.
   FIRST: Read .claude/reports/eda-analyst_report.json for data context.
   Recommend dashboard layout, key visualizations, and interactive features.
   WHEN DONE: Write your report using save_agent_report("frontend-ux-analyst", {...})
   Output: Dashboard design recommendations + agent bus report
   ```

**Wait for all three to complete before proceeding to Stage 3.**
```

**Step 2: Update Stage 5 evaluation section to add parallel review group**

After the existing Stage 5, add a parallel review group:

```markdown
### Stage 5b: Post-Training Review (PARALLEL)

Spawn ALL THREE review agents concurrently:

1. **brutal-code-reviewer** - Code quality review
   ```
   Review all code written in this workflow for quality and maintainability.
   FIRST: Read all reports in .claude/reports/ for context on what was built.
   Output: Code review report + agent bus report
   ```

2. **ml-theory-advisor** - Methodology validation
   ```
   Review the trained model and evaluation results for methodology issues.
   FIRST: Read all reports in .claude/reports/ for full workflow context.
   Output: Theory review report + agent bus report
   ```

3. **frontend-ux-analyst** - Dashboard UX review (if dashboard exists)
   ```
   Review the generated Streamlit dashboard for UX quality.
   FIRST: Read all reports in .claude/reports/ for context.
   Output: UX review report + agent bus report
   ```
```

**Step 3: Commit**

```bash
git add commands/team-coldstart.md skills/team-coldstart/SKILL.md
git commit -m "feat: update team-coldstart with parallel agent groups and report bus

Stage 2 now spawns feature-eng, ml-theory, and frontend-ux in parallel
after EDA. Stage 5b adds parallel post-training review group."
```

---

### Task 5: Update team-analyze.md with parallel spawning and report bus

**Files:**
- Modify: `commands/team-analyze.md`
- Modify: `skills/team-analyze/SKILL.md`

**Step 1: Update team-analyze.md to use report bus and parallel spawning**

Replace the "Agent Coordination" section (lines 303-309) with:

```markdown
## Agent Coordination (v1.2.0 — Report Bus)

This skill coordinates agents using the shared report bus:

1. **eda-analyst** - Data exploration (runs first, writes report)
2. **ml-theory-advisor** - Leakage assessment (runs in PARALLEL after EDA)
3. **feature-engineering-analyst** - Feature recommendations (runs in PARALLEL after EDA)

### Parallel Execution

After EDA completes, spawn ml-theory-advisor AND feature-engineering-analyst concurrently.
Both agents read the EDA report independently and write their own reports to the bus.

Each agent should:
- Read prior reports from `.claude/reports/` on startup
- Write their report using `save_agent_report()` on completion

### Status Check

After all agents complete, run a status check:
```python
from ml_utils import get_workflow_status
status = get_workflow_status()
print(f"Completed: {len(status['completed'])}, Pending: {len(status['pending'])}")
for insight in status['insights']:
    print(f"  {insight['from']} → {insight['to']}: {insight['action']}")
```
```

**Step 2: Commit**

```bash
git add commands/team-analyze.md skills/team-analyze/SKILL.md
git commit -m "feat: update team-analyze with parallel spawning and report bus

Agents now run in parallel where possible and communicate via the
shared report bus."
```

---

### Task 6: Add post-agent-report hook

**Files:**
- Create: `hooks/post-agent-report.sh`
- Modify: `hooks/hooks.json`
- Modify: `hooks/cursor-hooks.json`

**Step 1: Create the hook script**

Create `hooks/post-agent-report.sh`:

```bash
#!/bin/bash
# Post-agent report validation hook
# Runs after any agent completes to verify report was written correctly

set -e

AGENT_TYPE="${CLAUDE_TOOL_ARG_SUBAGENT_TYPE:-unknown}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "[Post-Agent Hook] Checking report for agent: $AGENT_TYPE"

# Check all possible report directories
REPORT_FOUND=false
for DIR in .claude/reports reports .cursor/reports .codex/reports .opencode/reports; do
    REPORT_FILE="$DIR/${AGENT_TYPE}_report.json"
    if [ -f "$REPORT_FILE" ]; then
        REPORT_FOUND=true
        echo "  ✓ Report found: $REPORT_FILE"

        # Basic schema validation
        if command -v python3 &>/dev/null; then
            python3 -c "
import json, sys
with open('$REPORT_FILE') as f:
    report = json.load(f)
required = ['agent', 'version', 'timestamp', 'status', 'findings']
missing = [k for k in required if k not in report]
if missing:
    print(f'  ⚠ Missing fields: {missing}')
    sys.exit(1)
print(f'  ✓ Schema valid (agent={report[\"agent\"]}, status={report[\"status\"]})')
" 2>/dev/null || echo "  ⚠ Schema validation skipped (python3 not available or invalid JSON)"
        fi
        break
    fi
done

if [ "$REPORT_FOUND" = false ]; then
    echo "  ⚠ No report found for agent: $AGENT_TYPE"
    echo "    Expected: .claude/reports/${AGENT_TYPE}_report.json"
    echo "    Tip: Agent should call save_agent_report() from ml_utils.py"
fi

# Log to workflow log
mkdir -p .claude
echo "$TIMESTAMP - Agent report check: $AGENT_TYPE (found=$REPORT_FOUND)" >> .claude/workflow.log 2>/dev/null || true
```

**Step 2: Make it executable**

```bash
chmod +x hooks/post-agent-report.sh
```

**Step 3: Update hooks.json — add SubagentStop matchers for all agents**

Replace the `SubagentStop` section (lines 62-71) in `hooks/hooks.json` with:

```json
    "SubagentStop": [
      {
        "matcher": "eda-analyst",
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/post-eda.sh"
          },
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/post-agent-report.sh"
          }
        ]
      },
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/post-agent-report.sh"
          }
        ]
      }
    ]
```

**Step 4: Update cursor-hooks.json — add equivalent hook**

Add to `hooks/cursor-hooks.json` after the `afterShellExecution` block:

```json
    "postToolUse": [
      {
        "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/post-dashboard.sh \"$CLAUDE_TOOL_ARG_FILE_PATH\"",
        "matcher": "dashboard"
      },
      {
        "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/post-agent-report.sh",
        "matcher": "agent"
      }
    ],
```

**Step 5: Commit**

```bash
git add hooks/post-agent-report.sh hooks/hooks.json hooks/cursor-hooks.json
git commit -m "feat: add post-agent report validation hook

Validates that agents write reports with correct schema after
completion. Logs results to workflow.log."
```

---

### Task 7: Version bump and platform config updates

**Files:**
- Modify: `.claude-plugin/plugin.json`
- Modify: `.cursor-plugin/plugin.json`
- Modify: `.opencode/plugins/ml-automation.js`
- Modify: `.codex/INSTALL.md`

**Step 1: Update .claude-plugin/plugin.json**

Change version to 1.2.0 and update description:

```json
{
  "name": "ml-automation",
  "version": "1.2.0",
  "description": "End-to-end ML automation workflow for Claude Code. Includes 10 specialized agents with shared report bus for inter-agent communication, parallel execution groups, /status command for workflow visibility, reusable ML utilities, and hooks for quality gates. v1.2: Shared Report Bus, parallel agent groups, /status command.",
  "author": {
    "name": "Maximo Lorenzo y Losada"
  },
  "repository": "https://github.com/maxilylm/ml-automation-plugin",
  "license": "MIT",
  "keywords": [
    "ml",
    "machine-learning",
    "data-science",
    "automation",
    "agents",
    "eda",
    "mlops",
    "deployment",
    "streamlit",
    "workflows",
    "report-bus",
    "parallel-agents"
  ]
}
```

**Step 2: Update .cursor-plugin/plugin.json**

Same version and description changes:

```json
{
  "name": "ml-automation",
  "description": "End-to-end ML automation workflow: 10 specialized agents with shared report bus, parallel execution groups, /status command, and quality gate hooks",
  "version": "1.2.0",
  "author": {
    "name": "Maximo Lorenzo y Losada"
  },
  "homepage": "https://github.com/maxilylm/ml-automation-plugin",
  "repository": "https://github.com/maxilylm/ml-automation-plugin",
  "license": "MIT",
  "keywords": ["ml", "machine-learning", "data-science", "automation", "agents", "eda", "mlops", "deployment", "workflows", "report-bus", "parallel-agents"],
  "skills": "./skills/",
  "agents": "./agents/",
  "commands": "./commands/",
  "hooks": "./hooks/cursor-hooks.json"
}
```

**Step 3: Update .opencode/plugins/ml-automation.js to register the status tool**

The OpenCode plugin auto-discovers commands from the `commands/` directory (line 42-49 of `ml-automation.js`), so the new `status.md` command file will be picked up automatically. No code change needed — just verify by checking the tool registration loop.

**Step 4: Update .codex/INSTALL.md**

Read the current file and update with v1.2.0 info.

**Step 5: Commit**

```bash
git add .claude-plugin/plugin.json .cursor-plugin/plugin.json .codex/INSTALL.md
git commit -m "feat: bump version to 1.2.0 across all platforms

Updates plugin manifests for Claude Code, Cursor, and Codex with
v1.2.0 version and updated descriptions mentioning shared report
bus, parallel agents, and /status command."
```

---

### Task 8: Update README.md with v1.2.0 features

**Files:**
- Modify: `README.md`

**Step 1: Read current README**

Read `README.md` to understand current structure.

**Step 2: Add v1.2.0 section**

Add a "What's New in v1.2.0" section and update the feature list to include:
- Shared Report Bus — inter-agent communication via JSON reports
- Parallel Agent Execution — post-EDA and post-training parallel groups
- `/status` command — unified workflow visibility
- Report validation hooks — automatic schema checking

Also update the version badge/reference from 1.1.0 to 1.2.0.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README with v1.2.0 features

Documents shared report bus, parallel agent execution, /status
command, and report validation hooks."
```

---

### Task 9: Update post-eda.sh for report bus compatibility

**Files:**
- Modify: `hooks/post-eda.sh`

**Step 1: Update post-eda.sh to check for bus-format reports**

Add after line 35 (the legacy report check):

```bash
# 3b. Check for new bus-format report
for DIR in .claude/reports reports .cursor/reports; do
    if [ -f "$DIR/eda-analyst_report.json" ]; then
        echo "  - Bus-format report found at $DIR/eda-analyst_report.json"
        break
    fi
done
```

**Step 2: Commit**

```bash
git add hooks/post-eda.sh
git commit -m "feat: update post-eda hook for report bus compatibility

Now checks for both legacy and bus-format EDA reports."
```

---

### Task 10: Final verification and tag

**Step 1: Verify all files are consistent**

```bash
# Check version is 1.2.0 everywhere
grep -r "1.2.0" .claude-plugin/plugin.json .cursor-plugin/plugin.json
grep "REPORT_SCHEMA_VERSION" templates/ml_utils.py

# Verify all agents mention report bus
grep -l "Report Bus" agents/*.md | wc -l  # Should be 10

# Verify new files exist
ls commands/status.md skills/status/SKILL.md hooks/post-agent-report.sh

# Verify hooks.json has SubagentStop wildcard
grep -A 5 '"matcher": "\*"' hooks/hooks.json
```

**Step 2: Run any existing tests**

```bash
# If tests exist
python3 -c "from templates.ml_utils import save_agent_report, load_agent_reports, get_workflow_status; print('Import OK')" 2>/dev/null || python3 -c "import sys; sys.path.insert(0, '.'); exec(open('templates/ml_utils.py').read()); print('Functions defined OK')"
```

**Step 3: Final commit if any fixes needed, then tag**

```bash
git tag -a v1.2.0 -m "v1.2.0: Shared Report Bus, parallel agents, /status command"
```
