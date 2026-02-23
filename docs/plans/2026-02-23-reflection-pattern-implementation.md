# Reflection Pattern Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reflection mode to ml-theory-advisor with three pre-execution gate checkpoints that validate plans before feature engineering, preprocessing, and training proceed.

**Architecture:** Enhance ml-theory-advisor's system prompt with a reflection mode that reads prior reports and writes verdicts (approved/revise). Insert gate sub-stages into team-coldstart and team-analyze. Add helper functions to ml_utils.py for reflection report I/O.

**Tech Stack:** Markdown agent definitions, JSON reports, Python utilities

---

### Task 1: Add reflection report helpers to ml_utils.py

**Files:**
- Modify: `templates/ml_utils.py:459` (append after `get_workflow_status`)

**Step 1: Add the two new functions at end of file**

```python
# =============================================================================
# 8. REFLECTION REPORTS
# =============================================================================

REFLECTION_GATES = ["post-feature-engineering", "post-preprocessing", "post-training"]


def save_reflection_report(gate, report_data, output_dirs=None):
    """
    Save a reflection gate report from ml-theory-advisor.

    Args:
        gate: One of 'post-feature-engineering', 'post-preprocessing', 'post-training'
        report_data: Dict with keys: verdict, reasoning, corrections
        output_dirs: List of directories (defaults to PLATFORM_REPORT_DIRS)
    """
    from datetime import datetime, timezone

    if output_dirs is None:
        output_dirs = PLATFORM_REPORT_DIRS

    report = {
        "agent": "ml-theory-advisor",
        "version": REPORT_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "findings": {
            "summary": f"Reflection gate: {gate}",
            "details": {
                "gate": gate,
                "verdict": report_data.get("verdict", "approved"),
                "reasoning": report_data.get("reasoning", ""),
                "corrections": report_data.get("corrections", []),
            },
        },
        "recommendations": report_data.get("corrections", []),
        "next_steps": [],
        "artifacts": [],
        "depends_on": [],
        "enables": [],
    }

    filename = f"ml-theory-advisor_reflection_{gate}_report.json"
    paths_written = []

    for d in output_dirs:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, filename)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        paths_written.append(path)

    return paths_written


def load_reflection_report(gate, search_dirs=None):
    """
    Load the most recent reflection report for a specific gate.

    Args:
        gate: One of 'post-feature-engineering', 'post-preprocessing', 'post-training'
        search_dirs: Directories to search (defaults to PLATFORM_REPORT_DIRS)

    Returns:
        dict with verdict, reasoning, corrections — or None if not found
    """
    if search_dirs is None:
        search_dirs = PLATFORM_REPORT_DIRS

    filename = f"ml-theory-advisor_reflection_{gate}_report.json"
    latest = None

    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    report = json.load(f)
                if latest is None or report.get("timestamp", "") > latest.get("timestamp", ""):
                    latest = report
            except (json.JSONDecodeError, KeyError):
                continue

    if latest:
        return latest.get("findings", {}).get("details", {})
    return None
```

**Step 2: Verify syntax**

Run: `python3 -c "exec(open('templates/ml_utils.py').read()); print('OK:', 'save_reflection_report' in dir(), 'load_reflection_report' in dir())"`
Expected: `OK: True True`

**Step 3: Commit**

```bash
git add templates/ml_utils.py
git commit -m "feat: add reflection report helpers to ml_utils.py"
```

---

### Task 2: Add Reflection Mode to ml-theory-advisor agent

**Files:**
- Modify: `agents/ml-theory-advisor.md:141` (before the existing "Agent Report Bus" section)

**Step 1: Insert Reflection Mode section before line 143 ("## Agent Report Bus")**

Add this content between the existing "Red Flags" section and the "Agent Report Bus" section:

```markdown

## Reflection Mode (v1.2.1)

When invoked for **reflection**, you act as a pre-execution gate — evaluating a proposed plan or completed output BEFORE the next workflow stage proceeds. This is different from your regular review mode which evaluates post-hoc.

### When to Reflect

You enter reflection mode when your prompt includes phrases like:
- "Reflect on the feature engineering plan"
- "Gate check: validate preprocessing approach"
- "Reflection mode: evaluate training strategy"

### Your Reflection Process

1. **Read ALL prior agent reports** from `.claude/reports/`, `reports/`
2. **Evaluate the strategy** — not code quality, but whether the *approach* is sound for the problem domain
3. **Consider domain context** — e.g., MMM needs adstock transformations, time series needs lag features, imbalanced data needs appropriate sampling
4. **Write a reflection report** with a clear verdict

### Reflection Verdicts

**Approved** — The approach is sound. Proceed to next stage.

**Revise** — Issues found that should be corrected before proceeding. Provide specific, actionable corrections with target agents.

### Reflection Report Format

Write your reflection report using ml_utils:

```python
from ml_utils import save_reflection_report

# When approving:
save_reflection_report("post-feature-engineering", {
    "verdict": "approved",
    "reasoning": "Feature strategy correctly addresses the problem domain. Adstock transformations included for media channels, interaction terms cover key relationships.",
    "corrections": []
})

# When requesting revision:
save_reflection_report("post-training", {
    "verdict": "revise",
    "reasoning": "Gradient boosting is suboptimal for this MMM problem. Bayesian methods better capture uncertainty in media effect estimates and provide interpretable posteriors.",
    "corrections": [
        {
            "target_agent": "developer",
            "issue": "Wrong model family for MMM",
            "recommendation": "Use Bayesian regression (PyMC or lightweight_mmm) instead of gradient boosting. MMM requires: (1) interpretable coefficients per channel, (2) uncertainty quantification, (3) prior incorporation for domain knowledge.",
            "priority": "critical"
        }
    ]
})
```

If `ml_utils.py` is not available, write JSON directly to `.claude/reports/ml-theory-advisor_reflection_{gate}_report.json`.

### Gate-Specific Evaluation Criteria

**Gate 1: Post-Feature-Engineering**
- Are features appropriate for the problem domain?
- Are domain-specific transformations included (adstock, lag, seasonal)?
- Any leakage risk in the proposed features?
- Is variable selection justified?

**Gate 2: Post-Preprocessing**
- Is the preprocessing pipeline appropriate for the chosen model family?
- Are scaling/encoding choices correct?
- Is the data flow from features to model consistent?
- Any information loss in transformations?

**Gate 3: Post-Training**
- Is the model family appropriate for the problem domain?
- Is the hyperparameter strategy reasonable?
- Is the validation approach sound (CV scheme, holdout strategy)?
- Are there better-suited alternatives for this specific problem?

### Iteration Protocol

You may be invoked multiple times for the same gate (max iterations configured by workflow). On subsequent iterations:
1. Read the PREVIOUS reflection report to see what you asked to fix
2. Read the UPDATED agent report to see what changed
3. Evaluate whether corrections were adequately addressed
4. Write a new verdict — either `approved` or `revise` with remaining issues
```

**Step 2: Commit**

```bash
git add agents/ml-theory-advisor.md
git commit -m "feat: add reflection mode to ml-theory-advisor agent"
```

---

### Task 3: Insert gate checkpoints into team-coldstart

**Files:**
- Modify: `commands/team-coldstart.md`

**Step 1: Add Gate 1 after Stage 2 (after line ~150, before Stage 3)**

Insert between Stage 2 output and Stage 3:

```markdown
### Stage 2c: Gate 1 — Reflect on Feature Engineering (Reflection Loop)

**Reflection gate** — validates feature engineering output before preprocessing.

```
iteration = 0
max_iterations = {--max-reflect, default: 2}

while iteration < max_iterations:
```

1. **Spawn ml-theory-advisor in reflection mode:**
   ```
   REFLECTION MODE — Gate 1: Post-Feature-Engineering

   Read ALL reports in .claude/reports/, especially:
   - eda-analyst_report.json
   - feature-engineering-analyst_report.json

   Evaluate whether the proposed feature engineering strategy is sound:
   - Are features appropriate for the problem domain?
   - Are domain-specific transformations included?
   - Any leakage risk?
   - Is variable selection justified?

   Write reflection report using save_reflection_report("post-feature-engineering", {...})
   ```

2. **Read reflection report verdict:**
   - If `"verdict": "approved"` → proceed to Stage 3
   - If `"verdict": "revise"` → spawn feature-engineering-analyst with corrections:
     ```
     REVISION REQUEST from ml-theory-advisor reflection gate.

     Read the reflection report at .claude/reports/ml-theory-advisor_reflection_post-feature-engineering_report.json

     Address the corrections listed. Update your feature engineering plan accordingly.
     Write your updated report using save_agent_report("feature-engineering-analyst", {...})
     ```
   - Increment iteration, loop back to step 1

3. **If max iterations reached with verdict still "revise":**
   - Log warning: "Gate 1: Max reflection iterations reached, proceeding with best effort"
   - Proceed to Stage 3
```

**Step 2: Add Gate 2 after Stage 3 (before Stage 4)**

Insert between Stage 3 and Stage 4, same pattern as Gate 1 but for preprocessing:

```markdown
### Stage 3b: Gate 2 — Reflect on Preprocessing Pipeline (Reflection Loop)

**Reflection gate** — validates preprocessing pipeline before training.

Same iteration loop as Gate 1, but:
- Gate name: `"post-preprocessing"`
- Reports to read: preprocessing report + feature-engineering report
- Evaluation focus: pipeline design, scaling/encoding choices, data flow correctness
- On revise: re-spawn preprocessing stage with corrections
```

**Step 3: Add Gate 3 after Stage 4 (before Stage 5)**

Insert between Stage 4 and Stage 5, same pattern for training:

```markdown
### Stage 4b: Gate 3 — Reflect on Training Approach (Reflection Loop)

**Reflection gate** — validates training approach before evaluation.

Same iteration loop as Gate 1, but:
- Gate name: `"post-training"`
- Reports to read: ALL prior reports + training report
- Evaluation focus: model family appropriateness, hyperparameter strategy, validation approach
- On revise: re-spawn training stage with corrections
```

**Step 4: Add `--max-reflect` to Configuration Options table**

Find the Configuration Options table and add this row:

```markdown
| `--max-reflect` | 2 | Maximum reflection iterations per gate (0 to skip gates) |
```

**Step 5: Commit**

```bash
git add commands/team-coldstart.md
git commit -m "feat: insert 3 reflection gate checkpoints into team-coldstart"
```

---

### Task 4: Add Gate 1 to team-analyze

**Files:**
- Modify: `commands/team-analyze.md`
- Modify: `skills/team-analyze/SKILL.md`

**Step 1: Add Gate 1 to team-analyze**

In `commands/team-analyze.md`, find the "Agent Coordination (v1.2.0 — Report Bus)" section and add after the parallel execution subsection:

```markdown
### Reflection Gate (v1.2.1)

After feature-engineering-analyst completes, run a reflection gate:

1. Spawn ml-theory-advisor in reflection mode for `post-feature-engineering` gate
2. If verdict is `revise`, re-spawn feature-engineering-analyst with corrections
3. Max iterations: configurable via `--max-reflect` (default: 2)
4. If `--max-reflect 0`, skip the gate entirely

Add `--max-reflect` to the Configuration Options table.
```

**Step 2: Update skills/team-analyze/SKILL.md**

Append to the file:

```markdown

## Reflection Gate (v1.2.1)

After parallel analysis completes, a reflection gate validates feature engineering output before the summary report. The ml-theory-advisor evaluates the strategy and either approves or requests revisions (max configurable iterations, default 2).
```

**Step 3: Commit**

```bash
git add commands/team-analyze.md skills/team-analyze/SKILL.md
git commit -m "feat: add reflection gate to team-analyze"
```

---

### Task 5: Update skills/team-coldstart/SKILL.md

**Files:**
- Modify: `skills/team-coldstart/SKILL.md`

**Step 1: Add reflection documentation to the skill**

Append to the existing file:

```markdown

## Reflection Gates (v1.2.1)

Three reflection checkpoints validate outputs before the next stage proceeds:

| Gate | After | Before | Evaluates |
|------|-------|--------|-----------|
| Gate 1 | Feature Engineering | Preprocessing | Feature strategy, domain fit, leakage risk |
| Gate 2 | Preprocessing | Training | Pipeline design, encoding, data flow |
| Gate 3 | Training | Evaluation | Model family, hyperparameters, validation |

Each gate spawns ml-theory-advisor in reflection mode. If verdict is `revise`, the upstream agent re-runs with corrections (max `--max-reflect` iterations, default 2). Set `--max-reflect 0` to skip gates.
```

**Step 2: Commit**

```bash
git add skills/team-coldstart/SKILL.md
git commit -m "feat: document reflection gates in team-coldstart skill"
```

---

### Task 6: Version bump and README update

**Files:**
- Modify: `.claude-plugin/plugin.json`
- Modify: `.cursor-plugin/plugin.json`
- Modify: `README.md`

**Step 1: Bump .claude-plugin/plugin.json version from 1.2.0 to 1.2.1**

Update the `"version"` field and add "reflection gates" to the description.

**Step 2: Bump .cursor-plugin/plugin.json version from 1.2.0 to 1.2.1**

Same changes.

**Step 3: Add reflection pattern section to README.md**

After the "What's New in v1.2.0" section, add:

```markdown
### Reflection Gates (v1.2.1)

Pre-execution validation gates that evaluate the *strategy* before the next workflow stage proceeds:

- **Gate 1** (post-feature-engineering): Validates feature strategy, domain-specific transformations, leakage risk
- **Gate 2** (post-preprocessing): Validates pipeline design, encoding choices, data flow
- **Gate 3** (post-training): Validates model family, hyperparameter strategy, validation approach

If issues are found, the upstream agent re-runs with corrections (max 2 iterations by default, configurable with `--max-reflect`).

```bash
# Use reflection gates (default)
/team coldstart data.csv --target Revenue

# Skip reflection gates
/team coldstart data.csv --target Revenue --max-reflect 0

# Allow more iterations
/team coldstart data.csv --target Revenue --max-reflect 3
```
```

**Step 4: Commit**

```bash
git add .claude-plugin/plugin.json .cursor-plugin/plugin.json README.md
git commit -m "feat: bump to v1.2.1, document reflection gates in README"
```

---

### Task 7: Final verification and cleanup

**Step 1: Verify all changes**

```bash
# Version check
grep '"1.2.1"' .claude-plugin/plugin.json .cursor-plugin/plugin.json

# Reflection mode in agent
grep -c "Reflection Mode" agents/ml-theory-advisor.md  # Should be >= 1

# Gates in team-coldstart
grep -c "Gate" commands/team-coldstart.md  # Should be >= 3

# New functions in ml_utils
python3 -c "exec(open('templates/ml_utils.py').read()); print('save_reflection_report' in dir(), 'load_reflection_report' in dir())"
```

**Step 2: Delete plan files**

```bash
rm docs/plans/2026-02-23-reflection-pattern-design.md docs/plans/2026-02-23-reflection-pattern-implementation.md
rmdir docs/plans docs 2>/dev/null || true
git add -A
git commit -m "chore: remove reflection pattern plan files"
```
