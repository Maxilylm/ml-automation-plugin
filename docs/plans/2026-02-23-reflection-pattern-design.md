# V1.2.1 Design: Reflection Pattern — Pre-Execution Validation Gates

**Date:** 2026-02-23
**Version:** 1.2.1
**Status:** Approved

## Problem

The current workflow validates outputs *after* execution (post-hoc). The ml-theory-advisor reviews feature engineering, preprocessing, and training results after they've been built. This means conceptual errors (e.g., wrong model family for the problem domain, missing domain-specific transformations) are caught late — after compute and implementation time has been spent.

Example: In a Marketing Mix Modeling (MMM) case, the workflow might train a gradient-boosted model, evaluate it, and only then the ml-theory-advisor says "Bayesian methods with adstock transformations would be appropriate." The feature engineering and training steps were wasted.

## Solution: Reflection Mode with Gate Checkpoints

Enhance the `ml-theory-advisor` agent with a **reflection mode** that acts as a pre-execution gate. Insert three gate checkpoints into the workflow where the reflector evaluates the *plan* before the next stage executes. If issues are found, the upstream agent re-runs with corrections (max configurable iterations, default 2).

## 1. Reflection Mode for ml-theory-advisor

When invoked for reflection, the agent:
1. Reads all prior reports from the bus
2. Evaluates the proposed strategy (not code — the approach)
3. Writes a reflection report with a verdict: `approved` or `revise`

### Reflection report schema

```json
{
  "agent": "ml-theory-advisor",
  "status": "completed",
  "findings": {
    "summary": "Reflection on feature engineering plan",
    "details": {
      "gate": "pre-feature-engineering",
      "verdict": "revise",
      "reasoning": "MMM data requires adstock transformations and carryover effects...",
      "corrections": [
        {
          "target_agent": "feature-engineering-analyst",
          "issue": "Missing media adstock transformations",
          "recommendation": "Add adstock decay features for each media channel",
          "priority": "critical"
        }
      ]
    }
  }
}
```

Filename convention: `ml-theory-advisor_reflection_{gate}_report.json`

## 2. Three Gate Checkpoints

### Updated workflow

```
Stage 1: Initialize
Stage 2a: EDA (sequential)
Stage 2b: Post-EDA parallel group (existing)

Stage 2c: GATE 1 — Reflect on feature engineering output
  → ml-theory-advisor reads feature-eng report
  → Verdict: approved → proceed | revise → feature-eng re-runs

Stage 3: Preprocessing

Stage 3b: GATE 2 — Reflect on preprocessing pipeline
  → ml-theory-advisor reads preprocessing report
  → Verdict: approved → proceed | revise → preprocessing re-runs

Stage 4: Training

Stage 4b: GATE 3 — Reflect on training approach
  → ml-theory-advisor reads training plan/results
  → Verdict: approved → proceed | revise → training re-runs

Stage 5: Evaluation
Stage 5b: Post-training review (existing)
Stage 6-9: Dashboard, Deploy, Finalize
```

### Iteration loop (same for all 3 gates)

```
iteration = 0
max_iterations = configurable (default: 2)

while iteration < max_iterations:
    spawn ml-theory-advisor in reflection mode for this gate
    wait for completion
    read reflection report

    if verdict == "approved":
        break → proceed to next stage

    if verdict == "revise":
        spawn target agent with corrections from reflection report
        wait for completion
        iteration += 1

if iteration == max_iterations and verdict still "revise":
    log warning: "Max reflection iterations reached, proceeding with best effort"
    proceed to next stage
```

### Gate evaluation matrix

| Gate | Reads | Evaluates | Example Correction |
|------|-------|-----------|-------------------|
| Gate 1: Post-Feature-Eng | EDA + feature-eng reports | Feature strategy, variable selection, leakage risk, domain-specific transformations | "Add adstock transformations for MMM media channels" |
| Gate 2: Post-Preprocessing | Preprocessing + feature reports | Pipeline design, scaling choices, encoding strategy, data flow correctness | "Use leave-one-out encoding instead of one-hot for high-cardinality" |
| Gate 3: Post-Training | All prior reports + training config | Model selection, hyperparameter strategy, validation approach, problem-method fit | "Use Bayesian regression for MMM, not gradient boosting" |

## 3. Configuration

New option for team-coldstart:

| Option | Default | Description |
|--------|---------|-------------|
| `--max-reflect` | 2 | Maximum reflection iterations per gate |

## 4. Files to Modify

| File | Change |
|------|--------|
| `agents/ml-theory-advisor.md` | Add "Reflection Mode" section |
| `commands/team-coldstart.md` | Insert 3 gate sub-stages |
| `commands/team-analyze.md` | Add Gate 1 (post-feature-eng) |
| `skills/team-coldstart/SKILL.md` | Document reflection gates |
| `skills/team-analyze/SKILL.md` | Document Gate 1 |
| `templates/ml_utils.py` | Add `save_reflection_report()` and `load_reflection_report()` |
| `.claude-plugin/plugin.json` | Bump to v1.2.1 |
| `.cursor-plugin/plugin.json` | Bump to v1.2.1 |
| `README.md` | Document reflection pattern |

No new agents, commands, or hooks needed.

## 5. Backward Compatibility

- Reflection gates are additive — they don't change existing stages
- The `--max-reflect 0` flag skips all gates entirely (opt-out)
- Reflection reports use the existing bus schema (extended with `gate`, `verdict`, `corrections`)
- Existing post-hoc reviews (Stage 5b) remain unchanged — they catch different issues
