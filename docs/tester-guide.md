# ML Automation Plugin — Tester Guide

## Quick Setup

1. **Install Claude Code** (if not already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Clone the plugin**:
   ```bash
   cd ~/.claude/plugins/
   git clone https://github.com/maxilylm/ml-automation-plugin.git
   ```

3. **Verify installation**:
   ```bash
   claude
   # Then type: /status
   # Should show "No agent reports found" message
   ```

## Test Scenarios

### Scenario 1: Quick EDA (5 min)
**Goal:** Verify basic data exploration works end-to-end.

```
/eda path/to/your/data.csv
```

**Check:**
- [ ] Data overview displayed (shape, columns, types)
- [ ] Missing values identified
- [ ] Distributions analyzed
- [ ] Correlations computed
- [ ] Quality issues flagged
- [ ] EDA report saved to `.claude/eda_report.json`

### Scenario 2: Full Pipeline (15-30 min)
**Goal:** Verify end-to-end workflow from data to dashboard.

```
/team coldstart path/to/data.csv --target target_column
```

**Check:**
- [ ] All stages complete (init → EDA → processing → training → eval → dashboard)
- [ ] Reflection gates fire between stages
- [ ] Model artifacts saved to `models/`
- [ ] Dashboard generated in `dashboard/app.py`
- [ ] Reports generated in report directories
- [ ] Traceability log populated

### Scenario 3: Methodology Document (5 min)
**Goal:** Verify governance report generation after a pipeline run.

```
/methodology
```

**Check:**
- [ ] Document includes all 12 sections
- [ ] Decision points extracted from trace log
- [ ] Model results from registry included
- [ ] Reproducibility guide has data fingerprint

### Scenario 4: Traceability Audit (2 min)
**Goal:** Verify audit trail is complete.

```
/trace --verbose
```

**Check:**
- [ ] All agent spawn/complete events logged
- [ ] Stage transitions visible
- [ ] Decisions and reflections captured

### Scenario 5: Feedback Collection (2 min)
**Goal:** Verify feedback system works.

```
/feedback
```

**Check:**
- [ ] Structured form presented
- [ ] Feedback saved to report bus
- [ ] Confirmation message displayed

## Test Datasets

Use datasets from `eda-workspace/test-datasets/`:
- `sales_data.csv` — Clean sales data (500 rows)
- `messy_customer_data.csv` — Deliberately dirty data
- `churn_data.csv` — Classification with imbalanced target

## Reporting Issues

Use `/feedback` to report issues directly, or create a GitHub issue at:
https://github.com/maxilylm/ml-automation-plugin/issues

### Feedback Template

When reporting issues, include:
1. **What command** you ran
2. **What happened** (copy error output if applicable)
3. **What you expected** to happen
4. **Your environment** (OS, Claude Code version)
5. **Severity** (critical / high / medium / low)

## Common Issues

| Issue | Solution |
|-------|----------|
| `/status` shows no reports | Run a workflow command first (`/eda`, `/team coldstart`) |
| Dashboard validation fails | Check that `dashboard/app.py` has valid Python syntax |
| Model training hangs | Large datasets may need patience; check with `/status` |
| Missing `ml_utils.py` | Should be auto-copied to `src/` during init stage |
