# ML Automation Plugin

End-to-end machine learning automation workflow for AI coding assistants. Takes you from raw data to deployed model with interactive dashboard — orchestrated by specialized AI agents.

Supports **Claude Code**, **GitHub Copilot**, **Cursor**, **Codex**, and **OpenCode**.

## Quick Start

```bash
# Full pipeline from raw data to deployed dashboard
/team-coldstart data/sales.csv --target Revenue

# Just explore a dataset
/eda data/customers.csv

# Quick multi-agent analysis
/team-analyze data/survey.csv

# Train a model
/train

# Deploy to Docker
/deploy local
```

## How It Works

The plugin provides a complete ML workflow where agents collaborate:

```
Raw Data → /eda → /preprocess → /train → /evaluate → /deploy
              ↓         ↓           ↓          ↓
         eda-analyst  ml-theory  ml-theory  mlops-engineer
                      advisor    advisor
```

Or run the full pipeline with a single command:

```
/team-coldstart data.csv --target y
```

This orchestrates all stages automatically:
1. Validates data and detects task type (ML vs analysis)
2. Runs parallel EDA, leakage review, and feature analysis
3. Builds preprocessing pipeline with leakage prevention
4. Trains and compares models
5. Generates comprehensive evaluation
6. Creates interactive Streamlit dashboard
7. Packages for production (FastAPI + Docker)
8. Deploys to target environment

## What's Included

### 10 Specialized Agents

| Agent | Role |
|-------|------|
| `eda-analyst` | Exploratory data analysis on any dataset |
| `ml-theory-advisor` | ML theory guidance, data leakage prevention |
| `feature-engineering-analyst` | Feature design and opportunity discovery |
| `mlops-engineer` | Model deployment, APIs, Docker, CI/CD |
| `developer` | Code implementation on feature branches |
| `brutal-code-reviewer` | Code quality and maintainability review |
| `pr-approver` | Pull request review and merge |
| `frontend-ux-analyst` | UI/UX design feedback |
| `orchestrator` | Multi-agent coordination |
| `assigner` | Automatic ticket routing |

### Skills / Slash Commands

| Skill | Description |
|-------|-------------|
| `eda` | Run exploratory data analysis |
| `preprocess` | Build data processing pipeline (leakage-safe) |
| `train` | Train ML models with proper validation |
| `evaluate` | Comprehensive model evaluation with visualizations |
| `deploy` | Deploy to Docker, Snowflake, AWS, or GCP |
| `report` | Generate EDA, model, drift, or project reports |
| `test` | Generate and run tests (80% coverage threshold) |
| `team-coldstart` | Full pipeline: raw data to deployed dashboard |
| `team-analyze` | Quick multi-agent data analysis |
| `team-review` | Multi-agent code review |
| `status` | Show workflow status and agent reports |
| `registry` | Inspect MLOps registries (models, features, experiments, data) |

### Hooks

- **Pre-commit**: Python syntax check, secrets detection, test coverage validation
- **Pre-deploy**: Deployment readiness checks (files, health endpoints, Docker, env vars)
- **Post-EDA**: Extract metrics, flag data quality issues
- **Post-dashboard**: Validate syntax (`ast.parse`), detect placeholders, import-level check, generate run scripts
- **Post-workflow**: Summarize outputs, generate quick-start commands

## Installation

### Claude Code

**Option 1: Install from marketplace (recommended)**

1. Open Claude Code and run `/plugin`
2. Select **"Add marketplace"**
3. Enter: `maxilylm/ml-automation-plugin`
4. Select **"Install plugins"** and choose `ml-automation`
5. Restart Claude Code

**Option 2: Local development**

```bash
git clone https://github.com/maxilylm/ml-automation-plugin.git
claude --plugin-dir /path/to/ml-automation-plugin
```

### GitHub Copilot

**Option 1: Install as plugin (VS Code / Copilot CLI)**

```bash
# If you have a marketplace configured:
copilot plugin install ml-automation@<marketplace>

# Or install locally:
git clone https://github.com/maxilylm/ml-automation-plugin.git
copilot plugin install ./ml-automation-plugin/.copilot
```

**Option 2: Personal skills (Copilot CLI)**

```bash
git clone https://github.com/maxilylm/ml-automation-plugin.git
ln -s $(pwd)/ml-automation-plugin/skills ~/.copilot/skills/ml-automation
```

Skills and agents are automatically discovered. The `.copilot/` directory contains the plugin manifest with `*.agent.md` symlinks for Copilot's naming convention.

### Cursor

```bash
git clone https://github.com/maxilylm/ml-automation-plugin.git
```

Place the repo (or symlink it) where Cursor discovers plugins. The `.cursor-plugin/plugin.json` manifest handles discovery automatically.

### Codex

1. **Clone the repository:**
   ```bash
   git clone https://github.com/maxilylm/ml-automation-plugin.git ~/.codex/ml-automation
   ```

2. **Create the skills symlink:**
   ```bash
   mkdir -p ~/.agents/skills
   ln -s ~/.codex/ml-automation/commands ~/.agents/skills/ml-automation
   ```

   **Windows (PowerShell):**
   ```powershell
   New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.agents\skills"
   cmd /c mklink /J "$env:USERPROFILE\.agents\skills\ml-automation" "$env:USERPROFILE\.codex\ml-automation\commands"
   ```

3. **Restart Codex** to discover the skills.

### OpenCode

1. **Clone the repository:**
   ```bash
   git clone https://github.com/maxilylm/ml-automation-plugin.git ~/.config/opencode/ml-automation
   ```

2. **Register the plugin:**
   ```bash
   mkdir -p ~/.config/opencode/plugins
   ln -s ~/.config/opencode/ml-automation/.opencode/plugins/ml-automation.js ~/.config/opencode/plugins/ml-automation.js
   ```

3. **Symlink skills:**
   ```bash
   mkdir -p ~/.config/opencode/skills
   ln -s ~/.config/opencode/ml-automation/commands ~/.config/opencode/skills/ml-automation
   ```

4. **Restart OpenCode.** Verify by asking: "what ML automation skills do you have?"

## Updating

| Platform | Command |
|----------|---------|
| Claude Code | Run `/plugin` → Update marketplace |
| GitHub Copilot | `copilot plugin update ml-automation` or `cd <plugin-path> && git pull` |
| Codex | `cd ~/.codex/ml-automation && git pull` |
| OpenCode | `cd ~/.config/opencode/ml-automation && git pull` |
| Cursor | `cd <plugin-path> && git pull` |

Skills update instantly through symlinks — no reinstall needed.

## Requirements

- Python 3.9+
- Common ML libraries: pandas, scikit-learn, matplotlib, seaborn
- Optional: streamlit, fastapi, docker

---

## Changelog

### v1.5 (v1.5.0 — v1.5.3)

**Evaluation Framework**

Built-in eval definitions for all 12 skills and agent routing, with a CLI runner for tracking quality across iterations:

```bash
python evals/eval_runner.py --init-iteration 1       # Initialize
python evals/eval_runner.py --record eda sales-data-exploration sales-correlation pass
python evals/eval_runner.py --summary                 # View results
python evals/eval_runner.py --compare 1 2             # Compare iterations
```

30 evals with 78 assertions covering all 12 skills + 15 routing accuracy tests. Eval definitions versioned in `evals/`. Iteration outputs gitignored.

**Routing Overhaul (33% → 100% accuracy)**

Assigner restructured with 6-level priority system. Domain-specific compound rules now fire before generic implementation keywords:

1. Multi-agent coordination → `orchestrator`
2. Domain-specific rules → MLOps, ML theory, feature engineering, data investigation
3. Review/analysis → appropriate analyst
4. Diagnostic language → `developer`
5. Implementation keywords → `developer`
6. Fallback → `orchestrator`

Plus contextual disambiguation for ambiguous terms (pipeline, features, model, accuracy).

**Skill Quality Overhaul** — All 12 SKILL.md files expanded from ~20 to ~50 lines with report bus integration, version feature awareness, agent coordination, flags, and command file pointers.

**EDA Improvements** — Date range validation, categorical label consistency checks, near-duplicate detection, domain-invalid range detection.

**Preprocessing Improvements** — Active leakage scan (feature-target >0.90, feature-feature >0.95), pipeline artifact saving, updated checklist.

**Agent Description Trimming** — YAML descriptions reduced ~90% (1800-2640 → 121-265 chars). Reduces routing token cost.

**Bug Fixes** — 6 phantom agent references fixed, `/report` "status" alias conflict resolved, hardcoded OpenCode paths replaced with multi-platform discovery.

**Orchestrator Update** — Now aware of all 9 agents and v1.2-v1.4 features.

### v1.4.0

**Iterative Self-Check Loops** — Every workflow stage validates its output before proceeding. Configurable max iterations.

```bash
/team coldstart data.csv --max-check 3   # Allow 3 retry iterations
/team coldstart data.csv --max-check 0   # Skip self-checks
```

**Pre-Stage Reflection** — Domain expert agents plan the approach by reading prior reports and lessons before each stage.

| Stage | Reflector |
|-------|-----------|
| Analysis | ml-theory-advisor |
| Processing | ml-theory-advisor |
| Modeling | ml-theory-advisor |
| Evaluation | ml-theory-advisor |
| Dashboard | frontend-ux-analyst |
| Production | mlops-engineer |

**Lessons Learned System** — Persistent knowledge base recording mistakes, solutions, and patterns. Auto-deduplicated, consulted before each stage.

```bash
/status --lessons   # View all recorded lessons
```

**New Utility Functions** — `save_lesson()`, `load_lessons()`, `validate_stage_output()`, `save_stage_plan()`, `load_stage_plan()`

### v1.3.0

**MLOps Registry Layer** — Convention-based registries for models, features, experiments, and data versions. Task-type aware (classification, regression, MMM, segmentation, time series).

```bash
/registry                          # Summary of all registries
/registry models --champion        # Show champion model details
/registry features --domain mmm    # Filter features by domain
/registry lineage model_id         # Trace full lineage
```

### v1.3.1

**Report Verification Checkpoints** — Orchestrator enforces report saving after parallel stages. Re-spawns agents once if reports missing.

**Grounded Dashboard Creation** — Developer reads actual reports before writing dashboard code. No placeholder templates.

**Dashboard Smoke Test Loop** — Validates syntax, placeholders, and undefined variables (max 2 iterations).

### v1.2.0

**Shared Report Bus** — All 10 agents communicate via JSON reports. Convention-based discovery (`*_report.json`), multi-platform support.

**Parallel Agent Execution** — Post-EDA and post-training agent groups run concurrently.

**`/status` Command** — Unified workflow visibility.

```bash
/status              # Show full workflow status
/status --agent eda  # Show specific agent's report
/status --pending    # Show only pending work
```

### v1.2.1

**Reflection Gates** — Pre-execution validation gates (3 checkpoints) evaluate strategy before proceeding. Configurable max iterations.

```bash
/team coldstart data.csv --max-reflect 0   # Skip gates
/team coldstart data.csv --max-reflect 3   # Allow 3 iterations
```

## License

MIT
