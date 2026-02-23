# ML Automation Plugin

End-to-end machine learning automation workflow for AI coding assistants. Takes you from raw data to deployed model with interactive dashboard — orchestrated by specialized AI agents.

Supports **Claude Code**, **Cursor**, **Codex**, and **OpenCode**.

## What's New in v1.2.0

### Shared Report Bus
All 10 agents now communicate through a shared JSON report system. Each agent reads prior reports on startup and writes its own report on completion, enabling:
- Cross-agent insights and recommendations
- Convention-based discovery (`*_report.json` files)
- Multi-platform support (`.claude/reports/`, `.cursor/reports/`, etc.)

### Parallel Agent Execution
Workflow commands now spawn agents in parallel where dependencies allow:
- **Post-EDA group**: feature-engineering-analyst + ml-theory-advisor + frontend-ux-analyst run concurrently
- **Post-Training review group**: brutal-code-reviewer + ml-theory-advisor + frontend-ux-analyst run concurrently

### `/status` Command
New slash command for unified workflow visibility:
```bash
/status              # Show full workflow status
/status --agent eda  # Show specific agent's report
/status --pending    # Show only pending work
```

### Report Validation Hooks
Automatic schema validation after each agent completes, ensuring reports are well-formed and discoverable.

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

### Hooks

- **Pre-commit**: Python syntax check, secrets detection, test coverage validation
- **Pre-deploy**: Deployment readiness checks (files, health endpoints, Docker, env vars)
- **Post-EDA**: Extract metrics, flag data quality issues
- **Post-dashboard**: Validate Streamlit syntax, check components, generate run scripts
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

## Updating

| Platform | Command |
|----------|---------|
| Claude Code | Run `/plugin` → Update marketplace |
| Codex | `cd ~/.codex/ml-automation && git pull` |
| OpenCode | `cd ~/.config/opencode/ml-automation && git pull` |
| Cursor | `cd <plugin-path> && git pull` |

Skills update instantly through symlinks — no reinstall needed.

## Requirements

- Python 3.9+
- Common ML libraries: pandas, scikit-learn, matplotlib, seaborn
- Optional: streamlit, fastapi, docker

## License

MIT
