# Installing ML Automation Plugin for Codex

Enable ML automation skills in Codex via native skill discovery. Clone and symlink.

## Prerequisites

- Git
- Python 3.9+ with pandas, scikit-learn, matplotlib, seaborn

## Installation

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

## Available Skills

| Skill | Description |
|-------|-------------|
| `ml-automation/eda` | Exploratory data analysis |
| `ml-automation/preprocess` | Data processing pipeline (leakage-safe) |
| `ml-automation/train` | Train ML models with proper validation |
| `ml-automation/evaluate` | Model evaluation with visualizations |
| `ml-automation/deploy` | Deploy to Docker, Snowflake, AWS, or GCP |
| `ml-automation/report` | Generate EDA, model, drift, or project reports |
| `ml-automation/test` | Generate and run tests (80% coverage threshold) |
| `ml-automation/team-coldstart` | Full pipeline: raw data to deployed dashboard |
| `ml-automation/team-analyze` | Quick multi-agent data analysis |
| `ml-automation/team-review` | Multi-agent code review |
| `ml-automation/status` | View workflow progress, agent states, and report bus contents |

> **New in v1.2.0:** The `/status` command provides real-time visibility into workflow execution, including agent states, parallel group progress, and shared report bus contents.

## Tool Mapping

When skills reference Claude Code tools, substitute Codex equivalents:
- `Task` with subagents → Codex's agent system
- `Skill` tool → Codex's native skill discovery
- `Read`, `Write`, `Edit`, `Bash` → Your native tools

## Verify

```bash
ls -la ~/.agents/skills/ml-automation
```

You should see a symlink pointing to the plugin's commands directory.

## Updating

```bash
cd ~/.codex/ml-automation && git pull
```

Skills update instantly through the symlink.

## Uninstalling

```bash
rm ~/.agents/skills/ml-automation
```

Optionally delete the clone: `rm -rf ~/.codex/ml-automation`
