# Installing ML Automation Plugin for OpenCode

## Prerequisites

- [OpenCode.ai](https://opencode.ai) installed
- Git installed
- Python 3.9+ with pandas, scikit-learn, matplotlib, seaborn

## Installation Steps

### 1. Clone ML Automation

```bash
git clone https://github.com/maxilylm/ml-automation-plugin.git ~/.config/opencode/ml-automation
```

### 2. Register the Plugin

Create a symlink so OpenCode discovers the plugin:

```bash
mkdir -p ~/.config/opencode/plugins
rm -f ~/.config/opencode/plugins/ml-automation.js
ln -s ~/.config/opencode/ml-automation/.opencode/plugins/ml-automation.js ~/.config/opencode/plugins/ml-automation.js
```

### 3. Symlink Skills

Create a symlink so OpenCode's native skill tool discovers ML automation skills:

```bash
mkdir -p ~/.config/opencode/skills
rm -rf ~/.config/opencode/skills/ml-automation
ln -s ~/.config/opencode/ml-automation/plugins/ml-automation/commands ~/.config/opencode/skills/ml-automation
```

### 4. Restart OpenCode

Restart OpenCode. The plugin will automatically inject ML automation context.

Verify by asking: "what ML automation skills do you have?"

## Usage

### Finding Skills

Use OpenCode's native `skill` tool to list available skills:

```
use skill tool to list skills
```

### Loading a Skill

Use OpenCode's native `skill` tool to load a specific skill:

```
use skill tool to load ml-automation/eda
```

### Available Skills

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

## Updating

```bash
cd ~/.config/opencode/ml-automation
git pull
```

## Troubleshooting

### Plugin not loading

1. Check plugin symlink: `ls -l ~/.config/opencode/plugins/ml-automation.js`
2. Check source exists: `ls ~/.config/opencode/ml-automation/.opencode/plugins/ml-automation.js`
3. Check OpenCode logs for errors

### Skills not found

1. Check skills symlink: `ls -l ~/.config/opencode/skills/ml-automation`
2. Verify it points to: `~/.config/opencode/ml-automation/plugins/ml-automation/commands`
3. Use `skill` tool to list what's discovered

### Tool Mapping

When skills reference Claude Code tools:
- `TodoWrite` → `update_plan`
- `Task` with subagents → `@mention` syntax
- `Skill` tool → OpenCode's native `skill` tool
- File operations → your native tools

## Uninstalling

```bash
rm ~/.config/opencode/plugins/ml-automation.js
rm ~/.config/opencode/skills/ml-automation
rm -rf ~/.config/opencode/ml-automation
```
