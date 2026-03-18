This is the ml-automation plugin — an end-to-end ML workflow for AI coding assistants.

## Key Architecture

- 10 agents in `agents/` coordinate via a shared JSON report bus
- 12 skills in `skills/*/SKILL.md` provide guided workflows
- 12 commands in `commands/*.md` have full implementation details
- Hooks in `hooks/` enforce quality gates
- `templates/ml_utils.py` is the shared utility library (~1300 lines)

## Conventions

- Agents communicate via `*_report.json` files using `save_agent_report()`
- Skills point to commands via "Full Specification" sections
- ml_utils.py is copied into user projects, not imported from the plugin
- All platforms supported: Claude Code, Cursor, Codex, OpenCode, Copilot
- Version bumps must update ALL manifests: `.claude-plugin/`, `.cursor-plugin/`, `.copilot/`, `.opencode/`

## When editing skills or agents

- SKILL.md files should be 40-60 lines with: When to Use, Workflow, Report Bus, Full Specification pointer
- Agent descriptions in YAML frontmatter must be under 300 chars (routing token budget)
- Never reference phantom agents — only these 10 exist: developer, pr-approver, brutal-code-reviewer, frontend-ux-analyst, eda-analyst, ml-theory-advisor, mlops-engineer, feature-engineering-analyst, orchestrator, assigner
