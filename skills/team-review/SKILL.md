---
name: team-review
description: Coordinate a multi-agent code review with specialized reviewers for different aspects of the code.
---

# Team Review

## When to Use
- Running comprehensive code reviews with multiple specialists
- Reviewing ML pipelines, data processing, or deployment code
- Getting feedback from code quality, ML theory, and UX perspectives

## Reviewers

| Agent | Scope | Always? |
|-------|-------|---------|
| `pr-approver` | Overall quality, approval authority | Yes |
| `brutal-code-reviewer` | Maintainability, test coverage, AI-friendliness | Yes |
| `ml-theory-advisor` | ML methodology, leakage, validation | If ML code |
| `mlops-engineer` | Deployment readiness, Snowflake code | If deployment code |
| `frontend-ux-analyst` | UI/UX patterns, accessibility | If frontend code |

## Workflow

1. **Identify changes** — `git diff --name-only` or `--pr <number>`
2. **Spawn reviewers in parallel** — Use Task tool with multiple calls
3. **Collect feedback** — Each reviewer writes a report to the report bus
4. **Synthesize** — Present unified findings sorted by severity (critical > warning > info)
5. **Decision** — Approve, request changes, or block merge

## Severity Levels

| Level | Meaning | Blocks Merge? |
|-------|---------|---------------|
| Critical | Security, data leakage, correctness bugs | Yes |
| Warning | Code quality, missing tests, style issues | No (but should fix) |
| Info | Suggestions, minor improvements | No |

## Report Bus Integration (v1.2.0)

Each reviewer writes its findings to the report bus. The coordination agent reads all reviewer reports and produces a unified summary.

## Full Specification

See `commands/team-review.md` for complete reviewer configurations, output templates, and permission controls.
