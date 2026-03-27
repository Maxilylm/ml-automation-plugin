# Modular Extension Architecture Design

**Date:** 2026-03-27
**Status:** Approved
**Branch:** `feature/modular-extension-architecture`

## Problem

The ml-automation plugin is a monolithic system where all agents, skills, commands, and routing rules are hardcoded within a single plugin. Adding new domain-specific capabilities (e.g., Media Mix Modeling, Snowflake integration) requires modifying the core — increasing bloat, coupling, and maintenance burden.

## Goal

Make the core plugin extensible so that **separate plugins** can add agents, skills, and commands that integrate seamlessly with core workflows — without modifying the core itself.

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Distribution model | Separate repos, separate Claude Code plugins | Independent versioning, install only what you need |
| Extension workflows | Extensions bring own commands AND hook into core workflows | Maximum flexibility for extension authors |
| Discovery mechanism | Self-describing agents via frontmatter metadata | Single source of truth, no new file formats, follows convention-over-configuration |
| Shared utilities | Core owns ml_utils, extensions depend on it | Simplest approach, natural prerequisite relationship |
| Workflow injection | Named hook points at stage boundaries | Explicit, predictable, debuggable |

## Section 1: Extension Protocol — Frontmatter Conventions

The extension protocol is a set of frontmatter fields that extension plugins use to declare their relationship to the core.

### Agent Frontmatter (extension agents)

Extension agents add three new optional fields:

```yaml
---
name: mmm-analyst
description: Media Mix Modeling analysis and optimization
model: sonnet
color: "#FF6B35"
tools: [Read, Write, Bash(*), Glob, Grep]
# --- Extension Protocol fields ---
extends: ml-automation
routing_keywords: [mmm, media mix, marketing mix, channel attribution, adstock]
hooks_into:
  - after-eda
  - after-evaluation
---
```

**Fields:**

- `extends: ml-automation` — required for the core to recognize an extension agent
- `routing_keywords` — list of terms the assigner uses to route tasks to this agent
- `hooks_into` — list of core hook point names where this agent should be spawned

### Skill Frontmatter (extension skills)

```yaml
---
name: mmm-coldstart
description: End-to-end Media Mix Modeling workflow
user_invocable: true
aliases: [mmm, media-mix]
extends: ml-automation
uses_core_agents: [eda-analyst, developer, mlops-engineer]
---
```

**Fields:**

- `extends: ml-automation` — same protocol field as agents
- `uses_core_agents` — documentation-only list of core agents this skill depends on

### Rules

- `extends: ml-automation` is required for the core to recognize an extension component
- `routing_keywords` feeds the assigner's dynamic routing
- `hooks_into` must reference valid hook point names defined by the core (see Section 2)
- `uses_core_agents` is not enforced at runtime
- Core agents do not need any frontmatter changes

## Section 2: Core Hook Points

Named stage boundaries in core workflows where extension agents can inject execution. This list is a versioned contract.

### Hook Points for `/team-coldstart`

| Hook Point | Fires After | Before | Purpose |
|---|---|---|---|
| `after-init` | Stage 1: Initialize | Stage 2: Analysis | Post-setup, data validated |
| `after-eda` | Stage 2: EDA + Analysis | Stage 3: Processing | Data understood, pre-processing |
| `after-feature-engineering` | Feature engineering complete | Preprocessing | Features designed, before pipeline |
| `after-preprocessing` | Stage 3: Processing | Stage 4: Modeling | Clean data ready |
| `before-training` | Pre-stage reflection | Training execution | Last chance to influence model strategy |
| `after-training` | Stage 4: Modeling | Stage 5: Evaluation | Model trained, pre-evaluation |
| `after-evaluation` | Stage 5: Evaluation | Stage 6: Dashboard | Metrics available |
| `after-dashboard` | Stage 6: Dashboard | Stage 7: Productionalize | Visuals ready |
| `before-deploy` | Stage 7: Productionalize | Stage 8: Deploy | Pre-deployment gate |
| `after-deploy` | Stage 8: Deploy | Stage 9: Finalize | Post-deployment |

### Hook Points for Other Core Commands

| Command | Hook Points |
|---|---|
| `/eda` | `after-eda` |
| `/train` | `before-training`, `after-training` |
| `/evaluate` | `after-evaluation` |
| `/deploy` | `before-deploy`, `after-deploy` |
| `/preprocess` | `after-preprocessing` |

### Hook Execution Contract

When the core reaches a hook point:

1. **Discover** — scan all installed plugin agent files for `hooks_into` containing this hook point
2. **Collect** — gather all matching extension agents
3. **Spawn sequentially** — run each extension agent, passing it the current report bus state (all prior `*_report.json` files)
4. **Extension agent writes its report** — using `save_agent_report()` convention, making its output available to downstream stages
5. **Continue** — core proceeds to the next stage

### Versioning

Hook points are versioned with the core plugin version. Renaming or removing a hook point is a major version bump with a deprecation notice. Extensions declare `extends: ml-automation` without a version constraint — hook point names are the implicit API.

## Section 3: Dynamic Discovery in Orchestrator and Assigner

### Discovery Mechanism

1. Scan plugin directories for all `agents/*.md` files across installed plugins
2. Parse YAML frontmatter — extract `name`, `extends`, `routing_keywords`, `hooks_into`
3. Build runtime agent table — core agents + any agent with `extends: ml-automation`

This logic lives in the orchestrator and assigner agent prompts as instructions. These agents already use `Glob` and `Read` tools — they scan at workflow start.

### Assigner Changes

Current assigner has 6 priority tiers of hardcoded rules. New behavior:

1. Read all `agents/*.md` from all installed plugins
2. Build routing table:
   - Core agents: use existing priority rules (unchanged)
   - Extension agents: match `routing_keywords` against ticket text
3. Priority ordering:
   - Core priority tiers 1-6 remain unchanged
   - Extension agents slot into a new **tier 3.5** (after domain-specific compound rules, before review tasks)
   - If multiple extension agents match, prefer the one with more keyword matches
4. Fallback remains: orchestrator

Extension agents never override core routing. They fill gaps for domain-specific requests that no core agent handles.

### Orchestrator Changes

1. At workflow start, scan for all agents with `extends: ml-automation`
2. Merge into available agents table alongside core agents
3. When reaching a hook point in any workflow:
   a. Find agents with `hooks_into` containing this hook point
   b. Spawn each one, passing current report bus context
   c. Wait for completion, collect reports
   d. Continue to next stage
4. For direct dispatch (non-workflow): use the merged agent table

### What Stays Hardcoded

- Core agent names and their roles
- Priority tier ordering in the assigner (core agents always take precedence)
- Stage ordering within core workflows (extensions inject between stages, never reorder)
- Reflection gate assignments (ml-theory-advisor) — extensions can add gates but not replace core ones

## Section 4: Extension Plugin Structure

### Directory Structure Example (`ml-automation-mmm`)

```
ml-automation-mmm/
├── .claude-plugin/
│   └── plugin.json
├── agents/
│   ├── mmm-analyst.md           # extends: ml-automation
│   └── mmm-optimizer.md         # extends: ml-automation
├── skills/
│   ├── mmm-coldstart/
│   │   └── SKILL.md             # extends: ml-automation
│   └── mmm-evaluate/
│       └── SKILL.md
├── commands/
│   ├── mmm-coldstart.md
│   └── mmm-evaluate.md
├── templates/
│   └── mmm_utils.py             # imports from ml_utils
└── hooks/
    └── hooks.json
```

### Extension `plugin.json`

```json
{
  "name": "ml-automation-mmm",
  "version": "0.1.0",
  "description": "Media Mix Modeling extension for ml-automation",
  "author": { "name": "Maximo Lorenzo y Losada" },
  "keywords": ["mmm", "media-mix", "marketing", "ml-automation-extension"],
  "dependencies": {
    "ml-automation": ">=1.8.0"
  }
}
```

The `dependencies` field is documentation-only (Claude Code doesn't enforce plugin dependencies) but signals to users and future tooling that the core is required.

### Extension Commands Using Core Agents

Extension commands spawn core agents directly since all plugins are loaded into the same session:

```markdown
## Stage 1: Data Exploration
Spawn **eda-analyst** (core) to profile the marketing dataset.

## Stage 2: MMM Analysis
Spawn **mmm-analyst** (extension) to run media mix decomposition.

## Stage 3: Model Training
Spawn **developer** (core) to implement the MMM model using mmm_utils.py.
```

### Extension Utils Pattern

`templates/mmm_utils.py` adds domain-specific functions and imports core utilities:

```python
from ml_utils import save_agent_report, load_agent_report, log_experiment

def compute_adstock(series, decay_rate=0.5):
    """Apply adstock transformation to media spend series."""
    ...

def decompose_contributions(model, X):
    """Decompose model predictions into channel contributions."""
    ...
```

This works because the core's `ml_utils.py` is already copied into the user's project before any extension runs.

## Section 5: Scope of Core Changes (This PR)

### Files to Modify

1. **`agents/orchestrator.md`** — Replace static agent table with dynamic discovery instructions. Add hook point execution logic.
2. **`agents/assigner.md`** — Add dynamic routing for extension agents via `routing_keywords`. Add priority tier 3.5.
3. **`commands/team-coldstart.md`** — Insert hook point markers at all 10 stage boundaries with discovery and spawn instructions.
4. **`commands/eda.md`**, **`train.md`**, **`evaluate.md`**, **`deploy.md`**, **`preprocess.md`** — Add relevant hook points.
5. **`templates/ml_utils.py`** — Add `discover_extension_agents()` and `get_hooked_agents(hook_point)` helper functions.

### Files to Create

6. **`docs/extension-protocol.md`** — Extension author guide covering frontmatter conventions, hook point reference, plugin structure, and utilities usage.

### Files NOT Changed

- Existing agent definitions (other than orchestrator/assigner)
- Existing skill definitions
- `plugin.json` — version bump only
- Hooks — unchanged
- Evals — new evals for extension discovery added separately
