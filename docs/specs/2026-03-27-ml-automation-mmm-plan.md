# ml-automation-mmm Extension Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `ml-automation-mmm` Claude Code extension plugin — a self-contained MMM package following the Extension Protocol, with 3 agents, 8 skills/commands, knowledge base, scripts, and core workflow hooks.

**Architecture:** New repo scaffolded as a Claude Code plugin. Agents declare `extends: ml-automation` with `routing_keywords` and `hooks_into` frontmatter. Commands are ported from the source BLEND360 repo with Extension Protocol adjustments (Stage 0 ml_utils check, report bus integration). Knowledge base and scripts copied as-is.

**Tech Stack:** Markdown (agents, skills, commands), Python (mmm_utils.py, scripts), YAML (KB index), JSON (plugin manifest, hooks)

**Spec:** `docs/specs/2026-03-27-ml-automation-mmm-extension-design.md`

**Source content:** All agent bodies, command content, and skill descriptions were provided by the user in the conversation. The implementer should reference the spec for structure and the task descriptions below for exact content.

**Working directory:** The new repo will be created at `~/Documents/ml-automation-mmm/` (sibling to the core plugin).

**Prerequisites:**
- Core plugin v1.8.0 must be on the `ml-automation-core` repo (done — pushed earlier this session)
- Source BLEND360 repo content: all agent/command content was pasted by the user in this conversation session. The implementer has it available.
- Core plugin's Extension Protocol (dynamic discovery in orchestrator/assigner) handles routing automatically — no additional wiring needed in the core.

---

### Task 1: Initialize Repository and Plugin Manifest

Create the new repo directory structure and plugin.json.

**Files:**
- Create: `~/Documents/ml-automation-mmm/.claude-plugin/plugin.json`
- Create: `~/Documents/ml-automation-mmm/.gitignore`

- [ ] **Step 1: Create the repo directory and initialize git**

```bash
mkdir -p ~/Documents/ml-automation-mmm
cd ~/Documents/ml-automation-mmm
git init
```

- [ ] **Step 2: Create the plugin manifest**

Create `.claude-plugin/plugin.json`:

```json
{
  "name": "ml-automation-mmm",
  "version": "0.1.0",
  "description": "Media Mix Modeling extension for ml-automation. Bayesian MMM with adstock, saturation, channel attribution, and client-ready reporting.",
  "author": { "name": "Maximo Lorenzo y Losada" },
  "repository": "https://github.com/Maxilylm/ml-automation-mmm",
  "license": "MIT",
  "keywords": ["mmm", "media-mix", "bayesian", "marketing", "ml-automation-extension", "pymc", "channel-attribution"],
  "dependencies": {
    "ml-automation": ">=1.8.0"
  }
}
```

- [ ] **Step 3: Create .gitignore**

```
# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/

# Environment (keep environment.yaml and requirements.txt, ignore installed env)
knowledge_base/mmm/BMMM-env/envs/
knowledge_base/mmm/BMMM-env/lib/
knowledge_base/mmm/BMMM-env/bin/
knowledge_base/mmm/BMMM-env/include/
knowledge_base/mmm/BMMM-env/ENV_META.json
*.conda
.venv/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Runtime artifacts (created in user projects, not plugin)
reports/
feedback/
models/
```

- [ ] **Step 4: Create empty directory structure**

```bash
mkdir -p agents skills commands knowledge_base/mmm scripts/mmm templates hooks
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: initialize ml-automation-mmm plugin scaffold"
```

---

### Task 2: Create the 3 Agent Definitions

Write the agent markdown files with Extension Protocol frontmatter and full body content from the source.

**Files:**
- Create: `~/Documents/ml-automation-mmm/agents/mmm-methodologist.md`
- Create: `~/Documents/ml-automation-mmm/agents/mmm-validator.md`
- Create: `~/Documents/ml-automation-mmm/agents/mmm-communicator.md`

- [ ] **Step 1: Create mmm-methodologist.md**

Write the file with this frontmatter (replacing the source frontmatter):

```yaml
---
name: mmm-methodologist
description: "Design and review Marketing Mix Models using the internal MMM knowledge pack. Selects modeling choices (adstock/saturation, hierarchy, priors), aligns validation strategy, and drafts MMM design memos."
model: sonnet
color: "#FF6B35"
tools: [Read, Write, Bash(*), Glob, Grep]
extends: ml-automation
routing_keywords: [mmm, media mix, marketing mix, adstock, saturation, bayesian mmm, channel attribution, media optimization, marketing budget, roas]
hooks_into:
  - after-eda
---
```

Then add the Relevance Gate section followed by the full source body content (starting from "# MMM Methodologist"). The relevance gate section:

```markdown
## Relevance Gate (when running at a hook point)

When invoked at a core workflow hook point (not via direct command):
1. Read the EDA report at `.claude/reports/eda-analyst_report.json`
2. Check if the dataset contains marketing/media spend columns by looking for keywords in column names: spend, cost, impressions, clicks, media, channel, campaign, ad, marketing
3. If NO marketing columns detected — write a skip report and exit:
   ```python
   from ml_utils import save_agent_report
   save_agent_report("mmm-methodologist", {
       "status": "skipped",
       "reason": "No marketing spend columns detected in dataset"
   })
   ```
4. If marketing columns detected: proceed with full methodology review
```

Then the full source body (the "**When to invoke:**" section, "## Retrieval", "## Deliverables", "## Next step suggestion" sections from the source).

- [ ] **Step 2: Create mmm-validator.md**

Frontmatter:

```yaml
---
name: mmm-validator
description: "Validate MMM implementation and results against the internal validation checklist. Focuses on diagnostics, plausibility, leakage, stability, and calibration alignment."
model: sonnet
color: "#E85D04"
tools: [Read, Write, Bash(*), Glob, Grep]
extends: ml-automation
routing_keywords: [mmm validation, mmm diagnostics, mmm plausibility, roas check, contribution check, mmm verify]
hooks_into:
  - after-evaluation
---
```

Relevance gate:

```markdown
## Relevance Gate (when running at a hook point)

When invoked at a core workflow hook point (not via direct command):
1. Check if MMM artifacts exist:
   - Look for `reports/mmm/` directory (MMM project reports)
   - Look for PyMC/Bayesian model artifacts in `models/` (*.nc, *mmm*, *bayesian*)
2. If NO MMM artifacts found — write a skip report and exit:
   ```python
   from ml_utils import save_agent_report
   save_agent_report("mmm-validator", {
       "status": "skipped",
       "reason": "No MMM model artifacts found"
   })
   ```
3. If MMM artifacts found: proceed with full validation
```

Then the source body ("# MMM Validator", Inputs, Use, Output).

- [ ] **Step 3: Create mmm-communicator.md**

Frontmatter:

```yaml
---
name: mmm-communicator
description: "Convert MMM technical results into client-ready business narratives using internal delivery patterns (exec summary, key findings, recommendations, caveats)."
model: sonnet
color: "#DC2F02"
tools: [Read, Write, Glob, Grep]
extends: ml-automation
routing_keywords: [mmm report, mmm client report, mmm narrative, mmm delivery, mmm presentation]
---
```

No hooks_into (MMM-specific only). Then the source body ("# MMM Communicator", Use, Output).

- [ ] **Step 4: Commit**

```bash
git add agents/
git commit -m "feat: add 3 MMM agents with Extension Protocol frontmatter"
```

---

### Task 3: Create the 8 Skill Definitions

Write SKILL.md files for all 8 skills following core conventions.

**Files:**
- Create: `~/Documents/ml-automation-mmm/skills/start-mmm-project/SKILL.md`
- Create: `~/Documents/ml-automation-mmm/skills/train-bmmm/SKILL.md`
- Create: `~/Documents/ml-automation-mmm/skills/bmmm-smoke/SKILL.md`
- Create: `~/Documents/ml-automation-mmm/skills/final-mmm-report/SKILL.md`
- Create: `~/Documents/ml-automation-mmm/skills/mmm-communicate/SKILL.md`
- Create: `~/Documents/ml-automation-mmm/skills/mmm-methodologist/SKILL.md`
- Create: `~/Documents/ml-automation-mmm/skills/ensure-bmmm-env/SKILL.md`
- Create: `~/Documents/ml-automation-mmm/skills/self-assess/SKILL.md`

- [ ] **Step 1: Create all 8 SKILL.md files**

Each SKILL.md follows this pattern (45-66 lines max, matching core conventions):

**start-mmm-project/SKILL.md:**
```yaml
---
name: start-mmm-project
description: "Bootstrap a new MMM project — generates problem brief, data readiness checklist, and design memo from the MMM knowledge pack"
user_invocable: true
aliases: [mmm start, start mmm, bootstrap mmm]
extends: ml-automation
---
```
Body: 2-3 sentence description + "## Full Specification" pointing to `commands/start-mmm-project.md`.

**train-bmmm/SKILL.md:**
```yaml
---
name: train-bmmm
description: "Run full Bayesian MMM training with MCMC sampling using the managed environment runner"
user_invocable: true
aliases: [train mmm, mmm train, bmmm train]
extends: ml-automation
---
```

**bmmm-smoke/SKILL.md:**
```yaml
---
name: bmmm-smoke
description: "Fast smoke test for the Bayesian MMM toolchain — verifies environment, imports, and sampler without full MCMC"
user_invocable: true
aliases: [mmm smoke, smoke test mmm]
extends: ml-automation
---
```

**final-mmm-report/SKILL.md:**
```yaml
---
name: final-mmm-report
description: "Generate a consolidated final report for an MMM project from all pipeline artifacts"
user_invocable: true
aliases: [mmm report, mmm final report]
extends: ml-automation
---
```

**mmm-communicate/SKILL.md:**
```yaml
---
name: mmm-communicate
description: "Convert MMM technical results into a client-ready business narrative using the mmm-communicator agent"
user_invocable: true
aliases: [mmm communicate, mmm client report, mmm narrative]
extends: ml-automation
---
```

**mmm-methodologist/SKILL.md:**
```yaml
---
name: mmm-methodologist
description: "Gate command for MMM methodologist review — scores complexity signals and decides whether full review is needed"
user_invocable: true
aliases: [mmm methodologist, methodologist review, mmm review]
extends: ml-automation
---
```

**ensure-bmmm-env/SKILL.md:**
```yaml
---
name: ensure-bmmm-env
description: "Ensure the Bayesian MMM conda/venv environment exists with all required dependencies"
user_invocable: true
aliases: [bmmm env, mmm env, ensure mmm env, ensure bmmm env]
extends: ml-automation
---
```

**self-assess/SKILL.md:**
```yaml
---
name: self-assess
description: "Aggregate self-assessment feedback files across MMM runs and surface recurring patterns as improvement recommendations"
user_invocable: true
aliases: [self assess, review feedback, mmm feedback]
extends: ml-automation
---
```

- [ ] **Step 2: Commit**

```bash
git add skills/
git commit -m "feat: add 8 MMM skill definitions with Extension Protocol frontmatter"
```

---

### Task 4: Create the 8 Command Files

Write command markdown files with the full source content, adding Stage 0 ml_utils check where needed.

**Files:**
- Create: `~/Documents/ml-automation-mmm/commands/start-mmm-project.md`
- Create: `~/Documents/ml-automation-mmm/commands/train-bmmm.md`
- Create: `~/Documents/ml-automation-mmm/commands/bmmm-smoke.md`
- Create: `~/Documents/ml-automation-mmm/commands/final-mmm-report.md`
- Create: `~/Documents/ml-automation-mmm/commands/mmm-communicate.md`
- Create: `~/Documents/ml-automation-mmm/commands/mmm-methodologist.md`
- Create: `~/Documents/ml-automation-mmm/commands/ensure-bmmm-env.md`
- Create: `~/Documents/ml-automation-mmm/commands/self-assess.md`

- [ ] **Step 1: Create all 8 command files**

**Source content:** The full content for all 8 commands was pasted by the user earlier in this conversation session (search for the command names: `/start-mmm-project`, `/train-bmmm`, `/bmmm-smoke`, etc.). Each command file uses that source content verbatim, with these modifications:

**For `/start-mmm-project` only** — add a Stage 0 before the existing workflow:

```markdown
### Stage 0: Ensure Core Utilities

Before generating artifacts, ensure core utilities are available:
1. Check if `ml_utils.py` exists in the project's `src/` directory
2. If missing, scan for the core plugin's copy:
   - `.claude/plugins/*/templates/ml_utils.py`
   - `~/.claude/plugins/*/templates/ml_utils.py`
   - Copy it to `src/ml_utils.py`
3. Check if `mmm_utils.py` exists in the project's `src/` directory
4. If missing, copy from this plugin's `templates/mmm_utils.py` to `src/mmm_utils.py`
```

**For `/train-bmmm`** — add the same Stage 0 before step 1.

**For `/bmmm-smoke` and `/final-mmm-report`** — add a lighter check at the top: "If `ml_utils.py` is not present in `src/`, copy it from the core plugin." These commands read reports but may need `save_agent_report()`.

**For `/mmm-communicate`, `/mmm-methodologist`, `/ensure-bmmm-env`, `/self-assess`** — use source content as-is. These commands either spawn agents (which handle their own utils) or operate on feedback files without needing ml_utils.

- [ ] **Step 2: Commit**

```bash
git add commands/
git commit -m "feat: add 8 MMM commands with Extension Protocol Stage 0"
```

---

### Task 5: Create mmm_utils.py

Write the MMM-specific utility library.

**Files:**
- Create: `~/Documents/ml-automation-mmm/templates/mmm_utils.py`

- [ ] **Step 1: Write mmm_utils.py**

```python
"""
MMM-specific utilities for the ml-automation-mmm extension plugin.

Requires ml_utils.py from the ml-automation core plugin to be present
in the same directory (copied via Stage 0 of MMM commands).
"""

from ml_utils import save_agent_report, load_agent_report, log_experiment


# --- Relevance Detection ---

MMM_KEYWORDS = {
    "spend", "cost", "impressions", "clicks", "media", "channel",
    "campaign", "ad", "marketing", "adstock", "cpm", "cpc", "ctr",
    "roas", "roi", "grp", "trp", "reach", "frequency",
}


def detect_mmm_relevance(source):
    """Check if a dataset has marketing/media columns suggesting MMM relevance.

    Args:
        source: list of column name strings, OR a dict (EDA report) with
                a 'columns' or 'column_types' key containing column names.

    Returns:
        dict with 'is_mmm': bool, 'matched_columns': list of matched column names
    """
    # Extract column names from various input formats
    if isinstance(source, dict):
        # EDA report format — try common keys
        column_names = (
            source.get("columns")
            or list(source.get("column_types", {}).get("numerical", []))
            + list(source.get("column_types", {}).get("categorical", []))
            or []
        )
    else:
        column_names = list(source)

    matched = []
    for col in column_names:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        for kw in MMM_KEYWORDS:
            if kw in col_lower:
                matched.append(col)
                break
    return {
        "is_mmm": len(matched) > 0,
        "matched_columns": matched,
    }


# --- Adstock Transformations ---

def compute_adstock(series, decay_rate=0.5):
    """Apply geometric adstock transformation to a media spend series.

    Args:
        series: array-like of spend values (ordered by time)
        decay_rate: float between 0 and 1 (higher = longer carryover)

    Returns:
        list of adstocked values
    """
    adstocked = [0.0] * len(series)
    adstocked[0] = float(series[0])
    for i in range(1, len(series)):
        adstocked[i] = float(series[i]) + decay_rate * adstocked[i - 1]
    return adstocked


# --- Contribution Decomposition ---

def decompose_contributions(coefficients, X_columns, X_means):
    """Compute channel contribution shares from model coefficients.

    Args:
        coefficients: dict of {channel_name: coefficient_value}
        X_columns: list of channel names
        X_means: dict of {channel_name: mean_value}

    Returns:
        dict of {channel_name: contribution_share} (sums to 1.0)
    """
    raw_contributions = {}
    for col in X_columns:
        if col in coefficients and col in X_means:
            raw_contributions[col] = abs(coefficients[col] * X_means[col])

    total = sum(raw_contributions.values())
    if total == 0:
        return {col: 0.0 for col in X_columns}

    return {col: val / total for col, val in raw_contributions.items()}
```

- [ ] **Step 2: Commit**

```bash
git add templates/mmm_utils.py
git commit -m "feat: add mmm_utils.py with relevance detection, adstock, and contribution decomposition"
```

---

### Task 6: Create hooks.json and Placeholder Knowledge Base

Set up the hooks file and knowledge base directory structure.

**Files:**
- Create: `~/Documents/ml-automation-mmm/hooks/hooks.json`
- Create: `~/Documents/ml-automation-mmm/knowledge_base/mmm/00_CORE.md` (placeholder)
- Create: `~/Documents/ml-automation-mmm/knowledge_base/mmm/02_KB_INDEX.yml` (placeholder)

- [ ] **Step 1: Create empty hooks.json**

```json
{
  "hooks": {}
}
```

- [ ] **Step 2: Create knowledge base placeholders**

The knowledge base content comes from the BLEND360 source repo. Since we cannot access it directly, create placeholder files that document what needs to be copied:

`knowledge_base/mmm/00_CORE.md`:
```markdown
# MMM Core Knowledge

> **TODO:** Copy the full 00_CORE.md content from the BLEND360/spark-bayesian-mmm-agent repository.
> This file contains the core MMM modeling rules, priors, adstock/saturation conventions.
```

`knowledge_base/mmm/02_KB_INDEX.yml`:
```yaml
# MMM Knowledge Base Index
# TODO: Copy from BLEND360/spark-bayesian-mmm-agent repository.
# This file indexes all chunks by tag for retrieval.
chunks: []
```

Create directories with .gitkeep files so they are tracked:
```bash
mkdir -p knowledge_base/mmm/chunks && touch knowledge_base/mmm/chunks/.gitkeep
mkdir -p knowledge_base/mmm/templates
mkdir -p knowledge_base/mmm/playbook && touch knowledge_base/mmm/playbook/.gitkeep
mkdir -p knowledge_base/mmm/BMMM-env && touch knowledge_base/mmm/BMMM-env/.gitkeep
mkdir -p scripts/mmm
```

Create placeholder for templates that commands reference:

`knowledge_base/mmm/templates/problem_brief_mmm.md`:
```markdown
# Problem Brief: {{project_name}}
> TODO: Copy template from source repo
```

`knowledge_base/mmm/templates/data_readiness_mmm.md`:
```markdown
# Data Readiness: {{project_name}}
> TODO: Copy template from source repo
```

`knowledge_base/mmm/templates/design_memo_mmm.md`:
```markdown
# Design Memo: {{project_name}}
> TODO: Copy template from source repo
```

`knowledge_base/mmm/templates/delivery_summary_mmm.md`:
```markdown
# Delivery Summary: {{project_name}}
> TODO: Copy template from source repo
```

`knowledge_base/mmm/validation_checklist_mmm.yaml`:
```yaml
# MMM Validation Checklist
# TODO: Copy from source repo
checks: []
```

- [ ] **Step 3: Create script placeholders**

`scripts/mmm/build.py`:
```python
"""MMM Training Entrypoint — TODO: Copy from BLEND360/spark-bayesian-mmm-agent."""
raise NotImplementedError("Copy build.py from source repo")
```

`scripts/ensure_domain_env.py`:
```python
"""Domain Environment Manager — TODO: Copy from source repo."""
raise NotImplementedError("Copy ensure_domain_env.py from source repo")
```

`scripts/ensure_bmmm_env.py`:
```python
"""BMMM Environment Setup — TODO: Copy from source repo."""
raise NotImplementedError("Copy ensure_bmmm_env.py from source repo")
```

`scripts/run_in_env.py`:
```python
"""Run commands inside managed environment — TODO: Copy from source repo."""
raise NotImplementedError("Copy run_in_env.py from source repo")
```

- [ ] **Step 4: Commit**

```bash
git add hooks/ knowledge_base/ scripts/
git commit -m "feat: add hooks.json, knowledge base placeholders, and script stubs"
```

---

### Task 7: Create README and Publish Repository

Write the README and push to GitHub.

**Files:**
- Create: `~/Documents/ml-automation-mmm/README.md`

- [ ] **Step 1: Write README.md**

```markdown
# ml-automation-mmm

Media Mix Modeling extension for [ml-automation](https://github.com/Maxilylm/ml-automation-core).

## Prerequisites

- [ml-automation](https://github.com/Maxilylm/ml-automation-core) core plugin (>= v1.8.0)
- Claude Code CLI

## Installation

```bash
claude plugin add /path/to/ml-automation-mmm
```

## What's Included

### Agents

| Agent | Purpose | Hooks Into |
|---|---|---|
| `mmm-methodologist` | Design and review MMM models | `after-eda` |
| `mmm-validator` | Validate MMM results | `after-evaluation` |
| `mmm-communicator` | Client-ready business narratives | *(MMM workflows only)* |

### Commands

| Command | Purpose |
|---|---|
| `/start-mmm-project` | Bootstrap MMM project (brief, data readiness, design memo) |
| `/mmm-methodologist` | Gate command — scores complexity, routes to review |
| `/bmmm-smoke` | Fast smoke test for Bayesian MMM toolchain |
| `/train-bmmm` | Full MCMC training run |
| `/final-mmm-report` | Consolidate artifacts into final report |
| `/mmm-communicate` | Convert results to client narrative |
| `/ensure-bmmm-env` | Ensure Python environment exists |
| `/self-assess` | Aggregate feedback across runs |

## How It Integrates

When installed alongside the core plugin:

1. **Automatic routing** — Tasks mentioning "MMM", "media mix", "channel attribution" etc. are routed to MMM agents via the core assigner (Priority 2.5)
2. **Core workflow hooks** — When running `/team-coldstart` on marketing data:
   - `mmm-methodologist` fires at `after-eda` to review data for MMM suitability
   - `mmm-validator` fires at `after-evaluation` to validate MMM-specific metrics
3. **Core agent reuse** — MMM commands use core agents (eda-analyst, developer, mlops-engineer) directly

## Setup Knowledge Base

After cloning, copy the knowledge base content from the source repository:

```bash
# Copy from BLEND360/spark-bayesian-mmm-agent (requires access)
cp -r /path/to/source/knowledge_base/mmm/* knowledge_base/mmm/
cp -r /path/to/source/scripts/* scripts/
```

## License

MIT
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with installation, commands, and integration guide"
```

- [ ] **Step 3: Create GitHub repo and push**

```bash
cd ~/Documents/ml-automation-mmm
gh repo create Maxilylm/ml-automation-mmm --public --description "Media Mix Modeling extension for ml-automation" --source . --push
```
