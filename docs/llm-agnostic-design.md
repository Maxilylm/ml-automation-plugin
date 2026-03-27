# LLM/System Agnostic Architecture Design

## Current State

The plugin already supports 5 platforms:
- Claude Code (`.claude/`)
- Cursor (`.cursor/`)
- Codex (`.codex/`)
- OpenCode (`.opencode/`)
- GitHub Copilot (`.copilot/`)

Multi-platform support exists for: report directories, MLOps registry, lessons learned, and traceability logs.

## Problem Statement

While file-based persistence is platform-agnostic, the **agent definitions** and **tool declarations** are tightly coupled to Claude Code's plugin format. Making the system work with different LLM backends (GPT-4, Gemini, Llama, Mistral) or different orchestration frameworks requires an abstraction layer.

## Proposed Architecture: Provider-Neutral Manifests

### Layer 1: Universal Agent Manifest (agent-manifest.yaml)

```yaml
agents:
  eda-analyst:
    role: "Data exploration and quality assessment"
    capabilities:
      - data_loading
      - statistical_analysis
      - visualization
      - quality_assessment
    inputs:
      - type: file
        formats: [csv, xlsx, json, parquet]
      - type: parameter
        name: target_col
        optional: true
    outputs:
      - type: report
        format: json
        schema: eda_report_schema
      - type: visualization
        formats: [png, html]
    tools_needed:
      - file_read
      - file_write
      - code_execution
      - shell_command

  ml-theory-advisor:
    role: "ML methodology validation and reflection"
    capabilities:
      - leakage_detection
      - methodology_review
      - assumption_validation
    inputs:
      - type: report
        from_agents: [eda-analyst, developer]
    outputs:
      - type: review
        format: json
        schema: reflection_report_schema
    tools_needed:
      - file_read
      - code_execution
```

### Layer 2: Platform Adapters

Each platform gets an adapter that translates universal manifests:

```
agent-manifest.yaml
    ├── adapters/claude-code/    → .claude-plugin/ format
    ├── adapters/cursor/         → .cursor/ agent format
    ├── adapters/copilot/        → .copilot/ agent.md format
    ├── adapters/langchain/      → LangChain agent definitions
    ├── adapters/crewai/         → CrewAI agent definitions
    └── adapters/autogen/        → AutoGen agent definitions
```

### Layer 3: Tool Abstraction

Map generic tool capabilities to platform-specific implementations:

```yaml
tool_mapping:
  file_read:
    claude_code: "Read"
    cursor: "read_file"
    langchain: "ReadFileTool"
    generic: "open(path).read()"

  code_execution:
    claude_code: "Bash"
    cursor: "run_command"
    langchain: "PythonREPLTool"
    generic: "subprocess.run()"

  file_write:
    claude_code: "Write"
    cursor: "write_file"
    langchain: "WriteFileTool"
    generic: "open(path, 'w').write()"
```

### Layer 4: LLM Backend Abstraction

Already partially solved by each platform choosing its own LLM. For self-hosted scenarios:

```yaml
llm_config:
  provider: "anthropic"  # or "openai", "local", "bedrock", "vertex"
  model: "claude-sonnet-4-6"
  fallback_model: "claude-haiku-4-5"

  # Provider-specific
  anthropic:
    api_key_env: "ANTHROPIC_API_KEY"
  openai:
    api_key_env: "OPENAI_API_KEY"
    model: "gpt-4o"
  local:
    endpoint: "http://localhost:8080/v1"
    model: "llama-3.1-70b"
```

## Implementation Plan

### Phase 1: Extract Universal Manifests (Low effort)
- Create `agent-manifest.yaml` from existing agent definitions
- No functional change — just documentation

### Phase 2: Build Claude Code Adapter (Medium effort)
- Adapter reads `agent-manifest.yaml` → generates `.claude-plugin/` format
- Validate round-trip: manifest → plugin format → works identically

### Phase 3: Build Additional Adapters (Per platform)
- LangChain/CrewAI adapters for self-hosted scenarios
- Each adapter is a script: `python adapt.py --platform langchain`

### Phase 4: Snowflake/Cloud Integration
- Snowpark adapter for running agents inside Snowflake
- AWS Bedrock adapter for managed LLM access
- GCP Vertex adapter for Gemini backend

## What Stays Platform-Specific

- Hooks (each platform has its own hook system)
- IDE integration (keybindings, UI panels)
- Auth/permissions model
- Real-time streaming behavior

## What Becomes Universal

- Agent capabilities and roles
- Report bus and persistence layer (ml_utils.py)
- MLOps registry
- Traceability log
- Business plausibility checks
- Methodology document generation
