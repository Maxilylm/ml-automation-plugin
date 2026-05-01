"""Smoke tests for spark-core — validate plugin layout invariants."""

import json
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_validity():
    """Validate that plugin.json is valid JSON and contains required fields."""
    manifest_path = PLUGIN_ROOT / ".cortex-plugin" / "plugin.json"

    # File must exist
    assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

    # Must be valid JSON
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Required fields
    assert "name" in manifest, "Missing 'name' field in plugin.json"
    assert "version" in manifest, "Missing 'version' field in plugin.json"
    assert "description" in manifest, "Missing 'description' field in plugin.json"

    # Plugin name must start with 'spark-'
    assert manifest["name"].startswith("spark-"), (
        f"Plugin name '{manifest['name']}' must start with 'spark-'"
    )


def test_agents_md_referential_integrity():
    """Validate AGENTS.md consistency with agents/ and skills/ directories."""
    agents_md_path = PLUGIN_ROOT / "AGENTS.md"
    assert agents_md_path.exists(), f"AGENTS.md not found at {agents_md_path}"

    with open(agents_md_path) as f:
        agents_content = f.read()

    # Extract agent names from "## Available Agents" section
    agents_dir = PLUGIN_ROOT / "agents"
    agent_files = set(f.stem for f in agents_dir.glob("*.md"))

    # Extract skill names from "## Available Skills" section
    skills_dir = PLUGIN_ROOT / "skills"
    skill_dirs = set(d.name for d in skills_dir.iterdir() if d.is_dir())

    # Each agent file mentioned in agents/ should have content in AGENTS.md
    for agent_file in agent_files:
        assert agent_file in agents_content, (
            f"Agent '{agent_file}' exists in agents/ but is not referenced in AGENTS.md"
        )

    # Each skill dir mentioned in skills/ should have content in AGENTS.md
    for skill_dir in skill_dirs:
        assert skill_dir in agents_content, (
            f"Skill '{skill_dir}' exists in skills/ but is not referenced in AGENTS.md"
        )
