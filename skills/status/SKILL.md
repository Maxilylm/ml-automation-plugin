---
name: status
description: "Show unified workflow status by reading all agent reports. Displays completed agents, pending work, and cross-agent insights."
---

# Workflow Status Skill

Read and display the current state of all agent reports in the shared report bus.

## Implementation

Scan all report directories for `*_report.json` files and present a unified view of:
1. Which agents have completed work (with summaries)
2. Which agents are pending (with dependency information)
3. Cross-agent insights (recommendations targeting other agents)

Use `ml_utils.get_workflow_status()` if available, otherwise scan directories manually.
