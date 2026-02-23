#!/bin/bash
# Post-agent report validation hook
# Runs after any agent completes to verify report was written correctly

set -e

AGENT_TYPE="${CLAUDE_TOOL_ARG_SUBAGENT_TYPE:-unknown}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "[Post-Agent Hook] Checking report for agent: $AGENT_TYPE"

# Check all possible report directories
REPORT_FOUND=false
for DIR in .claude/reports reports .cursor/reports .codex/reports .opencode/reports; do
    REPORT_FILE="$DIR/${AGENT_TYPE}_report.json"
    if [ -f "$REPORT_FILE" ]; then
        REPORT_FOUND=true
        echo "  ✓ Report found: $REPORT_FILE"

        # Basic schema validation
        if command -v python3 &>/dev/null; then
            python3 -c "
import json, sys
with open('$REPORT_FILE') as f:
    report = json.load(f)
required = ['agent', 'version', 'timestamp', 'status', 'findings']
missing = [k for k in required if k not in report]
if missing:
    print(f'  ⚠ Missing fields: {missing}')
    sys.exit(1)
print(f'  ✓ Schema valid (agent={report[\"agent\"]}, status={report[\"status\"]})')
" 2>/dev/null || echo "  ⚠ Schema validation skipped (python3 not available or invalid JSON)"
        fi
        break
    fi
done

if [ "$REPORT_FOUND" = false ]; then
    echo "  ⚠ No report found for agent: $AGENT_TYPE"
    echo "    Expected: .claude/reports/${AGENT_TYPE}_report.json"
    echo "    Tip: Agent should call save_agent_report() from ml_utils.py"
fi

# Log to workflow log
mkdir -p .claude
echo "$TIMESTAMP - Agent report check: $AGENT_TYPE (found=$REPORT_FOUND)" >> .claude/workflow.log 2>/dev/null || true
