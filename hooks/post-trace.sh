#!/bin/bash
# Post-agent traceability hook
# Appends a trace event whenever an agent completes (SubagentStop)

set -e

AGENT_TYPE="${CLAUDE_TOOL_ARG_SUBAGENT_TYPE:-unknown}"
TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

# Build JSONL entry
TRACE_ENTRY=$(python3 -c "
import json
entry = {
    'timestamp': '$TIMESTAMP',
    'event_type': 'agent_complete',
    'agent': '$AGENT_TYPE',
    'action': 'Agent completed execution',
    'details': {},
    'inputs_summary': '',
    'outputs_summary': '',
    'duration_ms': None,
    'stage': '',
    'command': '',
}
print(json.dumps(entry, default=str))
" 2>/dev/null || echo '{}')

# Append to all platform trace logs
for DIR in .claude .cursor .codex .opencode; do
    if [ -d "$DIR" ] || [ "$DIR" = ".claude" ]; then
        mkdir -p "$DIR"
        echo "$TRACE_ENTRY" >> "$DIR/traceability-log.jsonl"
    fi
done

echo "[Trace Hook] Logged agent completion: $AGENT_TYPE"
