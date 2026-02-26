#!/bin/bash
# Post-stage validation hook
# Runs stage-specific validation checks using ml_utils.py
# Usage: bash post-stage-check.sh <stage_name> [context_json]

STAGE="${1:-}"
CONTEXT="${2:-{}}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

if [ -z "$STAGE" ]; then
    echo "[Post-Stage Check] No stage specified, skipping"
    exit 0
fi

echo "[Post-Stage Check] Validating stage: $STAGE"

# Find ml_utils.py
ML_UTILS=""
for p in src/ml_utils.py ml_utils.py templates/ml_utils.py; do
    if [ -f "$p" ]; then
        ML_UTILS="$p"
        break
    fi
done

if [ -z "$ML_UTILS" ]; then
    echo "  ⚠ ml_utils.py not found, skipping validation"
    exit 0
fi

ML_UTILS_DIR=$(dirname "$ML_UTILS")

# Run validation — pass context via env var to avoid shell injection
RESULT=0
CONTEXT_JSON="$CONTEXT" python3 -c "
import sys, json, os
sys.path.insert(0, '$ML_UTILS_DIR')
from ml_utils import validate_stage_output

context = json.loads(os.environ.get('CONTEXT_JSON', '{}'))
passed, errors = validate_stage_output('$STAGE', context)

if passed:
    print('  ✓ Stage validation PASSED')
else:
    print('  ✗ Stage validation FAILED:')
    for err in errors:
        print(f'    - {err}')
    sys.exit(1)
" || RESULT=$?

# Log
mkdir -p .claude
echo "$TIMESTAMP - Stage check: $STAGE (result=$RESULT)" >> .claude/workflow.log 2>/dev/null || true

exit $RESULT
