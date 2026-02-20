#!/bin/bash
# Post-Workflow hook: Runs after team-coldstart completes
# Usage: Called at the end of the full workflow

set -e

PROJECT_NAME="${1:-Data Project}"
MODE="${2:-ml}"  # ml or analysis

echo ""
echo "=============================================="
echo "[Workflow Complete] $PROJECT_NAME"
echo "=============================================="
echo ""

# 1. Summarize created files
echo "Files Created:"
echo "--------------"

if [ -d "src" ]; then
    echo "Source Code:"
    ls -la src/*.py 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
fi

if [ -d "api" ]; then
    echo "API:"
    ls -la api/*.py 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
fi

if [ -d "dashboard" ]; then
    echo "Dashboard:"
    ls -la dashboard/*.py 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
fi

if [ -d "tests" ]; then
    echo "Tests:"
    find tests -name "*.py" 2>/dev/null | wc -l | xargs -I {} echo "  {} test files"
fi

if [ -d "reports" ]; then
    echo "Reports:"
    ls -la reports/*.md reports/*.html 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
fi

echo ""

# 2. Show key metrics
echo "Metrics:"
echo "--------"

# Line count
if [ -d "src" ]; then
    SRC_LINES=$(find src -name "*.py" -exec cat {} \; 2>/dev/null | wc -l)
    echo "  Source code: $SRC_LINES lines"
fi

# Test coverage hint
if [ -d "tests" ]; then
    TEST_FILES=$(find tests -name "test_*.py" 2>/dev/null | wc -l)
    echo "  Test files: $TEST_FILES"
fi

echo ""

# 3. Quick start commands
echo "Quick Start:"
echo "------------"

if [ -f "dashboard/app.py" ]; then
    echo "  Run dashboard:     streamlit run dashboard/app.py"
fi

if [ -f "api/app.py" ]; then
    echo "  Run API:           uvicorn api.app:app --reload"
fi

if [ -f "docker-compose.yml" ]; then
    echo "  Run with Docker:   docker-compose up"
fi

if [ -d "tests" ]; then
    echo "  Run tests:         pytest tests/ -v"
fi

echo ""

# 4. URLs (if deployed)
echo "Access URLs:"
echo "------------"
echo "  Dashboard:  http://localhost:8501"
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"

echo ""

# 5. Log workflow completion
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
mkdir -p .claude
echo "$TIMESTAMP - Workflow completed: $PROJECT_NAME (mode: $MODE)" >> .claude/workflow.log

# 6. Generate completion summary file
cat > .claude/WORKFLOW_COMPLETE.md << EOF
# Workflow Complete

**Project**: $PROJECT_NAME
**Mode**: $MODE
**Completed**: $TIMESTAMP

## Quick Start

\`\`\`bash
# Run dashboard
streamlit run dashboard/app.py

# Run API
uvicorn api.app:app --reload

# Run with Docker
docker-compose up

# Run tests
pytest tests/ -v
\`\`\`

## URLs

- Dashboard: http://localhost:8501
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Next Steps

1. Review the dashboard at http://localhost:8501
2. Customize visualizations in dashboard/app.py
3. Add more tests if needed
4. Configure monitoring and alerts
5. Set up CI/CD pipeline
EOF

echo "[Workflow Complete] Summary saved to .claude/WORKFLOW_COMPLETE.md"
echo ""
