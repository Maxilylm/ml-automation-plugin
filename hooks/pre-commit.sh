#!/bin/bash
# Pre-commit hook: Validates code before allowing commits
# Usage: Called automatically by Claude Code before git commit

set -e

echo "[Pre-Commit Hook] Running validations..."

# Check if we're in a data project
if [ -d "src" ] || [ -d "dashboard" ]; then

    # 1. Check Python syntax
    if command -v python3 &> /dev/null; then
        echo "  - Checking Python syntax..."
        for f in $(find . -name "*.py" -not -path "./.venv/*" -not -path "./venv/*" 2>/dev/null); do
            python3 -m py_compile "$f" 2>/dev/null || {
                echo "    ERROR: Syntax error in $f"
                exit 1
            }
        done
        echo "    OK"
    fi

    # 2. Check for secrets/credentials
    echo "  - Checking for potential secrets..."
    SECRETS_PATTERN='(password|secret|api_key|token|credential)[\s]*[=:][\s]*["\047][^"\047]+'
    if grep -rEi "$SECRETS_PATTERN" --include="*.py" --include="*.json" --include="*.yaml" --include="*.yml" . 2>/dev/null | grep -v ".claude/" | grep -v "example" | head -1; then
        echo "    WARNING: Potential secrets detected. Review before committing."
    else
        echo "    OK"
    fi

    # 3. Check test coverage exists for src files
    if [ -d "src" ] && [ -d "tests" ]; then
        echo "  - Checking test coverage..."
        SRC_COUNT=$(find src -name "*.py" -not -name "__init__.py" 2>/dev/null | wc -l)
        TEST_COUNT=$(find tests -name "test_*.py" 2>/dev/null | wc -l)
        if [ "$SRC_COUNT" -gt 0 ] && [ "$TEST_COUNT" -eq 0 ]; then
            echo "    WARNING: No tests found for $SRC_COUNT source files"
        else
            echo "    OK ($TEST_COUNT test files)"
        fi
    fi

    # 4. Validate Streamlit dashboard syntax
    if [ -f "dashboard/app.py" ]; then
        echo "  - Validating Streamlit dashboard..."
        python3 -m py_compile dashboard/app.py 2>/dev/null && echo "    OK" || {
            echo "    ERROR: Dashboard has syntax errors"
            exit 1
        }
    fi
fi

echo "[Pre-Commit Hook] All validations passed!"
