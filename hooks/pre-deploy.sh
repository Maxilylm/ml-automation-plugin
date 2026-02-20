#!/bin/bash
# Pre-Deploy hook: Validates everything is ready for deployment
# Usage: Called before deployment in team-coldstart workflow

set -e

DEPLOY_TARGET="${1:-local}"

echo "[Pre-Deploy Hook] Validating deployment readiness for: $DEPLOY_TARGET"

ERRORS=0
WARNINGS=0

# 1. Check required files exist
echo "  - Checking required files..."

REQUIRED_FILES=("requirements.txt")
if [ "$DEPLOY_TARGET" = "local" ] || [ "$DEPLOY_TARGET" = "docker" ]; then
    REQUIRED_FILES+=("Dockerfile" "docker-compose.yml")
fi

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "    OK: $file"
    else
        echo "    MISSING: $file"
        ((ERRORS++))
    fi
done

# 2. Check API health endpoint exists (if API)
if [ -f "api/app.py" ]; then
    echo "  - Checking API configuration..."
    if grep -q "/health" api/app.py; then
        echo "    OK: Health endpoint found"
    else
        echo "    WARNING: No /health endpoint found"
        ((WARNINGS++))
    fi
fi

# 3. Check dashboard exists
if [ -f "dashboard/app.py" ]; then
    echo "  - Dashboard found"
else
    echo "  - WARNING: No dashboard found at dashboard/app.py"
    ((WARNINGS++))
fi

# 4. Run tests if they exist
if [ -d "tests" ]; then
    echo "  - Running tests..."
    if command -v pytest &> /dev/null; then
        if pytest tests/ -q --tb=no 2>/dev/null; then
            echo "    OK: Tests passed"
        else
            echo "    ERROR: Tests failed"
            ((ERRORS++))
        fi
    else
        echo "    SKIP: pytest not installed"
    fi
fi

# 5. Check environment variables for specific targets
echo "  - Checking environment configuration..."
case "$DEPLOY_TARGET" in
    snowflake)
        REQUIRED_VARS=("SNOWFLAKE_ACCOUNT" "SNOWFLAKE_USER" "SNOWFLAKE_DATABASE")
        for var in "${REQUIRED_VARS[@]}"; do
            if [ -z "${!var}" ]; then
                echo "    MISSING: $var environment variable"
                ((ERRORS++))
            fi
        done
        ;;
    aws)
        if ! command -v aws &> /dev/null; then
            echo "    WARNING: AWS CLI not installed"
            ((WARNINGS++))
        fi
        ;;
    gcp)
        if ! command -v gcloud &> /dev/null; then
            echo "    WARNING: gcloud CLI not installed"
            ((WARNINGS++))
        fi
        ;;
    local|docker)
        if ! command -v docker &> /dev/null; then
            echo "    ERROR: Docker not installed"
            ((ERRORS++))
        else
            echo "    OK: Docker available"
        fi
        ;;
esac

# 6. Check Docker build (if applicable)
if [ -f "Dockerfile" ] && command -v docker &> /dev/null; then
    echo "  - Validating Dockerfile syntax..."
    if docker build --check . 2>/dev/null || docker build -f Dockerfile --target=syntax-check . 2>/dev/null; then
        echo "    OK: Dockerfile valid"
    else
        # Just warn, don't fail - some Docker versions don't support --check
        echo "    SKIP: Could not validate Dockerfile"
    fi
fi

# 7. Summary
echo ""
echo "[Pre-Deploy Hook] Validation complete"
echo "  Errors: $ERRORS"
echo "  Warnings: $WARNINGS"

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "  BLOCKED: Fix $ERRORS error(s) before deploying"
    exit 1
fi

if [ $WARNINGS -gt 0 ]; then
    echo ""
    echo "  PROCEED WITH CAUTION: $WARNINGS warning(s) detected"
fi

echo ""
echo "[Pre-Deploy Hook] Ready for deployment to $DEPLOY_TARGET!"
