#!/bin/bash
# Post-EDA hook: Runs after exploratory data analysis completes
# Usage: Called after EDA stage in team-coldstart workflow

set -e

DATA_PATH="${1:-}"
OUTPUT_DIR="${2:-reports}"

echo "[Post-EDA Hook] Processing EDA results..."

# Create reports directory if needed
mkdir -p "$OUTPUT_DIR"

# 1. Check if EDA report was generated
if [ -f "$OUTPUT_DIR/eda_report.md" ] || [ -f "$OUTPUT_DIR/eda_report.html" ]; then
    echo "  - EDA report found"

    # 2. Extract key metrics for dashboard
    if [ -f "$OUTPUT_DIR/eda_report.md" ]; then
        echo "  - Extracting metrics for dashboard..."

        # Count rows, columns, missing values mentioned
        grep -oP 'Rows: \K\d+' "$OUTPUT_DIR/eda_report.md" 2>/dev/null > "$OUTPUT_DIR/.eda_metrics" || true
        grep -oP 'Columns: \K\d+' "$OUTPUT_DIR/eda_report.md" 2>/dev/null >> "$OUTPUT_DIR/.eda_metrics" || true
    fi
fi

# 3. Notify about data quality issues
if [ -f "$OUTPUT_DIR/eda_report.md" ]; then
    MISSING_COUNT=$(grep -c "Missing" "$OUTPUT_DIR/eda_report.md" 2>/dev/null || echo "0")
    if [ "$MISSING_COUNT" -gt 5 ]; then
        echo "  - WARNING: $MISSING_COUNT columns with missing values detected"
        echo "    Consider reviewing data quality before proceeding"
    fi
fi

# 4. Log completion
echo "$(date '+%Y-%m-%d %H:%M:%S') - EDA completed for $DATA_PATH" >> .claude/workflow.log 2>/dev/null || true

echo "[Post-EDA Hook] EDA processing complete!"
