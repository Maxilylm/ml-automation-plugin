#!/bin/bash
# Post-Dashboard hook: Validates and prepares Streamlit dashboard
# Usage: Called after dashboard creation in team-coldstart workflow

set -e

DASHBOARD_PATH="${1:-dashboard/app.py}"

echo "[Post-Dashboard Hook] Validating dashboard..."

# 1. Check dashboard file exists
if [ ! -f "$DASHBOARD_PATH" ]; then
    echo "  - ERROR: Dashboard not found at $DASHBOARD_PATH"
    exit 1
fi

echo "  - Dashboard file found"

# 2. Validate Python syntax
echo "  - Checking syntax..."
python3 -m py_compile "$DASHBOARD_PATH" || {
    echo "  - ERROR: Dashboard has syntax errors"
    exit 1
}
echo "    OK"

# 3. Check for required Streamlit imports
echo "  - Checking Streamlit imports..."
if ! grep -q "import streamlit" "$DASHBOARD_PATH"; then
    echo "  - WARNING: Missing 'import streamlit' statement"
fi

# 4. Check for page config (best practice)
if ! grep -q "st.set_page_config" "$DASHBOARD_PATH"; then
    echo "  - TIP: Consider adding st.set_page_config() for better UX"
fi

# 5. Check for required components
echo "  - Checking dashboard components..."
COMPONENTS=("st.title\|st.header" "st.metric\|st.dataframe" "st.plotly_chart\|st.pyplot\|st.altair_chart")
for comp in "${COMPONENTS[@]}"; do
    if grep -qE "$comp" "$DASHBOARD_PATH"; then
        echo "    Found: $(echo $comp | sed 's/\\|/ or /g')"
    fi
done

# 6. Create requirements snippet for dashboard
if [ ! -f "dashboard/requirements.txt" ]; then
    echo "  - Generating dashboard requirements..."
    mkdir -p dashboard
    cat > dashboard/requirements.txt << 'EOF'
streamlit>=1.28.0
pandas>=2.0.0
plotly>=5.18.0
numpy>=1.24.0
EOF
    echo "    Created dashboard/requirements.txt"
fi

# 7. Create run script
if [ ! -f "dashboard/run.sh" ]; then
    echo "  - Creating run script..."
    cat > dashboard/run.sh << 'EOF'
#!/bin/bash
# Run the Streamlit dashboard locally
streamlit run app.py --server.port 8501 --server.address localhost
EOF
    chmod +x dashboard/run.sh
    echo "    Created dashboard/run.sh"
fi

# 8. Log completion
echo "$(date '+%Y-%m-%d %H:%M:%S') - Dashboard validated: $DASHBOARD_PATH" >> .claude/workflow.log 2>/dev/null || true

echo "[Post-Dashboard Hook] Dashboard ready!"
echo ""
echo "  To run locally: cd dashboard && streamlit run app.py"
echo "  Or use: ./dashboard/run.sh"
