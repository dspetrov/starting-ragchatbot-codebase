#!/bin/bash
# Check code formatting without modifying files

echo "Checking import order with isort..."
uv run isort --check-only backend/
ISORT_EXIT=$?

echo ""
echo "Checking code formatting with black..."
uv run black --check backend/
BLACK_EXIT=$?

if [ $ISORT_EXIT -eq 0 ] && [ $BLACK_EXIT -eq 0 ]; then
    echo ""
    echo "✓ All formatting checks passed!"
    exit 0
else
    echo ""
    echo "✗ Formatting issues found. Run ./format.sh to fix."
    exit 1
fi
