#!/bin/bash
# Run essential code quality checks (formatting and linting)

set -e  # Exit on first error

echo "================================================"
echo "Running Code Quality Checks"
echo "================================================"

echo ""
echo "1. Checking import order..."
echo "------------------------------------------------"
uv run isort --check-only backend/

echo ""
echo "2. Checking code formatting..."
echo "------------------------------------------------"
uv run black --check backend/

echo ""
echo "3. Running linter..."
echo "------------------------------------------------"
uv run flake8 backend/ --exclude=backend/tests --max-line-length=100 --extend-ignore=E203,W503

echo ""
echo "================================================"
echo "✓ All quality checks passed!"
echo "================================================"
echo ""
echo "Note: Run ./type-check.sh for optional type checking with mypy"
