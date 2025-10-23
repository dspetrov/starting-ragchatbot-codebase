#!/bin/bash
# Run linting checks with flake8

echo "Running flake8..."
uv run flake8 backend/ --exclude=backend/tests --max-line-length=100 --extend-ignore=E203,W503

if [ $? -eq 0 ]; then
    echo "✓ Linting checks passed!"
else
    echo "✗ Linting issues found. Please fix the issues above."
    exit 1
fi
