#!/bin/bash
# Run type checking with mypy (optional)

echo "Running type checker..."
uv run mypy backend/ --exclude backend/tests

if [ $? -eq 0 ]; then
    echo "✓ Type checking passed!"
else
    echo ""
    echo "Note: Type errors found. Type checking is optional and can be improved gradually."
    exit 1
fi
