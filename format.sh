#!/bin/bash
# Format code with black and isort

echo "Running isort..."
uv run isort backend/

echo "Running black..."
uv run black backend/

echo "✓ Code formatting complete!"
