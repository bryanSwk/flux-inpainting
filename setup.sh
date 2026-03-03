#!/usr/bin/env bash
set -e

# Create uv venv with Python 3.11
uv venv .venv --python 3.11

# Install dependencies
uv pip install -r requirements.txt

echo "Ensure you are logged in: hf auth login"
