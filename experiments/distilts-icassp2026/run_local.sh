#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

# The host image exposes a different system PyTorch through LD_LIBRARY_PATH.
# Removing it ensures that the isolated environment loads its own matching libs.
exec env -u LD_LIBRARY_PATH "$repo_dir/.venv/bin/python" -u run.py "$@"
