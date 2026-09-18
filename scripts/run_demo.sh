#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m embeduino ingest --reset --strategy structure
python -m embeduino ask "What does digitalWrite do?"
python -m embeduino eval
