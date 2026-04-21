#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python -m app.worker --fixture tests/fixtures/frame_ingested_example.json
