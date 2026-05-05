#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python -m pip install -e . || true
