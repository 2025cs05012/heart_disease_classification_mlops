#!/usr/bin/env bash
# Reproduce the full pipeline from a *clean* venv built only from
# requirements.txt. This is the offline equivalent of the CI run and
# is the proof for production-readiness clause 1:
#
#     "All scripts must execute from a clean setup using the
#      requirements file."
#
# What this script does (each step aborts on first error):
#
#   1. create an ephemeral venv at $VENV (default: /tmp/heart-clean-venv)
#   2. pip install -r requirements.txt   (only the pinned file)
#   3. python -m src.data.preprocess     (rebuild cleaned CSV)
#   4. pytest unit_test/ -q              (run the 35-case suite)
#   5. python -m src.models.train --no-mlflow
#   6. summarise: model size + reports/metrics.json
#
# Usage:
#   bash scripts/verify_clean_setup.sh                # use /tmp venv, keep it
#   VENV=/tmp/my-venv bash scripts/verify_clean_setup.sh
#   bash scripts/verify_clean_setup.sh --rm           # delete venv on exit
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-/tmp/heart-clean-venv}"
REMOVE_ON_EXIT=0
[[ "${1:-}" == "--rm" ]] && REMOVE_ON_EXIT=1

step() { echo; echo "==> $*"; }
cleanup() {
    if [[ $REMOVE_ON_EXIT -eq 1 ]]; then
        echo
        echo "==> Removing ephemeral venv at $VENV"
        rm -rf "$VENV"
    fi
}
trap cleanup EXIT

step "0/6  Sanity checks"
command -v python3 >/dev/null || { echo "python3 not on PATH"; exit 1; }
PY_VERSION="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "    python3 = $PY_VERSION"
if [[ "$PY_VERSION" != "3.11" ]]; then
    echo "    WARNING: project pins 3.11 (.python-version); you're on $PY_VERSION."
    echo "             Continuing, but mismatches with pinned wheels may surface."
fi

step "1/6  Create clean venv at $VENV"
if [[ -d "$VENV" ]]; then
    echo "    venv exists; recreating to guarantee clean state"
    rm -rf "$VENV"
fi
python3 -m venv "$VENV"
PIP="$VENV/bin/pip"
PY="$VENV/bin/python"
"$PIP" install --quiet --upgrade pip

step "2/6  Install dependencies from requirements.txt"
"$PIP" install -r requirements.txt
echo "    installed $($PIP list --format=freeze | wc -l | tr -d ' ') packages"

step "3/6  Rebuild cleaned dataset"
"$PY" -m src.data.preprocess
test -s data/processed/heart_disease_clean.csv || {
    echo "    cleaned CSV missing/empty"; exit 1; }
echo "    data/processed/heart_disease_clean.csv  $(wc -l < data/processed/heart_disease_clean.csv) lines"

step "4/6  Run pytest suite"
"$VENV/bin/pytest" unit_test/ -q --no-header

step "5/6  Train model end-to-end"
"$PY" -m src.models.train --no-mlflow
test -s models/heart_pipeline.joblib || {
    echo "    joblib model missing"; exit 1; }
test -d models/mlflow_model || {
    echo "    mlflow model dir missing"; exit 1; }

step "6/6  Summary"
ls -lh models/heart_pipeline.joblib models/mlflow_model/MLmodel 2>/dev/null || true
echo
echo "--- reports/metrics.json (best-model row) ---"
"$PY" -c "
import json, pathlib
m = json.loads(pathlib.Path('reports/metrics.json').read_text())
print('best_model :', m['best_model'])
print('test_metrics:', json.dumps(m['models'][m['best_model']]['test_metrics'], indent=2))
"

echo
echo "✅ clean-setup verification PASSED"
echo "   venv kept at $VENV (re-run with --rm to delete it)"
