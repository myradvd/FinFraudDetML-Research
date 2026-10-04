#!/usr/bin/env bash
# Execute notebooks in place (outputs are saved into the .ipynb), several at a time.
#   scripts/run_notebooks.sh                       # every experiment notebook + the final notebook
#   scripts/run_notebooks.sh notebooks/07_*.ipynb  # just these
# CB_THREADS=3 pins CatBoost's thread count, which its fitted trees depend on (3 reproduces the uploads).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
JOBS=${JOBS:-4}
export CB_THREADS=${CB_THREADS:-3} MPLBACKEND=Agg PYTHONWARNINGS=ignore
mkdir -p results/logs

if [ $# -eq 0 ]; then set -- notebooks/*.ipynb final_notebook/predict_catboost.ipynb; fi

run_one() {
  nb="$1"; log="results/logs/$(basename "${nb%.ipynb}").log"; t0=$(date +%s)
  if "$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
       --ExecutePreprocessor.kernel_name=python3 "$nb" >"$log" 2>&1; then
    echo "OK    $(( $(date +%s) - t0 ))s  $nb"
  else
    echo "FAIL  $(( $(date +%s) - t0 ))s  $nb   (see $log)"; return 1
  fi
}
export -f run_one; export PY

printf '%s\n' "$@" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}
