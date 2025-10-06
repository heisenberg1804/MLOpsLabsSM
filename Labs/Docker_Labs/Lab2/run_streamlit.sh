#!/usr/bin/env bash
# Helper to run the Streamlit dashboard for Lab2 from the mlopslabs conda env.
# Usage: from repo root or anywhere, run:
#   bash Labs/Docker_Labs/Lab2/run_streamlit.sh

set -euo pipefail

# Try to activate conda env named 'mlopslabs'. If conda is not available in non-interactive shells,
# user may need to run 'conda activate mlopslabs' manually before invoking this script.
if command -v conda >/dev/null 2>&1; then
  # Initialize conda for this shell (best-effort)
  eval "$(conda shell.bash hook 2>/dev/null || true)"
fi

echo "Activating conda env 'mlopslabs'..."
conda activate mlopslabs || {
  echo "Failed to activate 'mlopslabs' environment. Please activate it manually and re-run the script.";
  exit 1;
}

echo "Installing runtime deps (if missing)..."
pip install --upgrade pip >/dev/null 2>&1 || true
pip install streamlit requests scikit-learn pillow >/dev/null 2>&1 || true

echo "Launching Streamlit dashboard (Dashboard_wine.py) on port 8501..."
cd "$(dirname "$0")/src"
streamlit run Dashboard_wine.py --server.port 8501 --server.address 0.0.0.0
