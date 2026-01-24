#!/usr/bin/env bash
# Simple launcher for the enhanced Streamlit app.
# Usage: ./scripts/run_streamlit.sh [--port 8501] [--mock]
set -euo pipefail

# Ensure we run from the repo root regardless of caller CWD
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PORT=8501
MOCK=0

# Parse args robustly
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      shift
      PORT="${1:-8501}"
      shift || true
      ;;
    --mock)
      MOCK=1
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Load .env if present so users don't have to export manually.
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

# Best-effort cleanup for Streamlit component manifest glitches.
if [[ -f "$HOME/.streamlit/components.json" ]]; then
  rm -f "$HOME/.streamlit/components.json" || true
fi
if [ "${MOCK}" = "1" ]; then
  export ALLOW_FAKE_GROQ=1
fi
if [ -z "${GROQ_API_KEY:-}" ] && [ "${MOCK}" = "0" ]; then
  echo "ERROR: GROQ_API_KEY not set. Export it or use --mock." >&2
  exit 1
fi
APP_FILE="${STREAMLIT_APP:-simple_streamlit_app.py}"
exec /opt/anaconda3/.venv/bin/streamlit run "$APP_FILE" --server.port "$PORT"

