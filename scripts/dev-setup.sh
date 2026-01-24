#!/usr/bin/env bash
# Dev environment setup helper
# Usage: bash scripts/dev-setup.sh
# Creates a conda env named rag-py311 (if not exists), installs PyMuPDF from conda-forge,
# then pip-installs the requirements excluding PyMuPDF to avoid native build.

set -euo pipefail
ENV_NAME="rag-py311"
PYTHON_VER="3.11"
REQ_FILE="requirements.txt"

echo "Checking for conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Anaconda or Miniconda and retry." >&2
  exit 1
fi

# Create conda env if it doesn't exist
if conda info --envs | awk '{print $1}' | grep -q "^${ENV_NAME}$"; then
  echo "Conda env ${ENV_NAME} already exists. Skipping creation." 
else
  echo "Creating conda env ${ENV_NAME} with Python ${PYTHON_VER}..."
  conda create -n "${ENV_NAME}" python=${PYTHON_VER} -y
fi

echo "Activating conda env ${ENV_NAME}..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Installing PyMuPDF from conda-forge (binary wheel) to avoid local compile..."
if ! conda install -c conda-forge pymupdf -y; then
  echo "conda install pymupdf failed; will attempt to install PyMuPDF via pip as a fallback."
  echo "Note: pip will try to download a prebuilt wheel for your platform; if none exists it may attempt to build from source."
  python -m pip install --upgrade pip
  python -m pip install pymupdf || {
    echo "Failed to install PyMuPDF via pip. You may need to install system dependencies or install PyMuPDF via an alternate channel." >&2
    echo "Try: conda search -c conda-forge pymupdf or visit https://anaconda.org to find a matching package for your platform." >&2
  }
fi
echo "Installing heavy native packages from conda-forge (binary wheels) to avoid local C builds..."
# prefer prebuilt native packages: spacy, thinc, blis, pymupdf, numpy
conda config --env --add channels conda-forge || true
conda config --env --set channel_priority strict || true

HEAVY_PKGS=(spacy thinc blis pymupdf numpy)
echo "Attempting to install: ${HEAVY_PKGS[*]}"
if ! conda install -y -c conda-forge "${HEAVY_PKGS[@]}"; then
  echo "conda install of heavy packages failed; attempting best-effort fallback per-package."
  for pkg in "${HEAVY_PKGS[@]}"; do
    echo "Trying conda install -c conda-forge ${pkg} ..."
    if ! conda install -y -c conda-forge "${pkg}"; then
      echo "conda could not install ${pkg}. Will try pip fallback for ${pkg}." >&2
      python -m pip install --upgrade pip
      # pip may still try to build from source for some platforms; report but continue
      if ! python -m pip install "${pkg}"; then
        echo "Warning: pip install ${pkg} failed. ${pkg} may not be available as a prebuilt wheel for your platform." >&2
      fi
    fi
  done
fi

# Clean pip cache to reduce chance of stale builds
echo "Clearing pip cache..."
python -m pip cache purge || true

# Prepare temp requirements file without PyMuPDF
TMP_REQ="/tmp/requirements-no-pymupdf.txt"
if [ ! -f "${REQ_FILE}" ]; then
  echo "${REQ_FILE} not found in repo root. Please run this script from the project root." >&2
  exit 1
fi

# Filter out lines referencing PyMuPDF (case-insensitive)
# Filter out lines referencing heavy native packages (case-insensitive) to avoid rebuilding them via pip
grep -i -v '^\s*#' "${REQ_FILE}" \
  | grep -i -v -E '^(spacy|pymupdf|blis|thinc|numpy)($|[<=>])' > "${TMP_REQ}"

echo "Installing pip requirements (excluding PyMuPDF) - this may take a while..."
python -m pip install --upgrade pip
python -m pip install -r "${TMP_REQ}"

echo "Development environment setup complete. To use it, run:\n  conda activate ${ENV_NAME}" 

# Print quick checks
echo "Installed Python:"; python --version
echo "Installed pip:"; python -m pip --version
echo "PyMuPDF version:"; python -c "import fitz; print('PyMuPDF', fitz.__doc__.split()[0])" || true

echo "You can run scripts/env_audit.py to collect environment diagnostics:" 
echo "  python scripts/env_audit.py > ~/rereg_logs/env_audit.json"

exit 0
