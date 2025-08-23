#!/usr/bin/env bash
# scripts/install_conda.sh
# Install Miniforge3 (Linux/x86_64) into ./tools/miniforge3 non-interactively,
# then source conda and activate base. Re-runnable and idempotent.

set -euo pipefail

# Resolve repo root and target dirs (script may be called from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
INSTALL_DIR="${TOOLS_DIR}/miniforge3"
INSTALLER="${TOOLS_DIR}/Miniforge3-Linux-x86_64.sh"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"

# Check prerequisites
if ! command -v wget >/dev/null 2>&1; then
  echo "Error: wget is required but not found. Please install wget and re-run." >&2
  return 1 2>/dev/null || exit 1
fi

# Create tools dir
mkdir -p "${TOOLS_DIR}"

# Download installer if needed
if [[ ! -f "${INSTALLER}" ]]; then
  echo "Downloading Miniforge installer..."
  wget -q --show-progress -O "${INSTALLER}" "${MINIFORGE_URL}"
  chmod +x "${INSTALLER}"
else
  echo "Installer already exists at ${INSTALLER} (skipping download)."
fi

# Install (idempotent)
if [[ -d "${INSTALL_DIR}" ]]; then
  echo "Miniforge appears installed at ${INSTALL_DIR} (skipping install)."
else
  echo "Installing Miniforge to ${INSTALL_DIR} ..."
  bash "${INSTALLER}" -b -p "${INSTALL_DIR}"
fi

# Optional: cleanup installer (uncomment if you don't want to keep it)
# rm -f "${INSTALLER}"

# Source conda and activate base (only affects the current shell if this script is sourced)
CONDA_SH="${INSTALL_DIR}/etc/profile.d/conda.sh"
if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate base
  echo "Conda initialized. Active env: $(conda info --json | tr -d '\n' | sed -n 's/.*"active_prefix_name":"\([^"]*\)".*/\1/p')"
else
  # Fallback PATH update if conda.sh is missing (shouldn't happen)
  export PATH="${INSTALL_DIR}/bin:${PATH}"
  echo "Warning: ${CONDA_SH} not found. Added ${INSTALL_DIR}/bin to PATH."
fi

# Friendly reminder if script wasn’t sourced
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo
  echo "NOTE: You executed the script. The environment activation only applied to this process."
  echo "To use the conda env in your current shell, run:  source scripts/install.sh"
fi
