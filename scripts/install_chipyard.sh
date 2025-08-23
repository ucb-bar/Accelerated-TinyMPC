#!/usr/bin/env bash
# scripts/install_chipyard.sh
# Install Chipyard into ./tools/chipyard and source env.sh.
# Skips build-setup steps 6–9 (FireSim + FireMarshal related).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
CHIPYARD_DIR="${TOOLS_DIR}/chipyard"

# Ensure Miniforge was installed (from install.sh)
if [[ -z "${CONDA_EXE:-}" ]]; then
  echo "Error: Conda not found in PATH."
  echo "Please run: source scripts/install.sh"
  return 1 2>/dev/null || exit 1
fi

# Activate base env (some systems may not have it auto-active)
eval "$(conda shell.bash hook)"
conda activate base

# Install libmamba solver if not already present
if ! conda list -n base | grep -q "conda-libmamba-solver"; then
  echo "Installing conda-libmamba-solver..."
  conda install -y -n base conda-libmamba-solver
  conda config --set solver libmamba
fi

# Clone Chipyard if missing
if [[ ! -d "${CHIPYARD_DIR}" ]]; then
  echo "Cloning Chipyard into ${CHIPYARD_DIR}..."
  git clone https://github.com/ucb-bar/chipyard.git "${CHIPYARD_DIR}"
  cd "${CHIPYARD_DIR}"
  git checkout 1.13.0
else
  echo "Chipyard already exists at ${CHIPYARD_DIR}"
  cd "${CHIPYARD_DIR}"
fi

# Run build-setup, skipping steps 6–9
echo "Running Chipyard build-setup (skipping steps 6–9)..."
./build-setup.sh riscv-tools -s 6 -s 7 -s 8 -s 9

# Source Chipyard env.sh
if [[ -f "${CHIPYARD_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CHIPYARD_DIR}/env.sh"
  echo "Chipyard environment initialized. RISCV toolchain at: $RISCV"
else
  echo "Error: env.sh not found in ${CHIPYARD_DIR}"
  return 1 2>/dev/null || exit 1
fi

# Reminder about sourcing
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo
  echo "NOTE: You executed the script. The Chipyard environment is active only inside this process."
  echo "To have it in your current shell, run:  source scripts/install_chipyard.sh"
fi
