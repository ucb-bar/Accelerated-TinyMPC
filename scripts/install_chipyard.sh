#!/usr/bin/env bash
# scripts/install_chipyard.sh
# Install Chipyard into ./tools/chipyard and create a wrapper for env.sh.
# Skips build-setup steps 6–9 (FireSim + FireMarshal related).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
CHIPYARD_DIR="${TOOLS_DIR}/chipyard"
WRAPPER="${TOOLS_DIR}/chipyard_env.sh"

# Ensure Miniforge was installed
if ! command -v conda >/dev/null 2>&1; then
  echo "Error: Conda not found in PATH."
  echo "Please run: source scripts/install.sh"
  exit 1
fi

# Activate base env
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

# Verify env.sh exists
if [[ ! -f "${CHIPYARD_DIR}/env.sh" ]]; then
  echo "Error: env.sh not found in ${CHIPYARD_DIR}"
  exit 1
fi

# Create a wrapper script for running commands with Chipyard env
cat > "${WRAPPER}" <<EOF
#!/usr/bin/env bash
# Wrapper to run commands with Chipyard environment variables set
# Usage: ./tools/chipyard_env.sh <command> [args...]

set -euo pipefail
source "${CHIPYARD_DIR}/env.sh"

if [[ \$# -eq 0 ]]; then
  exec "\$SHELL"
else
  exec "\$@"
fi
EOF
chmod +x "${WRAPPER}"

echo
echo "Chipyard installation complete."
echo "You can now run commands inside the Chipyard environment via:"
echo "  ${WRAPPER} make verilog"
echo "Or open a new shell with Chipyard env by running:"
echo "  ${WRAPPER}"
