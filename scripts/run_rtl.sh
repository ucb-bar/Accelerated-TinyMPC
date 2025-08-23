#!/usr/bin/env bash
# scripts/run_rtl.sh
# Run Chipyard Verilator RTL sims over grouped binaries with minimal rebuilds.
# Groups:
#   scalar  -> RocketConfig
#   vector  -> REFV512D256RocketConfig
#   gemmini -> FPGemminiRocketConfig
#
# Usage:
#   bash scripts/run_rtl.sh
#   DRY_RUN=1 bash scripts/run_rtl.sh     # only print what would run
#   JOBS=16 bash scripts/run_rtl.sh       # control parallelism when building sims

set -euo pipefail

# ---------- config ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
CHIPYARD_DIR="${TOOLS_DIR}/chipyard"
SIM_DIR="${CHIPYARD_DIR}/sims/verilator"
WRAPPER="${TOOLS_DIR}/chipyard_env.sh"

JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}"
DRY_RUN="${DRY_RUN:-0}"

# ---------- helpers ----------
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
blue()  { printf "\033[34m%s\033[0m\n" "$*"; }

die() { red "ERROR: $*"; exit 1; }

run_in_env() {
  # Prefer the wrapper (ensures RISCV/conda env), otherwise run directly.
  if [[ -x "${WRAPPER}" ]]; then
    "${WRAPPER}" "$@"
  else
    "$@"
  fi
}

need_dir() {
  [[ -d "$1" ]] || die "Missing directory: $1"
}

need_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

# ---------- sanity checks ----------
need_dir "${CHIPYARD_DIR}"
need_dir "${SIM_DIR}"
command -v make >/dev/null 2>&1 || die "make not found"

# ---------- collect binaries ----------
# Absolute paths so Make can load them regardless of CWD.
abspath() { python3 - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

BIN_SCALAR=()
BIN_VECTOR=()
BIN_GEMMINI=()

# Scalar (RocketConfig): cpu/eigen + their cycles variants
for rel in \
  "build-cpu/example_quadrotor_tracking_cpu" \
  "build-eigen/example_quadrotor_tracking_eigen" \
  "build-cpu-cycles/example_quadrotor_tracking_cpu_cycles" \
  "build-eigen-cycles/example_quadrotor_tracking_eigen_cycles"
do
  path="${REPO_ROOT}/${rel}"
  [[ -x "${path}" ]] && BIN_SCALAR+=("$(abspath "${path}")")
done

# Vector (REFV512D256RocketConfig): rvv + rvv-handopt (+ cycles)
for rel in \
  "build-rvv/example_quadrotor_tracking_rvv" \
  "build-rvv-handopt/example_quadrotor_tracking_rvv_handopt" \
  "build-rvv-cycles/example_quadrotor_tracking_rvv_cycles" \
  "build-rvv-handopt-cycles/example_quadrotor_tracking_rvv_handopt_cycles"
do
  path="${REPO_ROOT}/${rel}"
  [[ -x "${path}" ]] && BIN_VECTOR+=("$(abspath "${path}")")
done

# Gemmini (FPGemminiRocketConfig): systolic (+ cycles)
for rel in \
  "build-gemmini/example_quadrotor_tracking_gemmini" \
  "build-gemmini-cycles/example_quadrotor_tracking_gemmini_cycles"
do
  path="${REPO_ROOT}/${rel}"
  [[ -x "${path}" ]] && BIN_GEMMINI+=("$(abspath "${path}")")
done

# ---------- runner ----------
declare -A RESULTS

ensure_sim_built() {
  local cfg="$1"
  blue "[Build sim] CONFIG=${cfg}"
  if (( DRY_RUN )); then
    echo "DRY_RUN: make -C '${SIM_DIR}' -j${JOBS} CONFIG=${cfg}"
    return 0
  fi
  run_in_env make -C "${SIM_DIR}" -j"${JOBS}" CONFIG="${cfg}"
}

run_group() {
  local label="$1"; shift
  local cfg="$1"; shift
  local -a bins=("$@")

  if [[ ${#bins[@]} -eq 0 ]]; then
    blue "[Skip] ${label}: no binaries found"
    return 0
  fi

  ensure_sim_built "${cfg}"

  for bin in "${bins[@]}"; do
    local key="${label}::$(basename "${bin}")"
    blue "[Run] ${label} CONFIG=${cfg} BINARY=$(basename "${bin}")"
    if (( DRY_RUN )); then
      echo "DRY_RUN: make -C '${SIM_DIR}' CONFIG=${cfg} BINARY='${bin}' LOADMEM=1 run-binary"
      RESULTS["$key"]="DRY-RUN"
      continue
    fi

    set +e
    run_in_env make -C "${SIM_DIR}" CONFIG="${cfg}" BINARY="${bin}" LOADMEM=1 run-binary
    rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
      RESULTS["$key"]="PASS"
      green "[PASS] ${key}"
    else
      RESULTS["$key"]="FAIL(${rc})"
      red   "[FAIL] ${key} (rc=${rc})"
    fi
  done
}

# ---------- execute ----------
run_group "scalar"  "RocketConfig"              "${BIN_SCALAR[@]}"
run_group "vector"  "REFV512D256RocketConfig"   "${BIN_VECTOR[@]}"
run_group "gemmini" "FPGemminiRocketConfig"     "${BIN_GEMMINI[@]}"

# ---------- summary ----------
echo
blue "=== RTL Simulation Summary ==="
pad() { printf "%-12s" "$1"; }
for k in "${!RESULTS[@]}"; do
  printf "%s  %s\n" "$(pad "${RESULTS[$k]}")" "$k"
done

# Non-zero exit if any FAIL
failed=0
for v in "${RESULTS[@]}"; do
  [[ "$v" == PASS || "$v" == DRY-RUN ]] || failed=1
done
exit $failed
