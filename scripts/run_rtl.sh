#!/usr/bin/env bash
# scripts/run_rtl.sh
# Run Chipyard Verilator RTL sims over grouped binaries with minimal rebuilds.
# Adds a bottom progress bar that updates as runs complete.
#
# Groups:
#   scalar  -> RocketConfig
#   vector  -> REFV512D256RocketConfig
#   gemmini -> FPGemminiRocketConfig
#
# Usage:
#   bash scripts/run_rtl.sh
#   DRY_RUN=1 bash scripts/run_rtl.sh
#   JOBS=16 bash scripts/run_rtl.sh

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
  if [[ -x "${WRAPPER}" ]]; then
    "${WRAPPER}" "$@"
  else
    "$@"
  fi
}

need_dir() { [[ -d "$1" ]] || die "Missing directory: $1"; }
need_file(){ [[ -f "$1" ]] || die "Missing file: $1"; }

# ---------- sanity checks ----------
need_dir "${CHIPYARD_DIR}"
need_dir "${SIM_DIR}"
command -v make >/dev/null 2>&1 || die "make not found"

# ---------- collect binaries ----------
abspath() { python3 - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
}

BIN_SCALAR=()
BIN_VECTOR=()
BIN_GEMMINI=()

# Scalar (RocketConfig): cpu/eigen + cycles
for rel in \
  "build-cpu/example_quadrotor_tracking_cpu" \
  "build-eigen/example_quadrotor_tracking_eigen" \
  "build-cpu-cycles/example_quadrotor_tracking_cpu_cycles" \
  "build-eigen-cycles/example_quadrotor_tracking_eigen_cycles"
do
  p="${REPO_ROOT}/${rel}"
  [[ -x "${p}" ]] && BIN_SCALAR+=("$(abspath "${p}")")
done

# Vector (REFV512D256RocketConfig): rvv + handopt + cycles
for rel in \
  "build-rvv/example_quadrotor_tracking_rvv" \
  "build-rvv-handopt/example_quadrotor_tracking_rvv_handopt" \
  "build-rvv-cycles/example_quadrotor_tracking_rvv_cycles" \
  "build-rvv-handopt-cycles/example_quadrotor_tracking_rvv_handopt_cycles"
do
  p="${REPO_ROOT}/${rel}"
  [[ -x "${p}" ]] && BIN_VECTOR+=("$(abspath "${p}")")
done

# Gemmini (FPGemminiRocketConfig): systolic + cycles
for rel in \
  "build-gemmini/example_quadrotor_tracking_gemmini" \
  "build-gemmini-cycles/example_quadrotor_tracking_gemmini_cycles"
do
  p="${REPO_ROOT}/${rel}"
  [[ -x "${p}" ]] && BIN_GEMMINI+=("$(abspath "${p}")")
done

# ---------- progress bar ----------
TOTAL_RUNS=$(( ${#BIN_SCALAR[@]} + ${#BIN_VECTOR[@]} + ${#BIN_GEMMINI[@]} ))
COMPLETED_RUNS=0
PROGRESS_SETUP=0

cleanup_progress() {
  # Restore cursor on exit
  if [[ "${PROGRESS_SETUP}" -eq 1 ]]; then
    tput cnorm 2>/dev/null || true
    printf "\n"  # ensure clean line after bar
  fi
}
trap cleanup_progress EXIT

progress_init() {
  [[ "${PROGRESS_SETUP}" -eq 1 ]] && return 0
  tput civis 2>/dev/null || true   # hide cursor
  PROGRESS_SETUP=1
  progress_draw
}

progress_draw() {
  # choose bar width nicely within terminal width
  local cols="${COLUMNS:-}"
  [[ -z "${cols}" ]] && cols="$(tput cols 2>/dev/null || echo 80)"
  local label="Progress:"
  local suffix=" ${COMPLETED_RUNS}/${TOTAL_RUNS}"
  local base=$(( ${#label} + ${#suffix} + 10 ))
  local width=$(( cols > base ? cols - base : 30 ))
  (( width < 10 )) && width=10

  local pct=0
  if (( TOTAL_RUNS > 0 )); then
    pct=$(( COMPLETED_RUNS * 100 / TOTAL_RUNS ))
  fi
  local filled=$(( width * pct / 100 ))
  local empty=$(( width - filled ))

  printf "\r\033[K%s [%s%s] %3d%%%s" \
    "${label}" \
    "$(printf "%0.s#" $(seq 1 ${filled}))" \
    "$(printf "%0.s-" $(seq 1 ${empty}))" \
    "${pct}" \
    "${suffix}"
}

progress_tick() {
  (( COMPLETED_RUNS++ ))
  progress_draw
}

# ---------- build/run orchestration ----------
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
      progress_tick
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
    progress_tick
  done
}

# ---------- execute ----------
progress_init
progress_draw

run_group "scalar"  "RocketConfig"            "${BIN_SCALAR[@]}"
run_group "vector"  "REFV512D256RocketConfig" "${BIN_VECTOR[@]}"
run_group "gemmini" "FPGemminiRocketConfig"   "${BIN_GEMMINI[@]}"

# ---------- summary ----------
printf "\n"
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
