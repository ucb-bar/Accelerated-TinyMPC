#!/usr/bin/env bash
# scripts/run_rtl.sh
# Run Chipyard Verilator RTL sims over grouped binaries with minimal rebuilds.
# Assumes tools/chipyard/env.sh has been sourced already (no wrapper).
#
# Groups:
#   scalar  -> RocketConfig
#   vector  -> REFV512D256RocketConfig
#   gemmini -> FPGemminiRocketConfig

set -euo pipefail

# --- sbt/socket safety: keep tmp short to avoid AF_UNIX path limit (~108B) ---
# Many environments inject JAVA_TOOL_OPTIONS with a long -Djava.io.tmpdir=…
# Force a short tmpdir and make sbt non-interactive.
export TMPDIR="${TMPDIR:-/tmp}"
if [[ "${TMPDIR}" != "/tmp" && ${#TMPDIR} -gt 20 ]]; then
  TMPDIR="/tmp"
  export TMPDIR
fi
if [[ -n "${JAVA_TOOL_OPTIONS:-}" ]]; then
  # Remove any preexisting -Djava.io.tmpdir=… token(s)
  JAVA_TOOL_OPTIONS="$(sed -E 's@-Djava\.io\.tmpdir=[^ ]+@@g' <<<"${JAVA_TOOL_OPTIONS}")"
fi
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djava.io.tmpdir=/tmp"
export SBT_NON_INTERACTIVE=1
# Clean stale sock dirs if present (best-effort)
rm -rf "${TMPDIR}/.sbt"/sbt-socket* 2>/dev/null || true

# ---------- config ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
CHIPYARD_DIR="${TOOLS_DIR}/chipyard"
SIM_DIR="${CHIPYARD_DIR}/sims/verilator"

JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}"
DRY_RUN="${DRY_RUN:-0}"

# ---------- helpers ----------
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
blue()  { printf "\033[34m%s\033[0m\n" "$*"; }
die() { red "ERROR: $*"; exit 1; }

need_dir() { [[ -d "$1" ]] || die "Missing directory: $1"; }
need_file(){ [[ -f "$1" ]] || die "Missing file: $1"; }

abspath() {
  if command -v realpath >/dev/null 2>&1; then realpath "$1";
  else python3 - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
  fi
}

# ---------- sanity checks ----------
need_dir "${CHIPYARD_DIR}"
need_dir "${SIM_DIR}"
command -v make >/dev/null 2>&1 || die "make not found"
if [[ -z "${RISCV:-}" ]]; then
  red "Warning: RISCV is not set. Did you source tools/chipyard/env.sh?"
fi

# ---------- collect binaries ----------
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
  if [[ "${PROGRESS_SETUP}" -eq 1 ]]; then
    tput cnorm 2>/dev/null || true
    printf "\n"
  fi
}
trap cleanup_progress EXIT

progress_init() { [[ "${PROGRESS_SETUP}" -eq 1 ]] || { tput civis 2>/dev/null || true; PROGRESS_SETUP=1; progress_draw; }; }
progress_draw() {
  local cols="${COLUMNS:-}"; [[ -z "${cols}" ]] && cols="$(tput cols 2>/dev/null || echo 80)"
  local label="Progress:" suffix=" ${COMPLETED_RUNS}/${TOTAL_RUNS}"
  local base=$(( ${#label} + ${#suffix} + 10 ))
  local width=$(( cols > base ? cols - base : 30 )); (( width < 10 )) && width=10
  local pct=0; (( TOTAL_RUNS > 0 )) && pct=$(( COMPLETED_RUNS * 100 / TOTAL_RUNS ))
  local filled=$(( width * pct / 100 )) empty=$(( width - filled ))
  printf "\r\033[K%s [%s%s] %3d%%%s" \
    "${label}" "$(printf "%0.s#" $(seq 1 ${filled}))" "$(printf "%0.s-" $(seq 1 ${empty}))" "${pct}" "${suffix}"
}
progress_tick() { (( COMPLETED_RUNS++ )); progress_draw; }

# ---------- build/run orchestration ----------
declare -A RESULTS

ensure_sim_built() {
  local cfg="$1"
  blue "[Build sim] CONFIG=${cfg}"
  if (( DRY_RUN )); then
    echo "DRY_RUN: make -C '${SIM_DIR}' -j${JOBS} CONFIG=${cfg}"
  else
    make -C "${SIM_DIR}" -j"${JOBS}" CONFIG="${cfg}"
  fi
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
    make -C "${SIM_DIR}" CONFIG="${cfg}" BINARY="${bin}" LOADMEM=1 run-binary
    rc=$?
    set -e

    if [[ $rc -eq 0 ]]; then
      RESULTS["$key"]="PASS"; green "[PASS] ${key}"
    else
      RESULTS["$key"]="FAIL(${rc})"; red "[FAIL] ${key} (rc=${rc})"
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
for k in "${!RESULTS[@]}"; do printf "%s  %s\n" "$(pad "${RESULTS[$k]}")" "$k"; done

failed=0
for v in "${RESULTS[@]}"; do [[ "$v" == PASS || "$v" == DRY-RUN ]] || failed=1; done
exit $failed
