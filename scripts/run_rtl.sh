#!/usr/bin/env bash
# scripts/run_rtl.sh
# Run Chipyard Verilator RTL sims and/or Spike runs grouped by CONFIG, skipping finished runs.
# Assumes tools/chipyard/env.sh has been sourced already (no wrapper).
#
# Groups:
#   scalar  -> RocketConfig (and any others you call run_group with)
#   vector  -> REFV512D256RocketConfig (etc.)
#   gemmini -> FPGemminiRocketConfig (etc.)
#
# Usage examples:
#   bash scripts/run_rtl.sh                              # RTL only (default)
#   RUNNER=spike bash scripts/run_rtl.sh                 # Spike only
#   RUNNER=both bash scripts/run_rtl.sh                  # RTL then Spike
#   DRY_RUN=1 bash scripts/run_rtl.sh
#   JOBS=16 bash scripts/run_rtl.sh
#   CLEAN=1 bash scripts/run_rtl.sh      # force re-run even if logs finished (or exist for Spike)

set -euo pipefail

# --- sbt/socket safety: keep tmp short to avoid AF_UNIX path limit (~108B) ---
export TMPDIR="${TMPDIR:-/tmp}"
if [[ "${TMPDIR}" != "/tmp" && ${#TMPDIR} -gt 20 ]]; then
  TMPDIR="/tmp"; export TMPDIR
fi
if [[ -n "${JAVA_TOOL_OPTIONS:-}" ]]; then
  JAVA_TOOL_OPTIONS="$(sed -E 's@-Djava\.io\.tmpdir=[^ ]+@@g' <<<"${JAVA_TOOL_OPTIONS}")"
fi
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:-} -Djava.io.tmpdir=/tmp"
export SBT_NON_INTERACTIVE=1
rm -rf "${TMPDIR}/.sbt"/sbt-socket* 2>/dev/null || true

# ---------- config ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOLS_DIR="${REPO_ROOT}/tools"
CHIPYARD_DIR="${TOOLS_DIR}/chipyard"
SIM_DIR="${CHIPYARD_DIR}/sims/verilator"

JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}"
DRY_RUN="${DRY_RUN:-0}"
TIMEOUT_CYCLES="${TIMEOUT_CYCLES:-100000000}"
CLEAN="${CLEAN:-}"
RUNNER="${RUNNER:-rtl}"   # rtl | spike | both

RESULTS_DIR_RTL="${REPO_ROOT}/results/rtl"
RESULTS_DIR_SPIKE="${REPO_ROOT}/results/spike"
mkdir -p "${RESULTS_DIR_RTL}" "${RESULTS_DIR_SPIKE}"

# Spike tool + flags (override via env if needed)
SPIKE="${SPIKE:-spike}"
SPIKE_ISA_VECTOR="${SPIKE_ISA_VECTOR:---isa=rv64gcv_zicntr_zvl512b}"
SPIKE_EXT_GEMMINI="${SPIKE_EXT_GEMMINI:---extension=gemmini}"

# ---------- helpers ----------
red()   { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
blue()  { printf "\033[34m%s\033[0m\n" "$*"; }
die() { red "ERROR: $*"; exit 1; }

need_dir() { [[ -d "$1" ]] || die "Missing directory: $1"; }

abspath() {
  if command -v realpath >/dev/null 2>&1; then realpath "$1";
  else python3 - <<'PY' "$1"
import os,sys
print(os.path.abspath(sys.argv[1]))
PY
  fi
}

repeat_char() {
  local n="${1:-0}" ch="${2:-=}" buf=""
  (( n > 0 )) && printf -v buf '%*s' "$n" '' && buf="${buf// /$ch}"
  printf '%s' "$buf"
}
divider() {
  local cols="${COLUMNS:-}"; [[ -z "${cols}" ]] && cols="$(tput cols 2>/dev/null || echo 80)"
  repeat_char "$cols" "="; echo
}

log_path_for() {
  # $1 = CONFIG, $2 = /abs/path/to/binary
  local cfg="$1" bin="$2"
  echo "${SIM_DIR}/output/chipyard.harness.TestHarness.${cfg}/$(basename "${bin}").log"
}

log_is_finished() {
  # returns 0 if last non-empty line contains 'Verilog $finish'
  local log="$1"
  [[ -f "${log}" ]] || return 1
  local last
  last="$(awk 'NF{line=$0} END{print line}' "${log}")"
  [[ "${last}" == *"Verilog \$finish"* ]]
}

finish_banner() {
  local label="$1" cfg="$2" bin="$3" status="$4" rc="$5" secs="$6" done="$7" total="$8"
  divider
  printf "[%d/%d] %s :: %s\n" "${done}" "${total}" "${label}" "$(basename "${bin}")"
  printf "CONFIG=%s  TIMEOUT_CYCLES=%s  RESULT=%s%s\n" \
    "${cfg}" "${TIMEOUT_CYCLES}" "${status}" \
    "$( [[ -n "${rc}" ]] && printf " (rc=%s)" "${rc}" || printf "" )"
  [[ -n "${secs}" ]] && printf "Duration: %ss\n" "${secs}"
  divider
}

copy_to_results_rtl() {
  local cfg="$1" bin="$2"
  local src_log; src_log="$(log_path_for "${cfg}" "${bin}")"
  local dst_dir="${RESULTS_DIR_RTL}/${cfg}"
  local dst_log="${dst_dir}/$(basename "${bin}").log"
  mkdir -p "${dst_dir}"
  if [[ -f "${src_log}" ]]; then
    cp -f "${src_log}" "${dst_log}"
  else
    printf "WARN: expected RTL log not found to copy: %s\n" "${src_log}" >&2
  fi
}

spike_log_path_for() {
  # $1 = CONFIG label dir, $2 = /abs/path/to/binary
  local cfg="$1" bin="$2"
  echo "${RESULTS_DIR_SPIKE}/${cfg}/$(basename "${bin}").log"
}

# ---------- sanity checks ----------
if [[ "${RUNNER}" == "rtl" || "${RUNNER}" == "both" ]]; then
  need_dir "${CHIPYARD_DIR}"
  need_dir "${SIM_DIR}"
  command -v make >/dev/null 2>&1 || die "make not found"
  if [[ -z "${RISCV:-}" ]]; then
    red "Warning: RISCV is not set. Did you source tools/chipyard/env.sh?"
  fi
fi
if [[ "${RUNNER}" == "spike" || "${RUNNER}" == "both" ]]; then
  command -v "${SPIKE}" >/dev/null 2>&1 || die "spike not found in PATH"
fi

# ---------- collect binaries ----------
BIN_SCALAR=()
BIN_VECTOR=()
BIN_GEMMINI=()

# Scalar: cpu/eigen + cycles
for rel in \
  "build-cpu/example_quadrotor_tracking_cpu" \
  "build-eigen/example_quadrotor_tracking_eigen" \
  "build-cpu-cycles/example_quadrotor_tracking_cpu_cycles" \
  "build-eigen-cycles/example_quadrotor_tracking_eigen_cycles"
do
  p="${REPO_ROOT}/${rel}"
  [[ -x "${p}" ]] && BIN_SCALAR+=("$(abspath "${p}")")
done

# Vector: rvv + handopt + cycles
for rel in \
  "build-rvv/example_quadrotor_tracking_rvv" \
  "build-rvv-handopt/example_quadrotor_tracking_rvv_handopt" \
  "build-rvv-cycles/example_quadrotor_tracking_rvv_cycles" \
  "build-rvv-handopt-cycles/example_quadrotor_tracking_rvv_handopt_cycles"
do
  p="${REPO_ROOT}/${rel}"
  [[ -x "${p}" ]] && BIN_VECTOR+=("$(abspath "${p}")")
done

# Gemmini: systolic + cycles
for rel in \
  "build-gemmini/example_quadrotor_tracking_gemmini" \
  "build-gemmini-cycles/example_quadrotor_tracking_gemmini_cycles"
do
  p="${REPO_ROOT}/${rel}"
  [[ -x "${p}" ]] && BIN_GEMMINI+=("$(abspath "${p}")")
done

TOTAL_RUNS=$(( ${#BIN_SCALAR[@]} + ${#BIN_VECTOR[@]} + ${#BIN_GEMMINI[@]} ))
COMPLETED_RUNS=0
(( TOTAL_RUNS > 0 )) || { blue "No binaries found to run."; exit 0; }

blue "Planned runs (per CONFIG call): ${TOTAL_RUNS}"
divider

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

run_group_rtl() {
  local label="$1"; shift
  local cfg="$1"; shift
  local -a bins=("$@")

  if [[ ${#bins[@]} -eq 0 ]]; then
    blue "[Skip] ${label}: no binaries found"
    return 0
  fi

  # Filter bins needing execution (unless CLEAN set)
  local -a to_run=()
  for bin in "${bins[@]}"; do
    local key="rtl:${label}/${cfg}::$(basename "${bin}")"
    local log; log="$(log_path_for "${cfg}" "${bin}")"
    if [[ -z "${CLEAN}" ]] && log_is_finished "${log}"; then
      RESULTS["$key"]="SKIP"
      ((++COMPLETED_RUNS))
      finish_banner "rtl/${label}" "${cfg}" "${bin}" "SKIP (finished log found)" "" "" "${COMPLETED_RUNS}" "${TOTAL_RUNS}"
      copy_to_results_rtl "${cfg}" "${bin}"
    else
      to_run+=("${bin}")
    fi
  done

  # If nothing to run for this config, don't build the sim
  if [[ ${#to_run[@]} -eq 0 ]]; then
    blue "[Skip build] ${label}/${cfg}: all binaries already finished"
    return 0
  fi

  ensure_sim_built "${cfg}"

  # Run the remaining binaries
  for bin in "${to_run[@]}"; do
    local key="rtl:${label}/${cfg}::$(basename "${bin}")"
    blue "[RTL Run] ${label} CONFIG=${cfg} BINARY=$(basename "${bin}")"
    local t0 t1 secs rc=0

    if (( DRY_RUN )); then
      echo "DRY_RUN: make -C '${SIM_DIR}' CONFIG=${cfg} BINARY='${bin}' LOADMEM=1 TIMEOUT_CYCLES=${TIMEOUT_CYCLES} run-binary"
      RESULTS["$key"]="DRY-RUN"
      ((++COMPLETED_RUNS))
      finish_banner "rtl/${label}" "${cfg}" "${bin}" "DRY-RUN" "" "0" "${COMPLETED_RUNS}" "${TOTAL_RUNS}"
      continue
    fi

    t0=$(date +%s)
    set +e
    make -C "${SIM_DIR}" CONFIG="${cfg}" BINARY="${bin}" LOADMEM=1 TIMEOUT_CYCLES="${TIMEOUT_CYCLES}" run-binary
    rc=$?
    set -e
    t1=$(date +%s); secs=$(( t1 - t0 ))

    if [[ $rc -eq 0 ]]; then
      RESULTS["$key"]="PASS"; green "[PASS] ${key}"
      status="PASS"
    else
      RESULTS["$key"]="FAIL(${rc})"; red "[FAIL] ${key} (rc=${rc})"
      status="FAIL"
    fi
    copy_to_results_rtl "${cfg}" "${bin}"
    ((++COMPLETED_RUNS))
    finish_banner "rtl/${label}" "${cfg}" "${bin}" "${status}" "${rc}" "${secs}" "${COMPLETED_RUNS}" "${TOTAL_RUNS}"
  done
}

run_group_spike() {
  local label="$1"; shift
  local cfg="$1"; shift          # we’ll use cfg as the folder label under results/spike/
  local -a bins=("$@")

  if [[ ${#bins[@]} -eq 0 ]]; then
    blue "[Skip] spike/${label}: no binaries found"
    return 0
  fi

  local -a spike_base_args=()
  case "${label}" in
    scalar)  spike_base_args=() ;;
    vector)  spike_base_args=(${SPIKE_ISA_VECTOR}) ;;
    gemmini) spike_base_args=(${SPIKE_EXT_GEMMINI}) ;;
    *)       spike_base_args=() ;;
  esac

  for bin in "${bins[@]}"; do
    local key="spike:${label}/${cfg}::$(basename "${bin}")"
    local out_dir="${RESULTS_DIR_SPIKE}/${cfg}"
    local out_log; out_log="$(spike_log_path_for "${cfg}" "${bin}")"
    mkdir -p "${out_dir}"

    if (( DRY_RUN )); then
      echo "DRY_RUN: ${SPIKE} ${spike_base_args[*]} '${bin}' | tee '${out_log}'"
      RESULTS["$key"]="DRY-RUN"
      ((++COMPLETED_RUNS))
      finish_banner "spike/${label}" "${cfg}" "${bin}" "DRY-RUN" "" "0" "${COMPLETED_RUNS}" "${TOTAL_RUNS}"
      continue
    fi

    if [[ -z "${CLEAN}" ]] && [[ -f "${out_log}" ]]; then
      RESULTS["$key"]="SKIP"
      ((++COMPLETED_RUNS))
      finish_banner "spike/${label}" "${cfg}" "${bin}" "SKIP (log exists)" "" "" "${COMPLETED_RUNS}" "${TOTAL_RUNS}"
      continue
    fi

    blue "[Spike Run] ${label} CFG_LABEL=${cfg} BINARY=$(basename "${bin}")"
    local t0 t1 secs rc=0
    t0=$(date +%s)
    set +e
    # run spike and tee stdout to the results log
    "${SPIKE}" "${spike_base_args[@]}" "${bin}" | tee "${out_log}"
    rc=${PIPESTATUS[0]}
    set -e
    t1=$(date +%s); secs=$(( t1 - t0 ))

    if [[ $rc -eq 0 ]]; then
      RESULTS["$key"]="PASS"; green "[PASS] ${key}"
      status="PASS"
    else
      RESULTS["$key"]="FAIL(${rc})"; red "[FAIL] ${key} (rc=${rc})"
      status="FAIL"
    fi
    ((++COMPLETED_RUNS))
    finish_banner "spike/${label}" "${cfg}" "${bin}" "${status}" "${rc}" "${secs}" "${COMPLETED_RUNS}" "${TOTAL_RUNS}"
  done
}

# ---------- execute ----------
# RTL groups
if [[ "${RUNNER}" == "rtl" || "${RUNNER}" == "both" ]]; then
  run_group_rtl "scalar"  "RocketConfig"               "${BIN_SCALAR[@]}"
  run_group_rtl "scalar"  "LargeBoomV3Config"          "${BIN_SCALAR[@]}"
  run_group_rtl "vector"  "REFV512D256RocketConfig"    "${BIN_VECTOR[@]}"
  run_group_rtl "vector"  "REFV512D256ShuttleConfig"   "${BIN_VECTOR[@]}"
  run_group_rtl "gemmini" "FPGemminiRocketConfig"      "${BIN_GEMMINI[@]}"
fi

# Spike groups (mirrors the same CONFIG labels for directory structure/plotting)
if [[ "${RUNNER}" == "spike" || "${RUNNER}" == "both" ]]; then
  run_group_spike "scalar"  "RocketConfig"               "${BIN_SCALAR[@]}"
  run_group_spike "scalar"  "LargeBoomV3Config"          "${BIN_SCALAR[@]}"
  run_group_spike "vector"  "REFV512D256RocketConfig"    "${BIN_VECTOR[@]}"
  run_group_spike "vector"  "REFV512D256ShuttleConfig"   "${BIN_VECTOR[@]}"
  run_group_spike "gemmini" "FPGemminiRocketConfig"      "${BIN_GEMMINI[@]}"
fi

# ---------- summary ----------
blue "=== Simulation Summary ==="
pad() { printf "%-12s" "$1"; }
for k in "${!RESULTS[@]}"; do
  printf "%s  %s\n" "$(pad "${RESULTS[$k]}")" "$k"
done

failed=0
for v in "${RESULTS[@]}"; do
  [[ "$v" == PASS || "$v" == DRY-RUN || "$v" == SKIP ]] || failed=1
done
exit $failed
