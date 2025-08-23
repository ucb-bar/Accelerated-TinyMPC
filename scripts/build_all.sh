#!/usr/bin/env bash
# scripts/build_all.sh
# Robust multi-config CMake build script with checks and clear logging.

set -euo pipefail

# ---------- helpers ----------
red()   { printf "\033[31m%s\033[0m\n" "$*" ; }
green() { printf "\033[32m%s\033[0m\n" "$*" ; }
blue()  { printf "\033[34m%s\033[0m\n" "$*" ; }

die() { red "ERROR: $*"; exit 1; }

trap 'die "Build failed at line $LINENO."' ERR

# ---------- prerequisites ----------
command -v cmake >/dev/null 2>&1 || die "cmake not found"
JOBS="${JOBS:-$(command -v nproc >/dev/null 2>&1 && nproc || echo 8)}"
BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

# Script can be invoked from anywhere; resolve project root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Prefer Ninja if present (faster, clearer output), else default generator.
GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

# ---------- build function ----------
# build_one <build_dir> <cmake_args> <produced_binary_relpath> [rename_to]
build_one() {
  local bdir="$1"; shift
  local cargs="$1"; shift
  local produced="$1"; shift
  local rename_to="${1:-}"

  local abs_bdir="${REPO_ROOT}/${bdir}"
  mkdir -p "${abs_bdir}"

  blue "[Config] ${bdir}"
  cmake -S "${REPO_ROOT}" -B "${abs_bdir}" "${GENERATOR_ARGS[@]}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" ${cargs}
  cmake --build "${abs_bdir}" --parallel "${JOBS}"

  local produced_path="${abs_bdir}/${produced}"
  [[ -x "${produced_path}" ]] || die "Expected binary not found or not executable: ${produced_path}"

  if [[ -n "${rename_to}" && "${rename_to}" != "$(basename "${produced_path}")" ]]; then
    local target_path="${abs_bdir}/${rename_to}"
    mv -f "${produced_path}" "${target_path}"
    green "Built → ${bdir}/${rename_to}"
  else
    green "Built → ${bdir}/$(basename "${produced_path}")"
  fi
}

# ---------- builds ----------
# Base (no cycle measurement)
build_one "build-cpu" \
  "-DUSE_RVV=OFF -DUSE_CPU=ON -DUSE_TYPE=float32" \
  "example_quadrotor_tracking_cpu"

build_one "build-rvv" \
  "-DUSE_RVV=ON -DUSE_TYPE=float32" \
  "example_quadrotor_tracking_rvv"

build_one "build-rvv-handopt" \
  "-DUSE_RVV=ON -DUSE_TYPE=float32 -DUSE_HANDOPT=ON" \
  "example_quadrotor_tracking_rvv" \
  "example_quadrotor_tracking_rvv_handopt"

build_one "build-eigen" \
  "-DUSE_RVV=OFF -DUSE_EIGEN=ON -DUSE_CPU=ON -DUSE_TYPE=float32" \
  "example_quadrotor_tracking_eigen"

build_one "build-gemmini" \
  "-DUSE_RVV=OFF -DUSE_CPU=ON -DUSE_GEMMINI=ON -DUSE_HANDOPT=ON -DUSE_MATVEC=OFF" \
  "example_quadrotor_tracking_cpu" \
  "example_quadrotor_tracking_gemmini"

# Cycles variants (-DMEASURE_CYCLES=ON)
build_one "build-cpu-cycles" \
  "-DUSE_RVV=OFF -DUSE_CPU=ON -DUSE_TYPE=float32 -DMEASURE_CYCLES=ON" \
  "example_quadrotor_tracking_cpu" \
  "example_quadrotor_tracking_cpu_cycles"

build_one "build-rvv-cycles" \
  "-DUSE_RVV=ON -DUSE_TYPE=float32 -DMEASURE_CYCLES=ON" \
  "example_quadrotor_tracking_rvv" \
  "example_quadrotor_tracking_rvv_cycles"

build_one "build-rvv-handopt-cycles" \
  "-DUSE_RVV=ON -DUSE_TYPE=float32 -DUSE_HANDOPT=ON -DMEASURE_CYCLES=ON" \
  "example_quadrotor_tracking_rvv" \
  "example_quadrotor_tracking_rvv_handopt_cycles"

build_one "build-eigen-cycles" \
  "-DUSE_RVV=OFF -DUSE_EIGEN=ON -DUSE_CPU=ON -DUSE_TYPE=float32 -DMEASURE_CYCLES=ON" \
  "example_quadrotor_tracking_eigen" \
  "example_quadrotor_tracking_eigen_cycles"

build_one "build-gemmini-cycles" \
  "-DUSE_RVV=OFF -DUSE_CPU=ON -DUSE_GEMMINI=ON -DUSE_HANDOPT=ON -DUSE_MATVEC=OFF -DMEASURE_CYCLES=ON" \
  "example_quadrotor_tracking_cpu" \
  "example_quadrotor_tracking_gemmini_cycles"

green "All builds completed successfully."

# ---------- tips ----------
# Customize parallelism:  JOBS=16 bash scripts/build_all.sh
# Customize build type:   CMAKE_BUILD_TYPE=RelWithDebInfo bash scripts/build_all.sh
