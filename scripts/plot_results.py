#!/usr/bin/env python3
# scripts/plot_results.py
#
# Generate plots + CSVs from RTL or Spike logs.
# - E2E: average of "Time for iter X: NNN" (base logs only, skip *_cycles.log)
# - Kernel: average of "<kernel_name> cycles: NNN"
#   -> Plotted as grouped bars: one group per kernel (in discovery order), one bar per HW/SW combo.
#
# Source selection:
#   env PLOT_SOURCE=rtl|spike (default rtl). If unset, falls back to env RUNNER.
#   CLI override: --source rtl|spike
#
# Inputs:
#   RTL   logs: results/rtl/<CONFIG>/*.log
#   Spike logs: results/spike/<CONFIG>/*.log
#
# Outputs (results/plots/):
#   - e2e_avg_cycles_<src>.png
#   - kernel_avg_cycles_by_kernel_<src>.png
#   - metrics_per_log_<src>.csv
#   - metrics_per_combo_<src>.csv
#   - metrics_per_kernel_combo_<src>.csv

import argparse
import csv
import math
import os
import re
from collections import defaultdict, OrderedDict

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RE_E2E    = re.compile(r"^\s*Time\s+for\s+iter\s+\d+\s*:\s*(\d+)\s*$")
RE_KERNEL = re.compile(r"^\s*([A-Za-z0-9_ ]+?)\s+cycles\s*:\s*(\d+)\s*$")

# --- helpers ---------------------------------------------------------------

def pick_source(env_default="rtl"):
    env_src = os.environ.get("PLOT_SOURCE") or os.environ.get("RUNNER") or env_default
    env_src = env_src.strip().lower()
    return "spike" if env_src == "spike" else "rtl"

# def config_to_hw(cfg: str) -> str:
#     if cfg == "RocketConfig":
#         return "scalar"
#     if cfg == "REFV512D256RocketConfig":
#         return "vector"
#     if cfg == "FPGemminiRocketConfig":
#         return "systolic"
#     return cfg

def bin_to_sw(basename: str) -> str:
    name = basename.lower()
    if "rvv_handopt" in name or "rvv-handopt" in name:
        return "rvv-handopt"
    if "rvv" in name:
        return "rvv"
    if "eigen" in name:
        return "eigen"
    if "gemmini" in name:
        return "gemmini"
    if "cpu" in name:
        return "cpu"
    return "unknown"

def parse_e2e_cycles_from_log(path: str):
    vals = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = RE_E2E.match(line)
            if m:
                vals.append(int(m.group(1)))
    return vals

def parse_kernel_lines_from_log(path: str):
    """Yield (kernel_name, cycles) in the order they appear."""
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = RE_KERNEL.match(line)
            if m:
                yield m.group(1).strip(), int(m.group(2))

def iter_logs(root: str):
    """Yield (config, log_path) for <root>/<CONFIG>/*.log."""
    if not os.path.isdir(root):
        return
    for cfg in sorted(os.listdir(root)):
        cfg_dir = os.path.join(root, cfg)
        if not os.path.isdir(cfg_dir):
            continue
        for fn in sorted(os.listdir(cfg_dir)):
            if fn.endswith(".log"):
                yield cfg, os.path.join(cfg_dir, fn)

def mean_or_none(xs):
    return (sum(xs) / len(xs)) if xs else None

# --- main ------------------------------------------------------------------

def main():
    default_src = pick_source()
    parser = argparse.ArgumentParser(description="Plot results into results/plots")
    parser.add_argument("--source", choices=["rtl", "spike"], default=default_src,
                        help="Log source to parse (default from env: PLOT_SOURCE or RUNNER)")
    parser.add_argument("--rtl_dir", default=os.path.join(RESULTS_DIR, "rtl"),
                        help="RTL results dir (default: results/rtl)")
    parser.add_argument("--spike_dir", default=os.path.join(RESULTS_DIR, "spike"),
                        help="Spike results dir (default: results/spike)")
    parser.add_argument("--out_dir", default=PLOTS_DIR,
                        help="Output plots dir (default: results/plots)")
    args = parser.parse_args()

    src = args.source
    base_dir = args.rtl_dir if src == "rtl" else args.spike_dir
    suffix = f"_{src}"

    os.makedirs(args.out_dir, exist_ok=True)

    # Per-log rows
    per_log_rows = []

    # Aggregations
    # E2E average per combo
    e2e_combo_values = defaultdict(list)

    # Kernel averages per kernel per combo
    # kernel_combo_values[kernel][combo] -> list of cycles
    kernel_combo_values = defaultdict(lambda: defaultdict(list))

    # Discovery order for kernels (first time any kernel name is seen across logs)
    kernel_order = []

    # Deterministic ordering for combos
    # hw_order = ["scalar", "vector", "systolic"]
    sw_order = ["cpu", "eigen", "rvv", "rvv-handopt", "gemmini", "unknown"]

    logs_found = False

    for cfg, log_path in iter_logs(base_dir):
        logs_found = True
        hw = config_to_hw(cfg)
        base = os.path.basename(log_path)
        sw = bin_to_sw(base.replace(".log", ""))
        combo = f"{hw}/{sw}"

        # E2E: only base logs (skip *_cycles.log)
        use_for_e2e = "_cycles" not in base
        e2e_vals = parse_e2e_cycles_from_log(log_path) if use_for_e2e else []

        # Kernel: collect all matches, also record discovery order
        kernel_vals = []
        for kname, cyc in parse_kernel_lines_from_log(log_path):
            kernel_vals.append(cyc)
            if kname not in kernel_order:
                kernel_order.append(kname)
            kernel_combo_values[kname][combo].append(cyc)

        e2e_avg = mean_or_none(e2e_vals)
        kernel_avg = mean_or_none(kernel_vals)

        per_log_rows.append({
            "source": src,
            "config": cfg,
            "hardware": hw,
            "software": sw,
            "combo": combo,
            "binary_log": base,
            "path": os.path.relpath(log_path, REPO_ROOT),
            "e2e_avg_cycles": f"{e2e_avg:.2f}" if e2e_avg is not None else "",
            "e2e_samples": len(e2e_vals),
            "kernel_avg_cycles": f"{kernel_avg:.2f}" if kernel_avg is not None else "",
            "kernel_samples": len(kernel_vals),
        })

        if e2e_avg is not None:
            e2e_combo_values[combo].append(e2e_avg)

    if not logs_found:
        print(f"No logs found under: {base_dir}")
        return

    def ordered_combos(values_dict):
    combos = list(values_dict.keys())
    def key_func(c):
        hw, sw = c.split("/", 1) if "/" in c else (c, "")
        return (hw,  # lexicographic by CONFIG name
                sw_order.index(sw) if sw in sw_order else len(sw_order),
                c)
    return [c for c in sorted(combos, key=key_func)]

    # E2E aggregated (per combo)
    e2e_combo_avgs = OrderedDict(
        (combo, sum(e2e_combo_values[combo]) / len(e2e_combo_values[combo]))
        for combo in ordered_combos(e2e_combo_values)
    )

    # Kernel aggregated: avg per (kernel, combo)
    # Also collect full set of combos that appear anywhere (so legend is consistent)
    all_combos_for_kernels = set()
    for kname, d in kernel_combo_values.items():
        for combo in d.keys():
            all_combos_for_kernels.add(combo)
    combo_list_for_kernels = ordered_combos({c: 1 for c in all_combos_for_kernels})  # reuse ordering

    kernel_combo_avgs = OrderedDict()  # kernel -> OrderedDict(combo -> avg or nan)
    for kname in kernel_order:
        d = kernel_combo_values.get(kname, {})
        row = OrderedDict()
        for combo in combo_list_for_kernels:
            xs = d.get(combo, [])
            row[combo] = (sum(xs)/len(xs)) if xs else math.nan
        kernel_combo_avgs[kname] = row

    # --- CSVs ----------------------------------------------------------------
    per_log_csv = os.path.join(args.out_dir, f"metrics_per_log{suffix}.csv")
    with open(per_log_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source","config","hardware","software","combo","binary_log","path",
            "e2e_avg_cycles","e2e_samples","kernel_avg_cycles","kernel_samples"
        ])
        writer.writeheader()
        for row in per_log_rows:
            writer.writerow(row)

    per_combo_csv = os.path.join(args.out_dir, f"metrics_per_combo{suffix}.csv")
    with open(per_combo_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source","combo","metric","avg_cycles"])
        for combo, v in e2e_combo_avgs.items():
            writer.writerow([src, combo, "e2e_avg_cycles", f"{v:.2f}"])

    per_kernel_combo_csv = os.path.join(args.out_dir, f"metrics_per_kernel_combo{suffix}.csv")
    with open(per_kernel_combo_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source","kernel","combo","avg_cycles"])
        for kname, combo_row in kernel_combo_avgs.items():
            for combo, v in combo_row.items():
                writer.writerow([src, kname, combo, "" if math.isnan(v) else f"{v:.2f}"])

    # --- Plots ---------------------------------------------------------------

    def log_safe(x, eps=1e-9):
        # Make values safe for log scale; keep NaNs as-is so matplotlib skips them
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return x
        return x if x > 0 else eps

    def plot_bar(filename: str, title: str, data: OrderedDict):
        if not data:
            print(f"[Skip] No data for {title} ({src})")
            return
        labels = list(data.keys())
        vals = [log_safe(v) for v in data.values()]

        plt.figure()
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=30, ha="right")
        plt.ylabel("cycles (log scale)")
        plt.title(f"{title} ({src})")
        plt.yscale("log")
        ax = plt.gca()
        ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        out = os.path.join(args.out_dir, filename)
        plt.savefig(out)
        plt.close()
        print(f"Wrote {out}")

    def plot_grouped_by_kernel(filename: str, title: str,
                            kernel_to_combo_avgs: OrderedDict, combos: list[str]):
        if not kernel_to_combo_avgs:
            print(f"[Skip] No data for {title} ({src})")
            return

        kernels = list(kernel_to_combo_avgs.keys())
        num_groups = len(kernels)
        num_series = len(combos)
        if num_groups == 0 or num_series == 0:
            print(f"[Skip] No data for {title} ({src})")
            return

        # Make it wider based on number of kernel groups
        # (baseline 12in + 0.6in per kernel group, capped to something reasonable)
        width_in = max(12.0, min(24.0, 12.0 + 0.6 * num_groups))
        height_in = 6.0
        plt.figure(figsize=(width_in, height_in))

        x = list(range(num_groups))
        bar_width = 0.8 / max(1, num_series)

        # One bar series per HW/SW combo
        for s_idx, combo in enumerate(combos):
            offsets = [xi + (s_idx - (num_series - 1) / 2.0) * bar_width for xi in x]
            heights_raw = [kernel_to_combo_avgs[k].get(combo, math.nan) for k in kernels]
            heights = [log_safe(v) for v in heights_raw]
            plt.bar(offsets, heights, width=bar_width, label=combo)

        plt.xticks(x, kernels, rotation=30, ha="right")
        plt.ylabel("cycles (log scale)")
        plt.title(f"{title} ({src})")
        plt.yscale("log")

        ax = plt.gca()
        ax.yaxis.grid(True, which="both", linestyle="--", alpha=0.5)

        # Legend to the RIGHT of the plot
        # bbox_to_anchor pushes it outside; bbox_inches='tight' (on save) prevents clipping
        plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize="small")

        # Tight layout for plot area (legend is outside, so use bbox_inches='tight' on save)
        plt.tight_layout()
        out = os.path.join(args.out_dir, filename)
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Wrote {out}")

    # E2E: same bar as before
    plot_bar(f"e2e_avg_cycles{suffix}.png", "Average E2E Cycles per HW/SW Combo", e2e_combo_avgs)

    # Kernel: grouped by kernel name (discovery order), one bar per combo
    plot_grouped_by_kernel(
        f"kernel_avg_cycles_by_kernel{suffix}.png",
        "Average Kernel Cycles (grouped by kernel)",
        kernel_combo_avgs,
        combo_list_for_kernels
    )

    print(f"Wrote {per_log_csv}")
    print(f"Wrote {per_combo_csv}")
    print(f"Wrote {per_kernel_combo_csv}")

if __name__ == "__main__":
    main()
