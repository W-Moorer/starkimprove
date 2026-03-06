#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from study_utils import FIGS_DIR, OUTPUT_BASE, latest_logger, parse_logger_metrics, resolve_executable, save_fig, setup_axes


def env_for_case(run_name: str, preconditioner: str, dt: float, end_time: float) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP5_RUN_NAME"] = run_name
    env["STARK_EXP5_DT"] = f"{dt:.12g}"
    env["STARK_EXP5_END_TIME"] = f"{end_time:.12g}"
    env["STARK_NEWTON_PRECONDITIONER"] = preconditioner
    return env


def run_or_collect_case(exe: Path, run_name: str, preconditioner: str, dt: float, end_time: float, force_run: bool) -> Dict[str, object]:
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None:
        env = env_for_case(run_name, preconditioner, dt, end_time)
        cmd = [str(exe), "exp5"]
        print(f"[b1] run {run_name} ({preconditioner})")
        ret = subprocess.run(cmd, cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"B1 case failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")

    metrics = parse_logger_metrics(logger)
    return {
        "preconditioner": preconditioner,
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "failed_step_count": metrics.get("failed_step_count"),
        "failed_step_time": metrics.get("failed_step_time"),
        "hardening_count": metrics.get("hardening_count"),
    }


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_ablation(rows: List[Dict[str, object]], fig_dir: Path):
    df = pd.DataFrame(rows)
    x = range(len(df))
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax in axs:
        setup_axes(ax)
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["preconditioner"])

    axs[0].bar(x, pd.to_numeric(df["total"], errors="coerce"), color=["#9ecae1", "#3182bd"])
    axs[0].set_ylabel("Runtime (s)")
    axs[0].set_title("B1: Exp5 Runtime")

    axs[1].bar(x, pd.to_numeric(df["linear_iterations"], errors="coerce"), color=["#fdd0a2", "#e6550d"])
    axs[1].set_ylabel("Linear iterations")
    axs[1].set_title("B1: Linear Solve Cost")
    save_fig(fig, fig_dir, "b1_preconditioner_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/collect B1 preconditioner ablation on Exp5.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--end-time", type=float, default=0.2)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "b1_preconditioner_ablation.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)
    rows = [
        run_or_collect_case(exe, "exp5_b1_diag", "diagonal", args.dt, args.end_time, args.force_run),
        run_or_collect_case(exe, "exp5_b1_blockdiag", "block_diagonal", args.dt, args.end_time, args.force_run),
    ]
    write_csv(rows, args.out_csv.resolve())
    plot_ablation(rows, args.fig_dir.resolve())
    print(f"Wrote {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
