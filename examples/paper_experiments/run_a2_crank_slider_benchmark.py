#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_utils import FIGS_DIR, OUTPUT_BASE, latest_logger, parse_logger_metrics, resolve_executable, sanitize_curve, save_fig, setup_axes


def stark_env(run_name: str, dt: float, end_time: float, use_al: bool) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP2_RUN_NAME"] = run_name
    env["STARK_EXP2_DT"] = f"{dt:.12g}"
    env["STARK_EXP2_END_TIME"] = f"{end_time:.12g}"
    if use_al:
        env["STARK_JOINT_AL_ENABLED"] = "1"
    else:
        env.pop("STARK_JOINT_AL_ENABLED", None)
    return env


def run_stark_case(exe: Path, dt: float, end_time: float, use_al: bool, force_run: bool) -> Dict[str, object]:
    method = "al" if use_al else "soft"
    run_name = f"exp2_crank_slider_a2_{method}"
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None:
        cmd = [str(exe), "exp2_slider"]
        env = stark_env(run_name, dt, end_time, use_al)
        print(f"[a2] run STARK {run_name}")
        ret = subprocess.run(cmd, cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"STARK A2 failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")
    metrics = parse_logger_metrics(logger)
    state = sanitize_curve(pd.read_csv(case_dir / "crank_slider_state.csv"), "t", "slider_cx")
    state_full = pd.read_csv(case_dir / "crank_slider_state.csv")
    gap = pd.to_numeric(state_full["gap_stop"], errors="coerce")
    return {
        "framework": "STARK",
        "method": "AL-IPC" if use_al else "soft",
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "min_gap": float(gap.min()),
        "state_csv": str(case_dir / "crank_slider_state.csv"),
    }


def run_pychrono_cases(dt: float, end_time: float, force_run: bool):
    summary_path = OUTPUT_BASE / "pychrono_exp2_crank_slider_summary.csv"
    mode_map = {
        "nsc_ncp": OUTPUT_BASE / "pychrono_exp2_crank_slider_nsc_ncp" / "summary.csv",
        "smc_penalty": OUTPUT_BASE / "pychrono_exp2_crank_slider_smc_penalty" / "summary.csv",
    }
    if force_run or not summary_path.exists() or any(not p.exists() for p in mode_map.values()):
        cmd = [
            "conda",
            "run",
            "-n",
            "chrono-baseline",
            "python",
            str((Path(__file__).resolve().parent / "pychrono_crank_slider_benchmark.py")),
            "--mode",
            "all",
            "--dt",
            f"{dt:.12g}",
            "--end-time",
            f"{end_time:.12g}",
            "--output-base",
            str(OUTPUT_BASE),
        ]
        print("[a2] run PyChrono crank-slider")
        ret = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
        if ret.returncode != 0:
            raise RuntimeError("PyChrono A2 benchmark failed.")
    rows: List[Dict[str, object]] = []
    for mode, summary_csv in mode_map.items():
        row = pd.read_csv(summary_csv).iloc[0].to_dict()
        rows.append(
            {
                "framework": "PyChrono",
                "method": "NSC/APGD" if mode == "nsc_ncp" else "SMC",
                "run_name": f"pychrono_exp2_crank_slider_{mode}",
                "logger_file": "",
                "total": float(row["wall_time_s"]),
                "newton_iterations": None,
                "linear_iterations": float(row["avg_solver_iterations"]),
                "min_gap": float(row["min_gap"]),
                "state_csv": str(summary_csv.parent / "crank_slider_state.csv"),
            }
        )
    return rows


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: List[Dict[str, object]], fig_dir: Path):
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True)
    for ax in axs:
        setup_axes(ax)
    palette = {
        "soft": "#e377c2",
        "AL-IPC": "#1f77b4",
        "NSC/APGD": "#2ca02c",
        "SMC": "#d62728",
    }
    for row in rows:
        state = pd.read_csv(Path(row["state_csv"]))
        x_curve = sanitize_curve(state, "t", "slider_cx")
        v_curve = sanitize_curve(state, "t", "slider_vx")
        axs[0].plot(x_curve["t"], x_curve["slider_cx"], color=palette[row["method"]], label=row["method"])
        axs[1].plot(v_curve["t"], v_curve["slider_vx"], color=palette[row["method"]], label=row["method"])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Slider x (m)")
    axs[0].set_title("A2: Slider Displacement")
    axs[0].legend()
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Slider vx (m/s)")
    axs[1].set_title("A2: Slider Velocity")
    save_fig(fig, fig_dir, "a2_crank_slider_compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/collect A2 crank-slider benchmark.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--end-time", type=float, default=0.4)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "a2_crank_slider_summary.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)
    rows = [
        run_stark_case(exe, args.dt, args.end_time, False, args.force_run),
        run_stark_case(exe, args.dt, args.end_time, True, args.force_run),
    ]
    rows.extend(run_pychrono_cases(args.dt, args.end_time, args.force_run))
    write_csv(rows, args.out_csv.resolve())
    plot_curves(rows, args.fig_dir.resolve())
    print(f"Wrote {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
