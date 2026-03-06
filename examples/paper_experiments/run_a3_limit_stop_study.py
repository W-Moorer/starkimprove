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

from study_utils import FIGS_DIR, OUTPUT_BASE, latest_logger, parse_logger_metrics, resolve_executable, save_fig, setup_axes


def stark_env(run_name: str, dt: float, end_time: float, limit_deg: float, use_al: bool) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP3_RUN_NAME"] = run_name
    env["STARK_EXP3_DT"] = f"{dt:.12g}"
    env["STARK_EXP3_END_TIME"] = f"{end_time:.12g}"
    env["STARK_EXP3_LIMIT_DEG"] = f"{limit_deg:.12g}"
    if use_al:
        env["STARK_JOINT_AL_ENABLED"] = "1"
    else:
        env.pop("STARK_JOINT_AL_ENABLED", None)
    return env


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_cols = [
        "t",
        "theta_deg",
        "omega_deg_s",
        "tip_x",
        "tip_y",
        "limit_violation_deg",
        "limit_torque_proxy",
        "support_point_force_proxy",
        "support_direction_torque_proxy",
        "support_reaction_torque_z",
        "support_reaction_force_norm",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", "theta_deg", "omega_deg_s"])
    df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t").reset_index(drop=True)
    return df


def first_trigger_time(df: pd.DataFrame, threshold: float) -> float | None:
    active = df[df["limit_violation_deg"] > threshold]
    if active.empty:
        return None
    return float(active["t"].iloc[0])


def run_case(exe: Path, dt: float, end_time: float, limit_deg: float, use_al: bool, force_run: bool) -> Dict[str, object]:
    method = "al" if use_al else "soft"
    pretty = "AL-IPC" if use_al else "soft"
    run_name = f"exp3_limit_stop_a3_{method}"
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None:
        env = stark_env(run_name, dt, end_time, limit_deg, use_al)
        print(f"[a3] run {run_name}")
        ret = subprocess.run([str(exe), "exp3"], cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"A3 case failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")

    metrics = parse_logger_metrics(logger)
    state_csv = case_dir / "limit_stop_state.csv"
    state = load_state(state_csv)

    overshoot = np.maximum(np.abs(state["theta_deg"].to_numpy()) - limit_deg, 0.0)
    trigger_time = first_trigger_time(state, threshold=1e-6)
    row: Dict[str, object] = {
        "framework": "STARK",
        "method": pretty,
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "trigger_time_s": trigger_time,
        "peak_abs_theta_deg": float(np.abs(state["theta_deg"]).max()),
        "peak_limit_overshoot_deg": float(np.max(overshoot)),
        "peak_limit_violation_deg": float(state["limit_violation_deg"].max()),
        "peak_limit_torque_proxy": float(state["limit_torque_proxy"].max()),
        "peak_support_force_proxy": float(state["support_point_force_proxy"].max()),
        "peak_support_reaction_torque_z": None,
        "peak_support_reaction_force_norm": None,
        "final_theta_deg": float(state["theta_deg"].iloc[-1]),
        "final_omega_deg_s": float(state["omega_deg_s"].iloc[-1]),
        "state_csv": str(state_csv),
    }
    return row


def run_pychrono_case(dt: float, end_time: float, limit_deg: float, mode: str, force_run: bool) -> Dict[str, object]:
    if mode not in {"nsc_ncp", "smc_penalty"}:
        raise ValueError(f"Unsupported PyChrono mode: {mode}")

    script_path = Path(__file__).resolve().parent / "pychrono_limit_stop_benchmark.py"
    summary_csv = OUTPUT_BASE / f"pychrono_exp3_limit_stop_{mode}" / "summary.csv"
    state_csv = OUTPUT_BASE / f"pychrono_exp3_limit_stop_{mode}" / "limit_stop_state.csv"
    if force_run or not summary_csv.exists() or not state_csv.exists():
        cmd = (
            "conda activate chrono-baseline; "
            f"python '{script_path}' "
            f"--mode {mode} "
            f"--dt {dt:.12g} "
            f"--end-time {end_time:.12g} "
            f"--limit-deg {limit_deg:.12g} "
            f"--output-base '{OUTPUT_BASE}'"
        )
        print(f"[a3] run PyChrono {mode}")
        ret = subprocess.run(
            ["powershell", "-NoLogo", "-Command", cmd],
            cwd=Path(__file__).resolve().parents[2],
        )
        if ret.returncode != 0:
            raise RuntimeError(f"PyChrono A3 benchmark failed: {mode}")

    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    return {
        "framework": "PyChrono",
        "method": "NSC/APGD" if mode == "nsc_ncp" else "SMC",
        "run_name": f"pychrono_exp3_limit_stop_{mode}",
        "logger_file": "",
        "total": float(row["wall_time_s"]),
        "newton_iterations": None,
        "linear_iterations": float(row["avg_solver_iterations"]),
        "trigger_time_s": None if pd.isna(row["trigger_time_s"]) or row["trigger_time_s"] == "" else float(row["trigger_time_s"]),
        "peak_abs_theta_deg": float(row["peak_abs_theta_deg"]),
        "peak_limit_overshoot_deg": float(row["peak_limit_overshoot_deg"]),
        "peak_limit_violation_deg": float(row["peak_limit_violation_deg"]),
        "peak_limit_torque_proxy": None,
        "peak_support_force_proxy": None,
        "peak_support_reaction_torque_z": float(row["peak_support_reaction_torque_z"]),
        "peak_support_reaction_force_norm": float(row["peak_support_reaction_force_norm"]),
        "final_theta_deg": float(row["final_theta_deg"]),
        "final_omega_deg_s": float(row["final_omega_deg_s"]),
        "state_csv": str(state_csv),
    }


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "framework",
        "method",
        "run_name",
        "logger_file",
        "total",
        "newton_iterations",
        "linear_iterations",
        "trigger_time_s",
        "peak_abs_theta_deg",
        "peak_limit_overshoot_deg",
        "peak_limit_violation_deg",
        "peak_limit_torque_proxy",
        "peak_support_force_proxy",
        "peak_support_reaction_torque_z",
        "peak_support_reaction_force_norm",
        "final_theta_deg",
        "final_omega_deg_s",
        "state_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _smooth(series: pd.Series, window: int = 9) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_curves(rows: List[Dict[str, object]], fig_dir: Path, limit_deg: float):
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.8), sharex=True)
    for ax in axs:
        setup_axes(ax)

    palette = {"soft": "#e377c2", "AL-IPC": "#1f77b4", "NSC/APGD": "#2ca02c", "SMC": "#d62728"}
    for row in rows:
        state = load_state(Path(row["state_csv"]))
        axs[0].plot(state["t"], _smooth(state["theta_deg"]), color=palette[row["method"]], label=row["method"])
        violation = _smooth(state["limit_violation_deg"])
        axs[1].plot(state["t"], violation, color=palette[row["method"]], label=row["method"])

    axs[0].axhline(limit_deg, color="0.35", linestyle="--", linewidth=1.0)
    axs[0].axhline(-limit_deg, color="0.35", linestyle="--", linewidth=1.0)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel(r"$\theta$ (deg)")
    axs[0].set_title("A3: Rod Angle and Stop Activation")
    axs[0].legend()

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Limit violation (deg)")
    axs[1].set_title("A3: Limit Violation After Switch")

    save_fig(fig, fig_dir, "a3_limit_stop_compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/collect A3 limit-stop hinge study.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--limit-deg", type=float, default=35.0)
    parser.add_argument("--pychrono-mode", choices=["none", "nsc_ncp", "all"], default="nsc_ncp")
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "a3_limit_stop_summary.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)
    rows = [
        run_case(exe, args.dt, args.end_time, args.limit_deg, False, args.force_run),
        run_case(exe, args.dt, args.end_time, args.limit_deg, True, args.force_run),
    ]
    if args.pychrono_mode != "none":
        rows.append(run_pychrono_case(args.dt, args.end_time, args.limit_deg, "nsc_ncp", args.force_run))
        if args.pychrono_mode == "all":
            rows.append(run_pychrono_case(args.dt, args.end_time, args.limit_deg, "smc_penalty", args.force_run))
    write_csv(rows, args.out_csv.resolve())
    plot_curves(rows, args.fig_dir.resolve(), args.limit_deg)
    print(f"Wrote {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
