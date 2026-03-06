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

from study_utils import FIGS_DIR, OUTPUT_BASE, latest_logger, parse_logger_metrics, resolve_executable, sanitize_curve, save_fig, setup_axes


def parse_list(values: str) -> List[float]:
    out: List[float] = []
    for token in values.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise ValueError("Empty dt list.")
    return out


def dt_tag(dt: float) -> str:
    return f"{dt:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def env_for_case(run_name: str, dt: float, end_time: float, use_al: bool) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP4_RUN_NAME"] = run_name
    env["STARK_EXP4_DT"] = f"{dt:.12g}"
    env["STARK_EXP4_END_TIME"] = f"{end_time:.12g}"
    if use_al:
        env["STARK_JOINT_AL_ENABLED"] = "1"
    else:
        env.pop("STARK_JOINT_AL_ENABLED", None)
    return env


def curve_stats(path: Path) -> Dict[str, float]:
    curve = sanitize_curve(pd.read_csv(path), "t", "max_drift")
    late = curve[curve["t"] >= 1.0]["max_drift"]
    return {
        "peak_drift": float(curve["max_drift"].max()),
        "late_drift": float(late.max()) if not late.empty else float(curve["max_drift"].iloc[-1]),
        "final_drift": float(curve["max_drift"].iloc[-1]),
    }


def run_or_collect_case(exe: Path, dt: float, end_time: float, use_al: bool, force_run: bool) -> Dict[str, object]:
    method = "al" if use_al else "soft"
    run_name = f"exp4_fourbar_a1dt_{method}_dt{dt_tag(dt)}"
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None:
        env = env_for_case(run_name, dt, end_time, use_al)
        cmd = [str(exe), "exp4_fourbar"]
        print(f"[a1] run {run_name}")
        ret = subprocess.run(cmd, cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"A1 case failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")

    metrics = parse_logger_metrics(logger)
    drift_path = case_dir / "joint_drift.csv"
    proxy_path = case_dir / "fourbar_constraint_proxy.csv"
    stats = curve_stats(drift_path)
    row: Dict[str, object] = {
        "method": "AL-IPC" if use_al else "soft",
        "dt": dt,
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "joint_error_max_l2": metrics.get("joint_error_max_l2"),
        "joint_error_max_deg": metrics.get("joint_error_max_deg"),
        "peak_drift": stats["peak_drift"],
        "late_drift": stats["late_drift"],
        "final_drift": stats["final_drift"],
        "proxy_csv": str(proxy_path),
    }
    return row


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "dt",
        "run_name",
        "logger_file",
        "total",
        "newton_iterations",
        "linear_iterations",
        "joint_error_max_l2",
        "joint_error_max_deg",
        "peak_drift",
        "late_drift",
        "final_drift",
        "proxy_csv",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_dt_sweep(rows: List[Dict[str, object]], out_dir: Path):
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax in axs:
        setup_axes(ax)
        ax.set_xscale("log")
    for method, color in [("soft", "#e377c2"), ("AL-IPC", "#1f77b4")]:
        sub = df[df["method"] == method].sort_values("dt")
        axs[0].plot(sub["dt"], sub["final_drift"], marker="o", color=color, label=method)
        axs[1].plot(sub["dt"], sub["total"], marker="o", color=color, label=method)
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\Delta t$ (s)")
    axs[0].set_ylabel("Final drift (m)")
    axs[0].set_title("A1 Four-Bar: Time-Step Sensitivity")
    axs[0].legend()
    axs[1].set_xlabel(r"$\Delta t$ (s)")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].set_title("A1 Four-Bar: Runtime vs Time Step")
    save_fig(fig, out_dir, "a1_fourbar_dt_sweep")


def _smooth(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_force_torque(rows: List[Dict[str, object]], out_dir: Path, reaction_dt: float):
    selected = [r for r in rows if abs(float(r["dt"]) - reaction_dt) < 1e-12]
    if len(selected) != 2:
        raise RuntimeError(f"Expected 2 cases at dt={reaction_dt}, found {len(selected)}.")

    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)
    for ax in axs:
        setup_axes(ax)

    palette = {"soft": "#e377c2", "AL-IPC": "#1f77b4"}
    for row in selected:
        proxy = pd.read_csv(Path(row["proxy_csv"]))
        proxy["t"] = pd.to_numeric(proxy["t"], errors="coerce")
        force = pd.to_numeric(proxy["support_point_force_proxy"], errors="coerce")
        torque = pd.to_numeric(proxy["support_direction_torque_proxy"], errors="coerce")
        mask = proxy["t"].notna() & force.notna() & torque.notna()
        t = proxy.loc[mask, "t"]
        axs[0].plot(t, _smooth(force[mask]), color=palette[row["method"]], label=row["method"])
        axs[1].plot(t, _smooth(torque[mask]), color=palette[row["method"]], label=row["method"])

    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Support force proxy")
    axs[0].set_title("A1 Four-Bar: Support Reaction Proxy")
    axs[0].legend()
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Support torque proxy")
    axs[1].set_title("A1 Four-Bar: Support Torque Proxy")
    save_fig(fig, out_dir, "a1_fourbar_force_torque")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/collect A1 four-bar dt study.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dts", type=str, default="0.02,0.01,0.005")
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--reaction-dt", type=float, default=0.01)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "a1_fourbar_dt_sweep.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)
    dts = parse_list(args.dts)

    rows: List[Dict[str, object]] = []
    for dt in dts:
        rows.append(run_or_collect_case(exe, dt, args.end_time, False, args.force_run))
        rows.append(run_or_collect_case(exe, dt, args.end_time, True, args.force_run))

    write_csv(rows, args.out_csv.resolve())
    plot_dt_sweep(rows, args.fig_dir.resolve())
    plot_force_torque(rows, args.fig_dir.resolve(), args.reaction_dt)
    print(f"Wrote {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
