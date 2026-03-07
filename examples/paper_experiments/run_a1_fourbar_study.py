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


def env_for_case(run_name: str, dt: float, end_time: float) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP4_RUN_NAME"] = run_name
    env["STARK_EXP4_DT"] = f"{dt:.12g}"
    env["STARK_EXP4_END_TIME"] = f"{end_time:.12g}"
    return env


def curve_stats(path: Path) -> Dict[str, float]:
    curve = sanitize_curve(pd.read_csv(path), "t", "max_drift")
    late = curve[curve["t"] >= 1.0]["max_drift"]
    return {
        "peak_drift": float(curve["max_drift"].max()),
        "late_drift": float(late.max()) if not late.empty else float(curve["max_drift"].iloc[-1]),
        "final_drift": float(curve["max_drift"].iloc[-1]),
    }


def load_reaction_curve(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["t", "support_f_norm", "support_t_norm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", "support_f_norm", "support_t_norm"])
    df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t").reset_index(drop=True)
    return df


def interpolate_rmse(ref: pd.DataFrame, cur: pd.DataFrame, column: str) -> float:
    t = cur["t"].to_numpy()
    y = cur[column].to_numpy()
    ref_t = ref["t"].to_numpy()
    ref_y = ref[column].to_numpy()
    ref_interp = np.interp(t, ref_t, ref_y)
    return float(np.sqrt(np.mean((y - ref_interp) ** 2)))


def append_reaction_error_metrics(rows: List[Dict[str, object]]):
    df = pd.DataFrame(rows)
    for method in df["method"].unique():
        sub = df[df["method"] == method].sort_values("dt")
        ref_row = sub.iloc[0]
        ref_curve = load_reaction_curve(Path(ref_row["reaction_csv"]))
        for idx in sub.index:
            curve = load_reaction_curve(Path(df.loc[idx, "reaction_csv"]))
            rows[idx]["support_force_rmse_vs_finest"] = interpolate_rmse(ref_curve, curve, "support_f_norm")
            rows[idx]["support_torque_rmse_vs_finest"] = interpolate_rmse(ref_curve, curve, "support_t_norm")


def run_or_collect_case(exe: Path, dt: float, end_time: float, force_run: bool) -> Dict[str, object]:
    run_name = f"exp4_fourbar_a1dt_contact_dt{dt_tag(dt)}"
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None:
        env = env_for_case(run_name, dt, end_time)
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
    reaction_path = case_dir / "fourbar_reaction.csv"
    stats = curve_stats(drift_path)
    row: Dict[str, object] = {
        "method": "STARK IPC",
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
        "reaction_csv": str(reaction_path),
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
        "support_force_rmse_vs_finest",
        "support_torque_rmse_vs_finest",
        "reaction_csv",
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
    sub = df.sort_values("dt")
    axs[0].plot(sub["dt"], sub["final_drift"], marker="o", color="#1f77b4", label="STARK IPC")
    axs[1].plot(sub["dt"], sub["total"], marker="o", color="#1f77b4", label="STARK IPC")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"$\Delta t$ (s)")
    axs[0].set_ylabel("Final drift (m)")
    axs[0].set_title("A1 Four-Bar: Time-Step Verification")
    axs[0].legend()
    axs[1].set_xlabel(r"$\Delta t$ (s)")
    axs[1].set_ylabel("Runtime (s)")
    axs[1].set_title("A1 Four-Bar: Runtime vs Time Step")
    save_fig(fig, out_dir, "a1_fourbar_dt_sweep")


def _smooth(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_force_torque(rows: List[Dict[str, object]], out_dir: Path, reaction_dt: float):
    selected = sorted(rows, key=lambda row: float(row["dt"]))
    if not selected:
        raise RuntimeError("No A1 rows available for force/torque plotting.")

    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)
    for ax in axs:
        setup_axes(ax)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, row in enumerate(selected):
        reaction = pd.read_csv(Path(row["reaction_csv"]))
        reaction["t"] = pd.to_numeric(reaction["t"], errors="coerce")
        force = pd.to_numeric(reaction["support_f_norm"], errors="coerce")
        torque = pd.to_numeric(reaction["support_t_norm"], errors="coerce")
        mask = reaction["t"].notna() & force.notna() & torque.notna()
        t = reaction.loc[mask, "t"]
        label = rf"$\Delta t={float(row['dt']):.4g}$"
        color = palette[idx % len(palette)]
        axs[0].plot(t, _smooth(force[mask]), color=color, label=label)
        axs[1].plot(t, _smooth(torque[mask]), color=color, label=label)

    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Support reaction norm (N)")
    axs[0].set_title("A1 Four-Bar: Recovered Support Reaction")
    axs[0].legend()
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Support torque norm (Nm)")
    axs[1].set_title("A1 Four-Bar: Recovered Support Torque")
    save_fig(fig, out_dir, "a1_fourbar_force_torque")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/collect A1 four-bar verification under the contact-centric paper framing.")
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
        rows.append(run_or_collect_case(exe, dt, args.end_time, args.force_run))

    append_reaction_error_metrics(rows)
    write_csv(rows, args.out_csv.resolve())
    plot_dt_sweep(rows, args.fig_dir.resolve())
    plot_force_torque(rows, args.fig_dir.resolve(), args.reaction_dt)
    print(f"Wrote {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
