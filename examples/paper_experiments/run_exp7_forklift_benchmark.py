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


def stark_env(
    run_name: str,
    dt: float,
    end_time: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    pallet_y: float,
    pallet_z: float,
    ground_pallet_friction: float,
    fork_pallet_friction: float,
    use_al: bool,
) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP7_RUN_NAME"] = run_name
    env["STARK_EXP7_DT"] = f"{dt:.12g}"
    env["STARK_EXP7_END_TIME"] = f"{end_time:.12g}"
    env["STARK_EXP7_LIFT_START"] = f"{lift_start:.12g}"
    env["STARK_EXP7_LIFT_END"] = f"{lift_end:.12g}"
    env["STARK_EXP7_LIFT_SPEED"] = f"{lift_speed:.12g}"
    env["STARK_EXP7_PALLET_Y"] = f"{pallet_y:.12g}"
    env["STARK_EXP7_PALLET_Z"] = f"{pallet_z:.12g}"
    env["STARK_EXP7_GROUND_PALLET_FRICTION"] = f"{ground_pallet_friction:.12g}"
    env["STARK_EXP7_FORK_PALLET_FRICTION"] = f"{fork_pallet_friction:.12g}"
    env["STARK_EXP7_ADAPTIVE_DT"] = "1"
    if use_al:
        env["STARK_JOINT_AL_ENABLED"] = "1"
    else:
        env.pop("STARK_JOINT_AL_ENABLED", None)
    return env


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "t", "target_lift_v",
        "fork_cy", "fork_vy",
        "pallet_cy", "pallet_vy", "pallet_cz", "pallet_vz",
        "vertical_gap", "actuator_force_proxy",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", "fork_cy", "pallet_cy", "pallet_vy", "pallet_cz"])
    df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t").reset_index(drop=True)
    return df


def classify_state(df: pd.DataFrame, end_time: float, dt: float) -> str:
    if df.empty:
        return "invalid"
    if not np.isfinite(df[["pallet_cy", "pallet_vy", "pallet_cz", "vertical_gap"]].to_numpy()).all():
        return "invalid"
    if float(np.abs(df["pallet_cy"]).max()) > 10.0:
        return "unstable"
    if float(np.abs(df["pallet_vy"]).max()) > 100.0:
        return "unstable"
    if float(np.abs(df["vertical_gap"]).max()) > 10.0:
        return "unstable"
    if float(df["t"].iloc[-1]) < end_time - max(1e-9, 0.5 * dt):
        return "incomplete"
    return "ok"


def error_against_reference(ref: pd.DataFrame, cur: pd.DataFrame) -> Dict[str, float]:
    if cur.empty:
        return {
            "rmse_pallet_y_to_ref": float("inf"),
            "rmse_pallet_vy_to_ref": float("inf"),
            "rmse_pallet_z_to_ref": float("inf"),
            "rmse_gap_to_ref": float("inf"),
            "composite_error": float("inf"),
        }
    t = cur["t"].to_numpy()
    pallet_y = cur["pallet_cy"].to_numpy()
    pallet_vy = cur["pallet_vy"].to_numpy()
    pallet_z = cur["pallet_cz"].to_numpy()
    gap = cur["vertical_gap"].to_numpy()

    ref_y = np.interp(t, ref["t"].to_numpy(), ref["pallet_cy"].to_numpy())
    ref_vy = np.interp(t, ref["t"].to_numpy(), ref["pallet_vy"].to_numpy())
    ref_z = np.interp(t, ref["t"].to_numpy(), ref["pallet_cz"].to_numpy())
    ref_gap = np.interp(t, ref["t"].to_numpy(), ref["vertical_gap"].to_numpy())

    rmse_y = float(np.sqrt(np.mean((pallet_y - ref_y) ** 2)))
    rmse_vy = float(np.sqrt(np.mean((pallet_vy - ref_vy) ** 2)))
    rmse_z = float(np.sqrt(np.mean((pallet_z - ref_z) ** 2)))
    rmse_gap = float(np.sqrt(np.mean((gap - ref_gap) ** 2)))

    y_scale = max(1e-3, float(ref["pallet_cy"].max() - ref["pallet_cy"].min()))
    vy_scale = max(1e-3, float(np.abs(ref["pallet_vy"]).max()))
    z_scale = max(1e-3, float(ref["pallet_cz"].max() - ref["pallet_cz"].min()))
    gap_scale = max(1e-3, float(np.abs(ref["vertical_gap"]).max()))
    composite = rmse_y / y_scale + rmse_vy / vy_scale + rmse_z / z_scale + rmse_gap / gap_scale
    return {
        "rmse_pallet_y_to_ref": rmse_y,
        "rmse_pallet_vy_to_ref": rmse_vy,
        "rmse_pallet_z_to_ref": rmse_z,
        "rmse_gap_to_ref": rmse_gap,
        "composite_error": composite,
    }


def run_stark_case(
    exe: Path,
    run_name: str,
    dt: float,
    end_time: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    pallet_y: float,
    pallet_z: float,
    ground_pallet_friction: float,
    fork_pallet_friction: float,
    use_al: bool,
    force_run: bool,
) -> Dict[str, object]:
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None or not (case_dir / "forklift_state.csv").exists():
        cmd = [str(exe), "exp7"]
        env = stark_env(
            run_name,
            dt,
            end_time,
            lift_start,
            lift_end,
            lift_speed,
            pallet_y,
            pallet_z,
            ground_pallet_friction,
            fork_pallet_friction,
            use_al,
        )
        print(f"[exp7] run STARK {run_name}")
        ret = subprocess.run(cmd, cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"STARK exp7 failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")
    state_csv = case_dir / "forklift_state.csv"
    state = load_state(state_csv)
    metrics = parse_logger_metrics(logger)
    status = classify_state(state, end_time, dt)
    return {
        "framework": "STARK",
        "method": "AL-IPC" if use_al else "soft",
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "failed_step_time": metrics.get("failed_steps"),
        "state_csv": str(state_csv),
        "final_pallet_y": float(state["pallet_cy"].iloc[-1]) if not state.empty else None,
        "final_pallet_z": float(state["pallet_cz"].iloc[-1]) if not state.empty else None,
        "status": status,
    }


def run_pychrono_case(
    mode: str,
    dt: float,
    end_time: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    pallet_y: float,
    pallet_z: float,
    friction: float,
    nsc_compliance: float,
    solver_max_iters: int,
    solver_tol: float,
    tag_extra: str,
    force_run: bool,
) -> Dict[str, object]:
    tag = f"dt{dt:.0e}_ls{lift_start:.2f}_le{lift_end:.2f}".replace("+", "")
    if tag_extra:
        tag = f"{tag}_{tag_extra}"
    case_dir = OUTPUT_BASE / f"pychrono_exp7_forklift_{mode}_{tag}"
    summary_csv = case_dir / "summary.csv"
    state_csv = case_dir / "forklift_state.csv"
    if force_run or not summary_csv.exists() or not state_csv.exists():
        script_path = Path(__file__).resolve().parent / "pychrono_forklift_benchmark.py"
        cmd = (
            "conda activate chrono-baseline; "
            f"python '{script_path}' "
            f"--mode {mode} "
            f"--dt {dt:.12g} "
            f"--end-time {end_time:.12g} "
            f"--lift-start {lift_start:.12g} "
            f"--lift-end {lift_end:.12g} "
            f"--lift-speed {lift_speed:.12g} "
            f"--pallet-y {pallet_y:.12g} "
            f"--pallet-z {pallet_z:.12g} "
            f"--friction {friction:.12g} "
            f"--nsc-compliance {nsc_compliance:.12g} "
            f"--solver-max-iters {solver_max_iters} "
            f"--solver-tol {solver_tol:.12g} "
            f"--output-base '{OUTPUT_BASE}' "
            f"--tag '{tag}'"
        )
        print(f"[exp7] run PyChrono {mode}")
        ret = subprocess.run(["powershell", "-NoLogo", "-Command", cmd], cwd=Path(__file__).resolve().parents[2])
        if ret.returncode != 0:
            raise RuntimeError(f"PyChrono exp7 failed: {mode}")
    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    state = load_state(state_csv)
    status = classify_state(state, end_time, dt)
    return {
        "framework": "PyChrono",
        "method": "NSC/PSOR" if mode == "nsc_psor" else "SMC",
        "run_name": case_dir.name,
        "logger_file": "",
        "total": float(row["wall_time_s"]),
        "newton_iterations": None,
        "linear_iterations": float(row["avg_solver_iterations"]),
        "failed_step_time": None,
        "state_csv": str(state_csv),
        "final_pallet_y": float(state["pallet_cy"].iloc[-1]) if not state.empty else None,
        "final_pallet_z": float(state["pallet_cz"].iloc[-1]) if not state.empty else None,
        "friction": friction,
        "nsc_compliance": nsc_compliance,
        "solver_max_iters": solver_max_iters,
        "solver_tol": solver_tol,
        "status": status,
    }


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
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
        "NSC/PSOR": "#2ca02c",
        "SMC": "#d62728",
    }
    for row in rows:
        if row.get("status") != "ok":
            continue
        state = pd.read_csv(Path(row["state_csv"]))
        state = state.dropna(subset=["t"])
        if state.empty:
            continue
        y_curve = sanitize_curve(state, "t", "pallet_cy")
        vy_curve = sanitize_curve(state, "t", "pallet_vy")
        axs[0].plot(y_curve["t"], y_curve["pallet_cy"], color=palette[row["method"]], label=row["method"])
        axs[1].plot(vy_curve["t"], vy_curve["pallet_vy"], color=palette[row["method"]], label=row["method"])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Pallet COM y (m)")
    axs[0].set_title("Forklift: Pallet Lift")
    axs[0].legend()
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Pallet COM vy (m/s)")
    axs[1].set_title("Forklift: Pallet Vertical Velocity")
    save_fig(fig, fig_dir, "exp7_forklift_compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and compare STARK/PyChrono fixed-base forklift benchmark.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--ref-dt", type=float, default=0.0025)
    parser.add_argument("--end-time", type=float, default=1.2)
    parser.add_argument("--lift-start", type=float, default=0.3)
    parser.add_argument("--lift-end", type=float, default=1.0)
    parser.add_argument("--lift-speed", type=float, default=-0.10)
    parser.add_argument("--pallet-y", type=float, default=0.15)
    parser.add_argument("--pallet-z", type=float, default=3.02)
    parser.add_argument("--ground-pallet-friction", type=float, default=0.1)
    parser.add_argument("--fork-pallet-friction", type=float, default=0.1)
    parser.add_argument("--pychrono-nsc-compliance", type=float, default=1e-3)
    parser.add_argument("--pychrono-solver-max-iters", type=int, default=800)
    parser.add_argument("--pychrono-solver-tol", type=float, default=1e-10)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "exp7_forklift_summary.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)

    ref_row = run_stark_case(
        exe,
        "exp7_forklift_reference",
        args.ref_dt,
        args.end_time,
        args.lift_start,
        args.lift_end,
        args.lift_speed,
        args.pallet_y,
        args.pallet_z,
        args.ground_pallet_friction,
        args.fork_pallet_friction,
        True,
        args.force_run,
    )
    ref_state = load_state(Path(ref_row["state_csv"]))

    rows = [
        run_stark_case(
            exe, "exp7_forklift_soft", args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.pallet_y, args.pallet_z, args.ground_pallet_friction, args.fork_pallet_friction, False, args.force_run),
        run_stark_case(
            exe, "exp7_forklift_al", args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.pallet_y, args.pallet_z, args.ground_pallet_friction, args.fork_pallet_friction, True, args.force_run),
        run_pychrono_case(
            "nsc_psor", args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.pallet_y, args.pallet_z, args.fork_pallet_friction,
            args.pychrono_nsc_compliance, args.pychrono_solver_max_iters, args.pychrono_solver_tol, "", args.force_run),
        run_pychrono_case(
            "smc_penalty", args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.pallet_y, args.pallet_z, args.fork_pallet_friction,
            args.pychrono_nsc_compliance, args.pychrono_solver_max_iters, args.pychrono_solver_tol, "", args.force_run),
    ]
    for row in rows:
        row.update(error_against_reference(ref_state, load_state(Path(row["state_csv"]))))

    write_csv(rows, args.out_csv)
    plot_curves(rows, args.fig_dir)
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
