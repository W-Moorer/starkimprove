#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_utils import FIGS_DIR, OUTPUT_BASE, build_conda_python_command, latest_logger, parse_logger_metrics, resolve_executable, sanitize_curve, save_fig, setup_axes


def parse_obj_bounds(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    min_v = np.full(3, np.inf, dtype=float)
    max_v = np.full(3, -np.inf, dtype=float)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            _, xs, ys, zs, *_ = line.split()
            vals = np.array([float(xs), float(ys), float(zs)], dtype=float)
            min_v = np.minimum(min_v, vals)
            max_v = np.maximum(max_v, vals)
    if not np.isfinite(min_v).all() or not np.isfinite(max_v).all():
        raise ValueError(f"Failed to parse OBJ bounds from {path}")
    return min_v, max_v


def estimate_fork_tine_top_y_local(path: Path, fork_reference_y: float) -> float:
    best = -np.inf
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            _, xs, ys, zs, *_ = line.split()
            x = float(xs)
            y = float(ys) - fork_reference_y
            z = float(zs)
            if 0.20 <= abs(x) <= 0.50 and y <= 0.0 and z >= 2.20:
                best = max(best, y)
    if not np.isfinite(best):
        raise ValueError(f"Failed to estimate fork tine support plane from {path}")
    return float(best)


def pallet_default_properties() -> Dict[str, float]:
    pallet_path = Path(__file__).resolve().parents[2] / "models" / "pallet.obj"
    min_v, max_v = parse_obj_bounds(pallet_path)
    size = np.maximum(max_v - min_v, 1e-6)
    mass = 300.0 * float(np.prod(size))
    ixx = mass * float(size[1] ** 2 + size[2] ** 2) / 12.0
    iyy = mass * float(size[0] ** 2 + size[2] ** 2) / 12.0
    izz = mass * float(size[0] ** 2 + size[1] ** 2) / 12.0
    return {
        "mass": mass,
        "ixx": ixx,
        "iyy": iyy,
        "izz": izz,
    }


def default_exp7_pallet_y(initial_gap: float = 0.009018) -> float:
    repo = Path(__file__).resolve().parents[2]
    pallet_path = repo / "models" / "pallet.obj"
    fork_path = repo / "models" / "forklift" / "forks.obj"
    pallet_min, _ = parse_obj_bounds(pallet_path)
    fork_ref_y = 0.362
    fork_support_y_local = estimate_fork_tine_top_y_local(fork_path, fork_ref_y)
    return float(fork_ref_y + fork_support_y_local + initial_gap - pallet_min[1])


def stark_env(
    scenario: str,
    run_name: str,
    dt: float,
    end_time: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    lift_max_force: float,
    pallet_y: float,
    pallet_z: float,
    pallet_mass: float,
    pallet_ixx: float,
    pallet_iyy: float,
    pallet_izz: float,
    ground_pallet_friction: float,
    fork_pallet_friction: float,
    min_contact_stiffness: float,
    contact_stiffness_update: bool,
    contact_adaptive_scheduling: bool,
    contact_inertia_consistent: bool,
) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP7_SCENARIO"] = scenario
    env["STARK_EXP7_RUN_NAME"] = run_name
    env["STARK_EXP7_DT"] = f"{dt:.12g}"
    env["STARK_EXP7_END_TIME"] = f"{end_time:.12g}"
    env["STARK_EXP7_LIFT_START"] = f"{lift_start:.12g}"
    env["STARK_EXP7_LIFT_END"] = f"{lift_end:.12g}"
    env["STARK_EXP7_LIFT_SPEED"] = f"{lift_speed:.12g}"
    env["STARK_EXP7_LIFT_MAX_FORCE"] = f"{lift_max_force:.12g}"
    env["STARK_EXP7_PALLET_Y"] = f"{pallet_y:.12g}"
    env["STARK_EXP7_PALLET_Z"] = f"{pallet_z:.12g}"
    env["STARK_EXP7_PALLET_MASS"] = f"{pallet_mass:.12g}"
    env["STARK_EXP7_PALLET_IXX"] = f"{pallet_ixx:.12g}"
    env["STARK_EXP7_PALLET_IYY"] = f"{pallet_iyy:.12g}"
    env["STARK_EXP7_PALLET_IZZ"] = f"{pallet_izz:.12g}"
    env["STARK_EXP7_GROUND_PALLET_FRICTION"] = f"{ground_pallet_friction:.12g}"
    env["STARK_EXP7_FORK_PALLET_FRICTION"] = f"{fork_pallet_friction:.12g}"
    env["STARK_EXP7_MIN_CONTACT_STIFFNESS"] = f"{min_contact_stiffness:.12g}"
    env["STARK_EXP7_CONTACT_STIFFNESS_UPDATE"] = "1" if contact_stiffness_update else "0"
    env["STARK_EXP7_CONTACT_ADAPTIVE_SCHEDULING"] = "1" if contact_adaptive_scheduling else "0"
    env["STARK_EXP7_CONTACT_INERTIA_CONSISTENT"] = "1" if contact_inertia_consistent else "0"
    env["STARK_EXP7_ADAPTIVE_DT"] = "0" if scenario == "demo_faithful" else "1"
    return env


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "t", "target_fork_vy", "target_fork_y", "target_lift_v",
        "fork_cy", "fork_vy",
        "pallet_cy", "pallet_vy", "pallet_cz", "pallet_vz",
        "vertical_gap", "effective_vertical_gap", "actuator_force_proxy",
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
    gap_col = "effective_vertical_gap" if "effective_vertical_gap" in df.columns else "vertical_gap"
    if not np.isfinite(df[["pallet_cy", "pallet_vy", "pallet_cz", gap_col]].to_numpy()).all():
        return "invalid"
    if float(np.abs(df["pallet_cy"]).max()) > 10.0:
        return "unstable"
    if float(np.abs(df["pallet_vy"]).max()) > 100.0:
        return "unstable"
    if float(np.abs(df[gap_col]).max()) > 10.0:
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
    cur_gap_col = "effective_vertical_gap" if "effective_vertical_gap" in cur.columns else "vertical_gap"
    ref_gap_col = "effective_vertical_gap" if "effective_vertical_gap" in ref.columns else "vertical_gap"
    gap = cur[cur_gap_col].to_numpy()

    ref_y = np.interp(t, ref["t"].to_numpy(), ref["pallet_cy"].to_numpy())
    ref_vy = np.interp(t, ref["t"].to_numpy(), ref["pallet_vy"].to_numpy())
    ref_z = np.interp(t, ref["t"].to_numpy(), ref["pallet_cz"].to_numpy())
    ref_gap = np.interp(t, ref["t"].to_numpy(), ref[ref_gap_col].to_numpy())

    rmse_y = float(np.sqrt(np.mean((pallet_y - ref_y) ** 2)))
    rmse_vy = float(np.sqrt(np.mean((pallet_vy - ref_vy) ** 2)))
    rmse_z = float(np.sqrt(np.mean((pallet_z - ref_z) ** 2)))
    rmse_gap = float(np.sqrt(np.mean((gap - ref_gap) ** 2)))

    y_scale = max(1e-3, float(ref["pallet_cy"].max() - ref["pallet_cy"].min()))
    vy_scale = max(1e-3, float(np.abs(ref["pallet_vy"]).max()))
    z_scale = max(1e-3, float(ref["pallet_cz"].max() - ref["pallet_cz"].min()))
    gap_scale = max(1e-3, float(np.abs(ref[ref_gap_col]).max()))
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
    scenario: str,
    run_name: str,
    dt: float,
    end_time: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    lift_max_force: float,
    pallet_y: float,
    pallet_z: float,
    pallet_mass: float,
    pallet_ixx: float,
    pallet_iyy: float,
    pallet_izz: float,
    ground_pallet_friction: float,
    fork_pallet_friction: float,
    min_contact_stiffness: float,
    contact_stiffness_update: bool,
    contact_adaptive_scheduling: bool,
    contact_inertia_consistent: bool,
    method_label: str,
    force_run: bool,
) -> Dict[str, object]:
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None or not (case_dir / "forklift_state.csv").exists():
        cmd = [str(exe), "exp7"]
        env = stark_env(
            scenario,
            run_name,
            dt,
            end_time,
            lift_start,
            lift_end,
            lift_speed,
            lift_max_force,
            pallet_y,
            pallet_z,
            pallet_mass,
            pallet_ixx,
            pallet_iyy,
            pallet_izz,
            ground_pallet_friction,
            fork_pallet_friction,
            min_contact_stiffness,
            contact_stiffness_update,
            contact_adaptive_scheduling,
            contact_inertia_consistent,
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
        "scenario": scenario,
        "method": method_label,
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "failed_step_time": metrics.get("failed_steps"),
        "state_csv": str(state_csv),
        "final_pallet_y": float(state["pallet_cy"].iloc[-1]) if not state.empty else None,
        "final_pallet_z": float(state["pallet_cz"].iloc[-1]) if not state.empty else None,
        "lift_max_force": lift_max_force,
        "initial_effective_gap": float(state["effective_vertical_gap"].iloc[0]) if ("effective_vertical_gap" in state.columns and not state.empty) else (float(state["vertical_gap"].iloc[0]) if not state.empty else None),
        "pallet_mass": pallet_mass,
        "pallet_ixx": pallet_ixx,
        "pallet_iyy": pallet_iyy,
        "pallet_izz": pallet_izz,
        "min_contact_stiffness": min_contact_stiffness,
        "contact_stiffness_update": int(contact_stiffness_update),
        "contact_adaptive_scheduling": int(contact_adaptive_scheduling),
        "contact_inertia_consistent": int(contact_inertia_consistent),
        "status": status,
    }


def run_pychrono_case(
    mode: str,
    scenario: str,
    dt: float,
    end_time: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    lift_max_force: float,
    pallet_y: float,
    pallet_z: float,
    pallet_mass: float,
    pallet_ixx: float,
    pallet_iyy: float,
    pallet_izz: float,
    friction: float,
    nsc_compliance: float,
    solver_max_iters: int,
    solver_tol: float,
    velocity_servo_gain: float | None,
    tag_extra: str,
    force_run: bool,
) -> Dict[str, object]:
    tag = f"dt{dt:.0e}_ls{lift_start:.2f}_le{lift_end:.2f}".replace("+", "")
    if tag_extra:
        tag = f"{tag}_{tag_extra}"
    scenario_suffix = "" if scenario == "benchmark" else f"_{scenario}"
    case_dir = OUTPUT_BASE / f"pychrono_exp7_forklift_{mode}{scenario_suffix}_{tag}"
    summary_csv = case_dir / "summary.csv"
    state_csv = case_dir / "forklift_state.csv"
    if force_run or not summary_csv.exists() or not state_csv.exists():
        script_path = Path(__file__).resolve().parent / "pychrono_forklift_benchmark.py"
        cmd_args: list[object] = [
            "--mode",
            mode,
            "--scenario",
            scenario,
            "--dt",
            f"{dt:.12g}",
            "--end-time",
            f"{end_time:.12g}",
            "--lift-start",
            f"{lift_start:.12g}",
            "--lift-end",
            f"{lift_end:.12g}",
            "--lift-speed",
            f"{lift_speed:.12g}",
            "--lift-max-force",
            f"{lift_max_force:.12g}",
            "--pallet-y",
            f"{pallet_y:.12g}",
            "--pallet-z",
            f"{pallet_z:.12g}",
            "--pallet-mass",
            f"{pallet_mass:.12g}",
            "--pallet-ixx",
            f"{pallet_ixx:.12g}",
            "--pallet-iyy",
            f"{pallet_iyy:.12g}",
            "--pallet-izz",
            f"{pallet_izz:.12g}",
            "--friction",
            f"{friction:.12g}",
            "--nsc-compliance",
            f"{nsc_compliance:.12g}",
            "--solver-max-iters",
            solver_max_iters,
            "--solver-tol",
            f"{solver_tol:.12g}",
            "--output-base",
            OUTPUT_BASE,
            "--tag",
            tag,
        ]
        if velocity_servo_gain is not None:
            cmd_args.extend(["--velocity-servo-gain", f"{velocity_servo_gain:.12g}"])
        cmd = build_conda_python_command(
            script_path,
            *cmd_args,
        )
        print(f"[exp7] run PyChrono {mode}")
        ret = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
        if ret.returncode != 0:
            raise RuntimeError(f"PyChrono exp7 failed: {mode}")
    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    state = load_state(state_csv)
    status = classify_state(state, end_time, dt)
    return {
        "framework": "PyChrono",
        "scenario": scenario,
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
        "lift_max_force": lift_max_force,
        "velocity_servo_gain": velocity_servo_gain,
        "initial_effective_gap": float(row["initial_effective_gap"]) if "initial_effective_gap" in row else (float(state["effective_vertical_gap"].iloc[0]) if ("effective_vertical_gap" in state.columns and not state.empty) else None),
        "pallet_mass": float(row["pallet_mass"]) if "pallet_mass" in row else pallet_mass,
        "pallet_ixx": float(row["pallet_ixx"]) if "pallet_ixx" in row else pallet_ixx,
        "pallet_iyy": float(row["pallet_iyy"]) if "pallet_iyy" in row else pallet_iyy,
        "pallet_izz": float(row["pallet_izz"]) if "pallet_izz" in row else pallet_izz,
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
        "STARK fixed-kappa baseline": "#e377c2",
        "STARK contact-consistent IPC": "#1f77b4",
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
    parser = argparse.ArgumentParser(description="Run and compare STARK/PyChrono fixed-base forklift benchmark under the contact-centric paper framing.")
    parser.add_argument("--scenario", choices=["benchmark", "demo_faithful"], default="benchmark")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--ref-dt", type=float, default=0.0025)
    parser.add_argument("--end-time", type=float, default=1.2)
    parser.add_argument("--lift-start", type=float, default=0.3)
    parser.add_argument("--lift-end", type=float, default=1.0)
    parser.add_argument("--lift-speed", type=float, default=-0.10)
    parser.add_argument("--lift-max-force", type=float, default=2e4)
    parser.add_argument("--pallet-y", type=float, default=default_exp7_pallet_y())
    parser.add_argument("--pallet-z", type=float, default=3.02)
    parser.add_argument("--pallet-mass", type=float, default=None)
    parser.add_argument("--pallet-ixx", type=float, default=None)
    parser.add_argument("--pallet-iyy", type=float, default=None)
    parser.add_argument("--pallet-izz", type=float, default=None)
    parser.add_argument("--ground-pallet-friction", type=float, default=0.1)
    parser.add_argument("--fork-pallet-friction", type=float, default=0.1)
    parser.add_argument("--stark-min-contact-stiffness", type=float, default=1e3)
    parser.add_argument("--pychrono-nsc-compliance", type=float, default=1e-3)
    parser.add_argument("--pychrono-solver-max-iters", type=int, default=800)
    parser.add_argument("--pychrono-solver-tol", type=float, default=1e-10)
    parser.add_argument("--pychrono-velocity-servo-gain", type=float, default=None)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "exp7_forklift_summary.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scenario == "demo_faithful":
        if abs(args.pallet_y - default_exp7_pallet_y()) < 1e-12:
            args.pallet_y = 0.4
        if abs(args.pallet_z - 3.02) < 1e-12:
            args.pallet_z = 3.0
        if abs(args.lift_speed + 0.10) < 1e-12:
            args.lift_speed = 0.0
        if abs(args.ground_pallet_friction - 0.1) < 1e-12:
            args.ground_pallet_friction = 1.0
        if abs(args.fork_pallet_friction - 0.1) < 1e-12:
            args.fork_pallet_friction = 1.0
        if args.out_csv == OUTPUT_BASE / "exp7_forklift_summary.csv":
            args.out_csv = OUTPUT_BASE / "exp7_forklift_demo_faithful_summary.csv"
    exe = resolve_executable(args.exe)
    pallet_defaults = pallet_default_properties()
    pallet_mass = args.pallet_mass if args.pallet_mass is not None else pallet_defaults["mass"]
    pallet_ixx = args.pallet_ixx if args.pallet_ixx is not None else pallet_defaults["ixx"]
    pallet_iyy = args.pallet_iyy if args.pallet_iyy is not None else pallet_defaults["iyy"]
    pallet_izz = args.pallet_izz if args.pallet_izz is not None else pallet_defaults["izz"]

    ref_row = run_stark_case(
        exe,
        args.scenario,
        "exp7_forklift_reference",
        args.ref_dt,
        args.end_time,
        args.lift_start,
        args.lift_end,
        args.lift_speed,
        args.lift_max_force,
        args.pallet_y,
        args.pallet_z,
        pallet_mass,
        pallet_ixx,
        pallet_iyy,
        pallet_izz,
        args.ground_pallet_friction,
        args.fork_pallet_friction,
        args.stark_min_contact_stiffness,
        True,
        True,
        True,
        "STARK contact-consistent",
        args.force_run,
    )
    ref_state = load_state(Path(ref_row["state_csv"]))

    rows = [
        run_stark_case(
            exe, args.scenario, "exp7_forklift_fixed", args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.lift_max_force,
            args.pallet_y, args.pallet_z, pallet_mass, pallet_ixx, pallet_iyy, pallet_izz, args.ground_pallet_friction, args.fork_pallet_friction,
            args.stark_min_contact_stiffness, True, False, False, "STARK fixed-kappa baseline", args.force_run),
        run_stark_case(
            exe, args.scenario, "exp7_forklift_contact_consistent", args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.lift_max_force,
            args.pallet_y, args.pallet_z, pallet_mass, pallet_ixx, pallet_iyy, pallet_izz, args.ground_pallet_friction, args.fork_pallet_friction,
            args.stark_min_contact_stiffness, True, True, True, "STARK contact-consistent IPC", args.force_run),
        run_pychrono_case(
            "nsc_psor", args.scenario, args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.lift_max_force, args.pallet_y, args.pallet_z, pallet_mass, pallet_ixx, pallet_iyy, pallet_izz, args.fork_pallet_friction,
            args.pychrono_nsc_compliance, args.pychrono_solver_max_iters, args.pychrono_solver_tol, args.pychrono_velocity_servo_gain, "", args.force_run),
        run_pychrono_case(
            "smc_penalty", args.scenario, args.dt, args.end_time, args.lift_start, args.lift_end, args.lift_speed,
            args.lift_max_force, args.pallet_y, args.pallet_z, pallet_mass, pallet_ixx, pallet_iyy, pallet_izz, args.fork_pallet_friction,
            args.pychrono_nsc_compliance, args.pychrono_solver_max_iters, args.pychrono_solver_tol, args.pychrono_velocity_servo_gain, "", args.force_run),
    ]
    for row in rows:
        row.update(error_against_reference(ref_state, load_state(Path(row["state_csv"]))))

    write_csv(rows, args.out_csv)
    plot_curves(rows, args.fig_dir)
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
