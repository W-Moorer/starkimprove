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

from study_utils import (
    FIGS_DIR,
    OUTPUT_BASE,
    build_conda_python_command,
    latest_logger,
    parse_logger_metrics,
    resolve_executable,
    sanitize_curve,
    save_fig,
    setup_axes,
)


def stark_env(
    run_name: str,
    dt: float,
    end_time: float,
    chain_count: int,
    link_length: float,
    link_width: float,
    link_thickness: float,
    link_mass: float,
    chain_spacing: float,
    anchor_x: float,
    anchor_y: float,
    anchor_z: float,
    initial_angle_deg: float,
    ground_top_z: float,
    min_contact_stiffness: float,
    contact_stiffness_update: bool,
    contact_adaptive_scheduling: bool,
    contact_inertia_consistent: bool,
    adaptive_dt: bool,
) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP2_RUN_NAME"] = run_name
    env["STARK_EXP2_DT"] = f"{dt:.12g}"
    env["STARK_EXP2_END_TIME"] = f"{end_time:.12g}"
    env["STARK_EXP2_CHAIN_COUNT"] = str(chain_count)
    env["STARK_EXP2_LINK_LENGTH"] = f"{link_length:.12g}"
    env["STARK_EXP2_LINK_WIDTH"] = f"{link_width:.12g}"
    env["STARK_EXP2_LINK_THICKNESS"] = f"{link_thickness:.12g}"
    env["STARK_EXP2_LINK_MASS"] = f"{link_mass:.12g}"
    env["STARK_EXP2_CHAIN_SPACING"] = f"{chain_spacing:.12g}"
    env["STARK_EXP2_ANCHOR_X"] = f"{anchor_x:.12g}"
    env["STARK_EXP2_ANCHOR_Y"] = f"{anchor_y:.12g}"
    env["STARK_EXP2_ANCHOR_Z"] = f"{anchor_z:.12g}"
    env["STARK_EXP2_INITIAL_ANGLE_DEG"] = f"{initial_angle_deg:.12g}"
    env["STARK_EXP2_GROUND_TOP_Z"] = f"{ground_top_z:.12g}"
    env["STARK_EXP2_CONTACT_THICKNESS"] = "1e-3"
    env["STARK_EXP2_MIN_CONTACT_STIFFNESS"] = f"{min_contact_stiffness:.12g}"
    env["STARK_EXP2_CONTACT_STIFFNESS_UPDATE"] = "1" if contact_stiffness_update else "0"
    env["STARK_EXP2_CONTACT_ADAPTIVE_SCHEDULING"] = "1" if contact_adaptive_scheduling else "0"
    env["STARK_EXP2_CONTACT_INERTIA_CONSISTENT"] = "1" if contact_inertia_consistent else "0"
    env["STARK_EXP2_ADAPTIVE_DT"] = "1" if adaptive_dt else "0"
    return env


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = ["t", "tip_cx", "tip_cy", "tip_cz", "tip_vx", "tip_vy", "tip_vz", "min_gap_ground", "max_joint_drift", "max_link_speed"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    required = ["t", "tip_cx", "tip_cz", "tip_vx", "tip_vz"]
    if "min_gap_ground" in df.columns:
        required.append("min_gap_ground")
    df = df.dropna(subset=required)
    df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t").reset_index(drop=True)
    return df


def classify_state(df: pd.DataFrame, end_time: float, dt: float) -> str:
    if df.empty:
        return "invalid"
    arr = df[["tip_cx", "tip_cz", "tip_vx", "tip_vz", "min_gap_ground"]].to_numpy()
    if not np.isfinite(arr).all():
        return "invalid"
    if float(np.abs(df["tip_cx"]).max()) > 50.0 or float(np.abs(df["tip_cz"]).max()) > 50.0:
        return "unstable"
    if float(np.abs(df[["tip_vx", "tip_vz"]]).to_numpy().max()) > 200.0:
        return "unstable"
    # Local benchmarks log accepted states at the beginning of each time step,
    # so a complete run typically ends at end_time - dt rather than exactly end_time.
    if float(df["t"].iloc[-1]) < end_time - max(dt, 1e-9):
        return "incomplete"
    return "ok"


def direct_contact_stats(cur: pd.DataFrame) -> Dict[str, float]:
    if cur.empty:
        return {
            "final_tip_z": float("nan"),
            "minimum_tip_z": float("nan"),
            "max_abs_tip_vz": float("nan"),
            "minimum_gap_ground": float("nan"),
            "first_contact_time": float("nan"),
            "post_contact_gap_std": float("nan"),
        }
    has_gap = "min_gap_ground" in cur.columns
    first_contact = cur.loc[cur["min_gap_ground"] <= 1e-4, "t"] if has_gap else pd.Series(dtype=float)
    return {
        "final_tip_z": float(cur["tip_cz"].iloc[-1]),
        "minimum_tip_z": float(cur["tip_cz"].min()),
        "max_abs_tip_vz": float(cur["tip_vz"].abs().max()),
        "minimum_gap_ground": float(cur["min_gap_ground"].min()) if has_gap else float("nan"),
        "first_contact_time": float(first_contact.iloc[0]) if not first_contact.empty else float("nan"),
        "post_contact_gap_std": float(cur.loc[cur["t"] >= 0.15, "min_gap_ground"].std()) if has_gap else float("nan"),
    }


def error_against_recurdyn(ref: pd.DataFrame, cur: pd.DataFrame) -> Dict[str, float]:
    if cur.empty:
        return {
            "rmse_tip_x_to_recurdyn": float("inf"),
            "rmse_tip_z_to_recurdyn": float("inf"),
            "rmse_tip_vx_to_recurdyn": float("inf"),
            "rmse_tip_vz_to_recurdyn": float("inf"),
            "composite_recurdyn_error": float("inf"),
        }

    t = cur["t"].to_numpy()
    ref_x = np.interp(t, ref["t"].to_numpy(), ref["tip_cx"].to_numpy())
    ref_z = np.interp(t, ref["t"].to_numpy(), ref["tip_cz"].to_numpy())
    ref_vx = np.interp(t, ref["t"].to_numpy(), ref["tip_vx"].to_numpy())
    ref_vz = np.interp(t, ref["t"].to_numpy(), ref["tip_vz"].to_numpy())

    rmse_x = float(np.sqrt(np.mean((cur["tip_cx"].to_numpy() - ref_x) ** 2)))
    rmse_z = float(np.sqrt(np.mean((cur["tip_cz"].to_numpy() - ref_z) ** 2)))
    rmse_vx = float(np.sqrt(np.mean((cur["tip_vx"].to_numpy() - ref_vx) ** 2)))
    rmse_vz = float(np.sqrt(np.mean((cur["tip_vz"].to_numpy() - ref_vz) ** 2)))

    x_scale = max(1e-3, float(ref["tip_cx"].max() - ref["tip_cx"].min()))
    z_scale = max(1e-3, float(ref["tip_cz"].max() - ref["tip_cz"].min()))
    vx_scale = max(1e-3, float(np.abs(ref["tip_vx"]).max()))
    vz_scale = max(1e-3, float(np.abs(ref["tip_vz"]).max()))
    composite = rmse_x / x_scale + rmse_z / z_scale + rmse_vx / vx_scale + rmse_vz / vz_scale

    return {
        "rmse_tip_x_to_recurdyn": rmse_x,
        "rmse_tip_z_to_recurdyn": rmse_z,
        "rmse_tip_vx_to_recurdyn": rmse_vx,
        "rmse_tip_vz_to_recurdyn": rmse_vz,
        "composite_recurdyn_error": composite,
    }


def run_stark_case(
    exe: Path,
    run_name: str,
    dt: float,
    end_time: float,
    chain_count: int,
    link_length: float,
    link_width: float,
    link_thickness: float,
    link_mass: float,
    chain_spacing: float,
    anchor_x: float,
    anchor_y: float,
    anchor_z: float,
    initial_angle_deg: float,
    ground_top_z: float,
    min_contact_stiffness: float,
    contact_stiffness_update: bool,
    contact_adaptive_scheduling: bool,
    contact_inertia_consistent: bool,
    adaptive_dt: bool,
    method_label: str,
    force_run: bool,
) -> Dict[str, object]:
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    state_csv = case_dir / "chain10_impact_state.csv"
    if force_run or logger is None or not state_csv.exists():
        cmd = [str(exe), "exp2_chain10"]
        env = stark_env(
            run_name, dt, end_time, chain_count, link_length, link_width, link_thickness, link_mass, chain_spacing,
            anchor_x, anchor_y, anchor_z, initial_angle_deg, ground_top_z, min_contact_stiffness,
            contact_stiffness_update, contact_adaptive_scheduling, contact_inertia_consistent, adaptive_dt,
        )
        print(f"[a2] run local {run_name}")
        ret = subprocess.run(cmd, cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"A2 local benchmark failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")
    state = load_state(state_csv)
    metrics = parse_logger_metrics(logger)
    status = classify_state(state, end_time, dt)
    return {
        "framework": "This work",
        "method": method_label,
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "failed_step_time": metrics.get("failed_steps"),
        "state_csv": str(state_csv),
        "status": status,
    }


def run_pychrono_case(
    mode: str,
    dt: float,
    end_time: float,
    chain_count: int,
    link_length: float,
    link_width: float,
    link_thickness: float,
    link_mass: float,
    chain_spacing: float,
    anchor_x: float,
    anchor_y: float,
    anchor_z: float,
    initial_angle_deg: float,
    ground_top_z: float,
    nsc_compliance: float,
    solver_max_iters: int,
    solver_tol: float,
    smc_kn: float,
    smc_gn: float,
    force_run: bool,
) -> Dict[str, object]:
    tag = f"dt{dt:.0e}_a{initial_angle_deg:.0f}".replace("+", "")
    case_dir = OUTPUT_BASE / f"pychrono_exp2_chain10_{mode}_{tag}"
    summary_csv = case_dir / "summary.csv"
    state_csv = case_dir / "chain10_impact_state.csv"
    if force_run or not summary_csv.exists() or not state_csv.exists():
        script_path = Path(__file__).resolve().parent / "pychrono_chain10_impact_benchmark.py"
        cmd = build_conda_python_command(
            script_path,
            "--mode",
            mode,
            "--dt",
            f"{dt:.12g}",
            "--end-time",
            f"{end_time:.12g}",
            "--tag",
            tag,
            "--solver-max-iters",
            solver_max_iters,
            "--solver-tol",
            f"{solver_tol:.12g}",
            "--chain-count",
            chain_count,
            "--link-length",
            f"{link_length:.12g}",
            "--link-width",
            f"{link_width:.12g}",
            "--link-thickness",
            f"{link_thickness:.12g}",
            "--link-mass",
            f"{link_mass:.12g}",
            "--chain-spacing",
            f"{chain_spacing:.12g}",
            "--anchor-x",
            f"{anchor_x:.12g}",
            "--anchor-y",
            f"{anchor_y:.12g}",
            "--anchor-z",
            f"{anchor_z:.12g}",
            "--initial-angle-deg",
            f"{initial_angle_deg:.12g}",
            "--ground-top-z",
            f"{ground_top_z:.12g}",
            "--nsc-compliance",
            f"{nsc_compliance:.12g}",
            "--smc-kn",
            f"{smc_kn:.12g}",
            "--smc-gn",
            f"{smc_gn:.12g}",
            "--output-base",
            OUTPUT_BASE,
        )
        print(f"[a2] run PyChrono {mode}")
        ret = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
        if ret.returncode != 0:
            raise RuntimeError(f"PyChrono A2 benchmark failed: {mode}")
    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    state = load_state(state_csv)
    status = classify_state(state, end_time, dt)
    return {
        "framework": "PyChrono",
        "method": "PyChrono NSC/PSOR" if mode == "nsc_psor" else "PyChrono SMC",
        "run_name": case_dir.name,
        "logger_file": "",
        "total": float(row["wall_time_s"]),
        "newton_iterations": None,
        "linear_iterations": float(row["avg_solver_iterations"]),
        "failed_step_time": None,
        "state_csv": str(state_csv),
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


def plot_curves(rows: List[Dict[str, object]], recurdyn_state: pd.DataFrame, fig_dir: Path):
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True)
    for ax in axs:
        setup_axes(ax)
    palette = {
        "Contact-consistent IPC": "#1f77b4",
        "PyChrono NSC/PSOR": "#2ca02c",
        "PyChrono SMC": "#d62728",
    }
    axs[0].plot(
        recurdyn_state["t"],
        recurdyn_state["tip_cz"],
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.4,
        label="RecurDyn",
    )
    for row in rows:
        if row.get("status") != "ok" or row.get("framework") == "RecurDyn":
            continue
        state = pd.read_csv(Path(row["state_csv"]))
        z_curve = sanitize_curve(state, "t", "tip_cz")
        gap_curve = sanitize_curve(state, "t", "min_gap_ground")
        axs[0].plot(z_curve["t"], z_curve["tip_cz"], color=palette[row["method"]], label=row["method"])
        axs[1].plot(gap_curve["t"], gap_curve["min_gap_ground"], color=palette[row["method"]], label=row["method"])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Tip COM z (m)")
    axs[0].set_title("A2: Frictionless chain impact tip height")
    axs[0].legend()
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Minimum gap to ground (m)")
    axs[1].set_title("A2: Frictionless chain impact gap")
    save_fig(fig, fig_dir, "a2_chain10_impact_compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frictionless chain10 impact A2 benchmark and compare with PyChrono.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--chain-count", type=int, default=10)
    parser.add_argument("--link-length", type=float, default=0.30)
    parser.add_argument("--link-width", type=float, default=0.06)
    parser.add_argument("--link-thickness", type=float, default=0.06)
    parser.add_argument("--link-mass", type=float, default=1.0)
    parser.add_argument("--chain-spacing", type=float, default=0.35)
    parser.add_argument("--anchor-x", type=float, default=0.0)
    parser.add_argument("--anchor-y", type=float, default=0.0)
    parser.add_argument("--anchor-z", type=float, default=0.75)
    parser.add_argument("--initial-angle-deg", type=float, default=-10.0)
    parser.add_argument("--ground-top-z", type=float, default=0.0)
    parser.add_argument("--stark-min-contact-stiffness", type=float, default=1e3)
    parser.add_argument("--pychrono-nsc-compliance", type=float, default=1e-4)
    parser.add_argument("--pychrono-solver-max-iters", type=int, default=600)
    parser.add_argument("--pychrono-solver-tol", type=float, default=1e-10)
    parser.add_argument("--pychrono-smc-kn", type=float, default=5e6)
    parser.add_argument("--pychrono-smc-gn", type=float, default=5e2)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--recurdyn-csv", type=Path, default=Path(__file__).resolve().parents[2] / "temps" / "chain10.csv")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "a2_chain10_impact_summary.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)
    recurdyn_state = load_state(args.recurdyn_csv)

    rows = [
        {
            "framework": "RecurDyn",
            "method": "RecurDyn",
            "run_name": args.recurdyn_csv.stem,
            "logger_file": "",
            "total": None,
            "newton_iterations": None,
            "linear_iterations": None,
            "failed_step_time": None,
            "state_csv": str(args.recurdyn_csv),
            "status": "reference",
        },
        run_stark_case(
            exe, "exp2_chain10_a2_method", args.dt, args.end_time, args.chain_count,
            args.link_length, args.link_width, args.link_thickness, args.link_mass, args.chain_spacing,
            args.anchor_x, args.anchor_y, args.anchor_z, args.initial_angle_deg, args.ground_top_z,
            args.stark_min_contact_stiffness, True, True, True, True, "Contact-consistent IPC", args.force_run,
        ),
        run_pychrono_case(
            "nsc_psor", args.dt, args.end_time, args.chain_count, args.link_length, args.link_width,
            args.link_thickness, args.link_mass, args.chain_spacing, args.anchor_x, args.anchor_y,
            args.anchor_z, args.initial_angle_deg, args.ground_top_z, args.pychrono_nsc_compliance,
            args.pychrono_solver_max_iters, args.pychrono_solver_tol, args.pychrono_smc_kn,
            args.pychrono_smc_gn, args.force_run,
        ),
        run_pychrono_case(
            "smc_penalty", args.dt, args.end_time, args.chain_count, args.link_length, args.link_width,
            args.link_thickness, args.link_mass, args.chain_spacing, args.anchor_x, args.anchor_y,
            args.anchor_z, args.initial_angle_deg, args.ground_top_z, args.pychrono_nsc_compliance,
            args.pychrono_solver_max_iters, args.pychrono_solver_tol, args.pychrono_smc_kn,
            args.pychrono_smc_gn, args.force_run,
        ),
    ]
    for row in rows:
        row.update(direct_contact_stats(load_state(Path(row["state_csv"]))))
        if row["framework"] != "RecurDyn":
            row.update(error_against_recurdyn(recurdyn_state, load_state(Path(row["state_csv"]))))

    write_csv(rows, args.out_csv)
    plot_curves(rows, recurdyn_state, args.fig_dir)
    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
