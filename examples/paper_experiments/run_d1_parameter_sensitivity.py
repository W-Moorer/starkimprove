#!/usr/bin/env python3
"""
D1 contact-parameter sensitivity sweep for the contact-centric paper framing.

This runner uses the fixed-base forklift benchmark (`exp7`) as a contact-dominant
scene and sweeps the contact strategy parameters that now define the method line:
- STARK_EXP7_MIN_CONTACT_STIFFNESS
- STARK_EXP7_CONTACT_ADAPTIVE_SCHEDULING
- STARK_EXP7_CONTACT_INERTIA_CONSISTENT
- STARK_NEWTON_TOL
- STARK_LINEAR_TOL

Output:
- output/paper_experiments/d1_parameter_sensitivity.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from study_utils import OUTPUT_BASE, latest_logger, parse_logger_metrics, resolve_executable


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUT_CSV = OUTPUT_BASE / "d1_parameter_sensitivity.csv"


def parse_list(values: str, cast=float) -> List[float]:
    out: List[float] = []
    for token in values.split(","):
        token = token.strip()
        if token:
            out.append(cast(token))
    if not out:
        raise ValueError("Empty value list.")
    return out


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric = [
        "t",
        "pallet_cy",
        "pallet_vy",
        "pallet_cz",
        "vertical_gap",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", "pallet_cy", "pallet_vy", "pallet_cz", "vertical_gap"])
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


def env_for_case(
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
    min_contact_stiffness: float,
    adaptive_scheduling: bool,
    inertia_consistent: bool,
    newton_tol: float,
    linear_tol: float,
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
    env["STARK_EXP7_MIN_CONTACT_STIFFNESS"] = f"{min_contact_stiffness:.12g}"
    env["STARK_EXP7_CONTACT_STIFFNESS_UPDATE"] = "1"
    env["STARK_EXP7_CONTACT_ADAPTIVE_SCHEDULING"] = "1" if adaptive_scheduling else "0"
    env["STARK_EXP7_CONTACT_INERTIA_CONSISTENT"] = "1" if inertia_consistent else "0"
    env["STARK_EXP7_ADAPTIVE_DT"] = "1"
    env["STARK_NEWTON_TOL"] = f"{newton_tol:.12g}"
    env["STARK_LINEAR_TOL"] = f"{linear_tol:.12g}"
    return env


def iter_cases(
    min_contact_stiffness_values: Iterable[float],
    adaptive_values: Iterable[int],
    inertia_values: Iterable[int],
    newton_tol_values: Iterable[float],
    linear_tol_values: Iterable[float],
):
    for kappa_min, adaptive, inertia, newton_tol, linear_tol in itertools.product(
        min_contact_stiffness_values,
        adaptive_values,
        inertia_values,
        newton_tol_values,
        linear_tol_values,
    ):
        yield {
            "min_contact_stiffness": float(kappa_min),
            "adaptive_scheduling": int(adaptive),
            "inertia_consistent": int(inertia),
            "newton_tol": float(newton_tol),
            "linear_tol": float(linear_tol),
        }


def run_case(
    exe: Path,
    case_idx: int,
    case: Dict[str, object],
    ref_state: pd.DataFrame,
    dt: float,
    end_time: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    pallet_y: float,
    pallet_z: float,
    ground_pallet_friction: float,
    fork_pallet_friction: float,
) -> Dict[str, object]:
    run_name = f"exp7_forklift_d1_case{case_idx:03d}"
    case_dir = OUTPUT_BASE / run_name
    env = env_for_case(
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
        float(case["min_contact_stiffness"]),
        bool(case["adaptive_scheduling"]),
        bool(case["inertia_consistent"]),
        float(case["newton_tol"]),
        float(case["linear_tol"]),
    )
    result = subprocess.run([str(exe), "exp7"], cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Case {case_idx} failed with exit code {result.returncode}")

    logger_path = latest_logger(case_dir)
    if logger_path is None:
        raise FileNotFoundError(f"No logger found in {case_dir}")
    metrics = parse_logger_metrics(logger_path)
    state_path = case_dir / "forklift_state.csv"
    state = load_state(state_path)
    status = classify_state(state, end_time, dt)

    row: Dict[str, object] = dict(case)
    row["case_idx"] = case_idx
    row["run_name"] = run_name
    row["logger_file"] = logger_path.name
    row["status"] = status
    row["total"] = metrics.get("total")
    row["newton_iterations"] = metrics.get("newton_iterations")
    row["linear_iterations"] = metrics.get("linear_iterations")
    row["failed_step_count"] = metrics.get("failed_step_count")
    row["failed_step_time"] = metrics.get("failed_step_time")
    row["hardening_count"] = metrics.get("hardening_count")
    row["contact_hardening_count"] = metrics.get("contact_hardening_count")
    row["final_pallet_y"] = float(state["pallet_cy"].iloc[-1]) if not state.empty else None
    row["final_pallet_z"] = float(state["pallet_cz"].iloc[-1]) if not state.empty else None
    row.update(error_against_reference(ref_state, state))
    return row


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_idx",
        "run_name",
        "logger_file",
        "status",
        "min_contact_stiffness",
        "adaptive_scheduling",
        "inertia_consistent",
        "newton_tol",
        "linear_tol",
        "total",
        "newton_iterations",
        "linear_iterations",
        "failed_step_count",
        "failed_step_time",
        "hardening_count",
        "contact_hardening_count",
        "final_pallet_y",
        "final_pallet_z",
        "rmse_pallet_y_to_ref",
        "rmse_pallet_vy_to_ref",
        "rmse_pallet_z_to_ref",
        "rmse_gap_to_ref",
        "composite_error",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D1 contact-parameter sensitivity sweep on the fixed-base forklift benchmark.")
    parser.add_argument("--exe", type=Path, default=None, help="Path to examples executable.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help=f"Output CSV (default: {DEFAULT_OUT_CSV}).")
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
    parser.add_argument("--min-contact-stiffness", type=str, default="1e2,1e3,1e4", help="Comma-separated min contact stiffness values.")
    parser.add_argument("--adaptive", type=str, default="0,1", help="Comma-separated adaptive-scheduling flags.")
    parser.add_argument("--inertia", type=str, default="0,1", help="Comma-separated inertia-consistent flags.")
    parser.add_argument("--newton-tol", type=str, default="1e-3,1e-4", help="Comma-separated Newton tolerance values.")
    parser.add_argument("--linear-tol", type=str, default="1.0,0.1", help="Comma-separated linear tolerance scale values.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0 means all).")
    parser.add_argument("--dry-run", action="store_true", help="Print cases without running simulations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe.resolve() if args.exe else None)

    ref_run_name = "exp7_forklift_d1_reference"
    ref_env = env_for_case(
        ref_run_name,
        args.ref_dt,
        args.end_time,
        args.lift_start,
        args.lift_end,
        args.lift_speed,
        args.pallet_y,
        args.pallet_z,
        args.ground_pallet_friction,
        args.fork_pallet_friction,
        1e3,
        True,
        True,
        1e-4,
        0.1,
    )
    if not args.dry_run:
        ref_ret = subprocess.run([str(exe), "exp7"], cwd=REPO_ROOT, env=ref_env)
        if ref_ret.returncode != 0:
            raise RuntimeError(f"Reference case failed with exit code {ref_ret.returncode}")
    ref_state = load_state(OUTPUT_BASE / ref_run_name / "forklift_state.csv")

    all_cases = list(
        iter_cases(
            parse_list(args.min_contact_stiffness, float),
            parse_list(args.adaptive, int),
            parse_list(args.inertia, int),
            parse_list(args.newton_tol, float),
            parse_list(args.linear_tol, float),
        )
    )
    if args.max_cases > 0:
        all_cases = all_cases[: args.max_cases]

    rows: List[Dict[str, object]] = []
    for case_idx, case in enumerate(all_cases, start=1):
        print(
            f"[d1] case={case_idx}/{len(all_cases)} "
            f"kappa_min={case['min_contact_stiffness']:.3g} "
            f"adaptive={case['adaptive_scheduling']} inertia={case['inertia_consistent']} "
            f"newton_tol={case['newton_tol']:.3g} linear_tol={case['linear_tol']:.3g}"
        )

        if args.dry_run:
            continue

        row = run_case(
            exe,
            case_idx,
            case,
            ref_state,
            args.dt,
            args.end_time,
            args.lift_start,
            args.lift_end,
            args.lift_speed,
            args.pallet_y,
            args.pallet_z,
            args.ground_pallet_friction,
            args.fork_pallet_friction,
        )
        rows.append(row)

    if not args.dry_run:
        write_csv(rows, args.out_csv.resolve())
        print(f"Wrote: {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[run_d1_parameter_sensitivity] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
