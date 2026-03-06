#!/usr/bin/env python3
"""
D1 parameter sensitivity sweep for AL-IPC joint constraints.

This runner drives the existing `exp4` 10-link chain scene through environment variables:
- STARK_JOINT_AL_ENABLED=1
- STARK_JOINT_AL_RHO0
- STARK_JOINT_AL_RHO_UPDATE_RATIO
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
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
EXP4_AL_DIR = OUTPUT_BASE / "exp4_coupled_joints_al_d1"
DEFAULT_OUT_CSV = OUTPUT_BASE / "d1_parameter_sensitivity.csv"
LOGGER_GLOB = "logger_*.txt"
DEFAULT_EXE_CANDIDATES = [
    REPO_ROOT / "build" / "examples" / "Release" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "Debug" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "examples.exe",
]


def resolve_executable(path_arg: Optional[Path]) -> Path:
    if path_arg is not None:
        if path_arg.exists():
            return path_arg
        raise FileNotFoundError(f"Executable not found: {path_arg}")
    for candidate in DEFAULT_EXE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find examples executable. Build target `examples` first or pass --exe.")


def normalize_key(raw: str) -> str:
    key = raw.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def latest_logger(case_dir: Path) -> Optional[Path]:
    if not case_dir.exists():
        return None
    logger_files = sorted(case_dir.glob(LOGGER_GLOB), key=lambda p: p.stat().st_mtime)
    if not logger_files:
        return None
    return logger_files[-1]


def parse_logger_metrics(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            value = parse_float(right)
            if value is None:
                continue
            metrics[normalize_key(left)] = value
    return metrics


def env_with_case(rho0: float, rho_update_ratio: float, newton_tol: float, linear_tol: float) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_JOINT_AL_ENABLED"] = "1"
    env["STARK_EXP4_RUN_NAME"] = "exp4_coupled_joints_al_d1"
    env["STARK_JOINT_AL_ADAPTIVE_RHO"] = "1"
    env["STARK_JOINT_AL_RHO0"] = f"{rho0:.12g}"
    env["STARK_JOINT_AL_RHO_UPDATE_RATIO"] = f"{rho_update_ratio:.12g}"
    env["STARK_NEWTON_TOL"] = f"{newton_tol:.12g}"
    env["STARK_LINEAR_TOL"] = f"{linear_tol:.12g}"
    return env


def iter_cases(
    rho0_values: Iterable[float],
    rho_update_values: Iterable[float],
    newton_tol_values: Iterable[float],
    linear_tol_values: Iterable[float],
):
    for rho0, rho_update, newton_tol, linear_tol in itertools.product(
        rho0_values, rho_update_values, newton_tol_values, linear_tol_values
    ):
        yield {
            "rho0": float(rho0),
            "rho_update_ratio": float(rho_update),
            "newton_tol": float(newton_tol),
            "linear_tol": float(linear_tol),
        }


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "case_idx",
        "rho0",
        "rho_update_ratio",
        "newton_tol",
        "linear_tol",
        "logger_file",
        "total",
        "time_steps",
        "newton_iterations",
        "linear_iterations",
        "failed_step_count",
        "failed_step_time",
        "hardening_count",
        "joint_error_max_l2",
        "joint_error_max_deg",
        "joint_al_outer_iterations",
        "joint_al_outer_iterations_total",
        "joint_al_outer_limit_hits",
        "joint_rho_update_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_list(values: str) -> List[float]:
    out: List[float] = []
    for token in values.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise ValueError("Empty value list.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run D1 AL-IPC parameter sensitivity sweep.")
    parser.add_argument("--exe", type=Path, default=None, help="Path to examples executable.")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help=f"Output CSV (default: {DEFAULT_OUT_CSV}).")
    parser.add_argument("--rho0", type=str, default="1e2,1e3,1e4", help="Comma-separated rho0 values.")
    parser.add_argument("--rho-update", type=str, default="1.2,1.5,2.0", help="Comma-separated rho update ratio values.")
    parser.add_argument("--newton-tol", type=str, default="1e-3,1e-4,1e-5", help="Comma-separated Newton tolerance values.")
    parser.add_argument("--linear-tol", type=str, default="1.0,0.1,0.01", help="Comma-separated linear tolerance scale values.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of cases (0 means all).")
    parser.add_argument("--dry-run", action="store_true", help="Print cases without running simulations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe.resolve() if args.exe else None)

    rho0_values = parse_list(args.rho0)
    rho_update_values = parse_list(args.rho_update)
    newton_tol_values = parse_list(args.newton_tol)
    linear_tol_values = parse_list(args.linear_tol)

    all_cases = list(iter_cases(rho0_values, rho_update_values, newton_tol_values, linear_tol_values))
    if args.max_cases > 0:
        all_cases = all_cases[: args.max_cases]

    rows: List[Dict[str, object]] = []
    for case_idx, case in enumerate(all_cases, start=1):
        cmd = [str(exe), "exp4"]
        print(
            f"[d1] case={case_idx}/{len(all_cases)} "
            f"rho0={case['rho0']:.3g} rho_up={case['rho_update_ratio']:.3g} "
            f"newton_tol={case['newton_tol']:.3g} linear_tol={case['linear_tol']:.3g}"
        )

        if args.dry_run:
            continue

        env = env_with_case(
            rho0=case["rho0"],
            rho_update_ratio=case["rho_update_ratio"],
            newton_tol=case["newton_tol"],
            linear_tol=case["linear_tol"],
        )
        result = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"Case {case_idx} failed with exit code {result.returncode}")

        logger_path = latest_logger(EXP4_AL_DIR)
        if logger_path is None:
            raise FileNotFoundError(f"No logger found in {EXP4_AL_DIR}")
        metrics = parse_logger_metrics(logger_path)

        row: Dict[str, object] = dict(case)
        row["case_idx"] = case_idx
        row["logger_file"] = logger_path.name
        row["total"] = metrics.get("total")
        row["time_steps"] = metrics.get("time_steps")
        row["newton_iterations"] = metrics.get("newton_iterations")
        row["linear_iterations"] = metrics.get("linear_iterations")
        row["failed_step_count"] = metrics.get("failed_step_count")
        row["failed_step_time"] = metrics.get("failed_step_time")
        row["hardening_count"] = metrics.get("hardening_count")
        row["joint_error_max_l2"] = metrics.get("joint_error_max_l2")
        row["joint_error_max_deg"] = metrics.get("joint_error_max_deg")
        row["joint_al_outer_iterations"] = metrics.get("joint_al_outer_iterations")
        row["joint_al_outer_iterations_total"] = metrics.get("joint_al_outer_iterations_total")
        row["joint_al_outer_limit_hits"] = metrics.get("joint_al_outer_limit_hits")
        row["joint_rho_update_count"] = metrics.get("joint_rho_update_count")
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
