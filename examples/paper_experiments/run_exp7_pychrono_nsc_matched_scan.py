#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import pandas as pd

from run_exp7_forklift_benchmark import (
    OUTPUT_BASE,
    error_against_reference,
    load_state,
    resolve_executable,
    run_pychrono_case,
    run_stark_case,
)


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
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


def parse_grid(values: str) -> List[float]:
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def state_metrics(path: Path, lift_start: float) -> Dict[str, float]:
    df = load_state(path)
    pre = df[df["t"] <= lift_start + 1e-12]
    return {
        "pre_dy": float(pre["pallet_cy"].iloc[-1] - pre["pallet_cy"].iloc[0]),
        "pre_dz": float(pre["pallet_cz"].iloc[-1] - pre["pallet_cz"].iloc[0]),
        "end_dy": float(df["pallet_cy"].iloc[-1] - df["pallet_cy"].iloc[0]),
        "end_dz": float(df["pallet_cz"].iloc[-1] - df["pallet_cz"].iloc[0]),
        "min_gap": float(df["vertical_gap"].min()),
        "max_gap": float(df["vertical_gap"].max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matched-error scan for PyChrono NSC on exp7 forklift benchmark.")
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
    parser.add_argument("--fork-pallet-friction-grid", type=str, default="0.0,0.05,0.1,0.15,0.2")
    parser.add_argument("--nsc-compliance-grid", type=str, default="5e-4,1e-3,2e-3,5e-3")
    parser.add_argument("--solver-max-iters-grid", type=str, default="800")
    parser.add_argument("--solver-tol", type=float, default=1e-10)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "exp7_pychrono_nsc_fixedpose_scan.csv")
    parser.add_argument("--best-csv", type=Path, default=OUTPUT_BASE / "exp7_pychrono_nsc_fixedpose_best.csv")
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

    rows: List[Dict[str, object]] = []
    mu_grid = parse_grid(args.fork_pallet_friction_grid)
    compliance_grid = parse_grid(args.nsc_compliance_grid)
    solver_iters_grid = [int(x) for x in parse_grid(args.solver_max_iters_grid)]

    for mu in mu_grid:
        for compliance in compliance_grid:
            for solver_max_iters in solver_iters_grid:
                tag = f"mu{mu:.2f}_c{compliance:.0e}_it{solver_max_iters}".replace(".", "p").replace("+", "")
                row = run_pychrono_case(
                    "nsc_psor",
                    args.dt,
                    args.end_time,
                    args.lift_start,
                    args.lift_end,
                    args.lift_speed,
                    args.pallet_y,
                    args.pallet_z,
                    mu,
                    compliance,
                    solver_max_iters,
                    args.solver_tol,
                    tag,
                    args.force_run,
                )
                row.update(error_against_reference(ref_state, load_state(Path(row["state_csv"]))))
                row.update(state_metrics(Path(row["state_csv"]), args.lift_start))
                rows.append(row)

    rows.sort(key=lambda r: float(r["composite_error"]))
    write_csv(rows, args.out_csv)

    best = rows[0] if rows else None
    if best is not None:
        write_csv([best], args.best_csv)
        print(pd.DataFrame([best]).to_string(index=False))
    else:
        print("No valid PyChrono NSC cases found.")

    print(args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
