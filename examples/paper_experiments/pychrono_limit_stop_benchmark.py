#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

from pychrono_bootstrap import configure_windows_dll_search

configure_windows_dll_search()

try:
    import pychrono.core as chrono
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pychrono.core is not available. Run from the chrono-baseline conda environment."
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"


def make_material(mode: str):
    if mode == "nsc_ncp":
        mat = chrono.ChContactMaterialNSC()
        mat.SetFriction(0.0)
        mat.SetRestitution(0.0)
        mat.SetCompliance(1e-9)
        mat.SetComplianceT(1e-9)
        return mat
    if mode == "smc_penalty":
        mat = chrono.ChContactMaterialSMC()
        mat.SetFriction(0.0)
        mat.SetRestitution(0.0)
        mat.SetYoungModulus(2e8)
        mat.SetPoissonRatio(0.3)
        mat.SetKn(5e6)
        mat.SetKt(2e6)
        mat.SetGn(1e3)
        mat.SetGt(1e3)
        return mat
    raise ValueError(f"Unsupported mode: {mode}")


def configure_system(mode: str):
    if mode == "nsc_ncp":
        system = chrono.ChSystemNSC()
        system.SetSolverType(chrono.ChSolver.Type_APGD)
        system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_PROJECTED)
    elif mode == "smc_penalty":
        system = chrono.ChSystemSMC()
        system.SetSolverType(chrono.ChSolver.Type_MINRES)
        system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, -9.81, 0.0))
    iterative = system.GetSolver().AsIterative()
    if iterative is not None:
        iterative.SetMaxIterations(200)
        iterative.SetTolerance(1e-8 if mode == "nsc_ncp" else 1e-10)
        iterative.EnableWarmStart(True)
    return system


def run_case(mode: str, dt: float, end_time: float, limit_deg: float, output_base: Path) -> Dict[str, str]:
    system = configure_system(mode)
    material = make_material(mode)

    rod_length = 1.0
    rod_width = 0.06
    rod_thickness = 0.06
    density = 1000.0

    support = chrono.ChBody()
    support.SetFixed(True)
    support.SetPos(chrono.ChVector3d(-0.06, 0.0, 0.0))
    system.Add(support)

    rod = chrono.ChBodyEasyBox(rod_length, rod_width, rod_thickness, density, True, True, material)
    rod.SetPos(chrono.ChVector3d(0.5 * rod_length, 0.0, 0.0))
    rod.SetRot(chrono.QuatFromAngleZ(0.0))
    system.Add(rod)

    hinge = chrono.ChLinkLockRevolute()
    hinge.Initialize(support, rod, chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, 0.0), chrono.QUNIT))
    limit = hinge.LimitRz()
    limit.SetActive(True)
    limit.SetRotation(True)
    limit.SetMin(-math.radians(limit_deg))
    limit.SetMax(math.radians(limit_deg))
    limit.SetMinCushion(0.0)
    limit.SetMaxCushion(0.0)
    system.AddLink(hinge)

    case_dir = output_base / f"pychrono_exp3_limit_stop_{mode}"
    case_dir.mkdir(parents=True, exist_ok=True)
    state_path = case_dir / "limit_stop_state.csv"
    summary_path = case_dir / "summary.csv"

    n_steps = int(round(end_time / dt))
    solver_iters: List[int] = []
    trigger_time = None
    peak_abs_theta_deg = 0.0
    peak_violation_deg = 0.0
    peak_overshoot_deg = 0.0
    peak_support_reaction_torque_z = 0.0
    peak_support_reaction_force_norm = 0.0
    final_theta_deg = 0.0
    final_omega_deg_s = 0.0

    with state_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "theta_deg",
                "omega_deg_s",
                "tip_x",
                "tip_y",
                "limit_violation_deg",
                "support_reaction_torque_z",
                "support_reaction_force_norm",
            ]
        )
        start = time.perf_counter()
        for i in range(n_steps + 1):
            t = system.GetChTime()
            axis_x = rod.TransformDirectionLocalToParent(chrono.ChVector3d(1.0, 0.0, 0.0))
            theta_deg = math.degrees(math.atan2(axis_x.y, axis_x.x))
            omega_deg_s = math.degrees(rod.GetAngVelParent().z)
            tip = rod.TransformPointLocalToParent(chrono.ChVector3d(0.5 * rod_length, 0.0, 0.0))
            violation_deg = max(abs(theta_deg) - limit_deg, 0.0)
            reaction = hinge.GetReaction2()
            reaction_torque_z = reaction.torque.z
            reaction_force_norm = math.sqrt(
                reaction.force.x * reaction.force.x
                + reaction.force.y * reaction.force.y
                + reaction.force.z * reaction.force.z
            )

            if violation_deg > 1e-9 and trigger_time is None:
                trigger_time = t
            peak_abs_theta_deg = max(peak_abs_theta_deg, abs(theta_deg))
            peak_violation_deg = max(peak_violation_deg, violation_deg)
            peak_overshoot_deg = max(peak_overshoot_deg, max(abs(theta_deg) - limit_deg, 0.0))
            peak_support_reaction_torque_z = max(peak_support_reaction_torque_z, abs(reaction_torque_z))
            peak_support_reaction_force_norm = max(peak_support_reaction_force_norm, reaction_force_norm)
            final_theta_deg = theta_deg
            final_omega_deg_s = omega_deg_s

            writer.writerow(
                [
                    f"{t:.9f}",
                    f"{theta_deg:.9f}",
                    f"{omega_deg_s:.9f}",
                    f"{tip.x:.9f}",
                    f"{tip.y:.9f}",
                    f"{violation_deg:.9f}",
                    f"{reaction_torque_z:.9f}",
                    f"{reaction_force_norm:.9f}",
                ]
            )
            if i < n_steps:
                system.DoStepDynamics(dt)
                iterative = system.GetSolver().AsIterative()
                iters = int(iterative.GetIterations()) if iterative is not None else 0
                solver_iters.append(iters)
        wall_time = time.perf_counter() - start

    avg_iters = (sum(solver_iters) / len(solver_iters)) if solver_iters else 0.0
    row = {
        "mode": mode,
        "wall_time_s": f"{wall_time:.6g}",
        "avg_solver_iterations": f"{avg_iters:.6g}",
        "trigger_time_s": "" if trigger_time is None else f"{trigger_time:.9g}",
        "peak_abs_theta_deg": f"{peak_abs_theta_deg:.9g}",
        "peak_limit_overshoot_deg": f"{peak_overshoot_deg:.9g}",
        "peak_limit_violation_deg": f"{peak_violation_deg:.9g}",
        "peak_support_reaction_torque_z": f"{peak_support_reaction_torque_z:.9g}",
        "peak_support_reaction_force_norm": f"{peak_support_reaction_force_norm:.9g}",
        "final_theta_deg": f"{final_theta_deg:.9g}",
        "final_omega_deg_s": f"{final_omega_deg_s:.9g}",
        "time_step_s": f"{dt:.6g}",
        "end_time_s": f"{end_time:.6g}",
    }
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyChrono limit-stop hinge benchmark.")
    parser.add_argument("--mode", choices=["all", "nsc_ncp", "smc_penalty"], default="nsc_ncp")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--limit-deg", type=float, default=35.0)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_base = args.output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    modes: Sequence[str] = ["nsc_ncp", "smc_penalty"] if args.mode == "all" else [args.mode]
    rows = [run_case(mode, args.dt, args.end_time, args.limit_deg, output_base) for mode in modes]
    out_csv = output_base / "pychrono_exp3_limit_stop_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
