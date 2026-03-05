#!/usr/bin/env python3
"""
PyChrono double-pendulum benchmark aligned with STARK exp6.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

try:
    import pychrono.core as chrono
except ImportError as exc:
    raise SystemExit(
        "pychrono.core is not available in current Python env.\n"
        "Run with: conda run -n chrono-baseline python examples/paper_experiments/pychrono_double_pendulum.py"
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"


def configure_system() -> chrono.ChSystemNSC:
    system = chrono.ChSystemNSC()
    system.SetSolverType(chrono.ChSolver.Type_MINRES)
    system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, -9.81, 0.0))

    iterative = system.GetSolver().AsIterative()
    if iterative is not None:
        iterative.SetMaxIterations(600)
        iterative.SetTolerance(1e-12)
        iterative.EnableWarmStart(True)
    return system


def run_double_pendulum(dt: float, end_time: float, output_base: Path):
    output_dir = output_base / "exp6_double_pendulum_pychrono"
    output_dir.mkdir(parents=True, exist_ok=True)

    state_csv = output_dir / "double_pendulum_state.csv"
    reaction_csv = output_dir / "support_reaction.csv"
    summary_csv = output_dir / "summary.csv"

    system = configure_system()

    L = 1.0
    W = 0.05
    link_mass = 1.0
    density = link_mass / (L * W * W)
    total_mass = 2.0 * link_mass
    gravity = chrono.ChVector3d(0.0, -9.81, 0.0)

    mat = chrono.ChContactMaterialNSC()
    mat.SetFriction(0.0)
    mat.SetRestitution(0.0)

    ground = chrono.ChBody()
    ground.SetFixed(True)
    ground.SetPos(chrono.ChVector3d(-0.05, 0.0, 0.0))
    system.Add(ground)

    rod1 = chrono.ChBodyEasyBox(L, W, W, density, True, False, mat)
    rod1.SetPos(chrono.ChVector3d(0.5 * L, 0.0, 0.0))
    system.Add(rod1)

    rod2 = chrono.ChBodyEasyBox(L, W, W, density, True, False, mat)
    rod2.SetPos(chrono.ChVector3d(1.5 * L, 0.0, 0.0))
    system.Add(rod2)

    support_hinge = chrono.ChLinkLockRevolute()
    support_hinge.Initialize(ground, rod1, chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, 0.0), chrono.QUNIT))
    system.AddLink(support_hinge)

    mid_hinge = chrono.ChLinkLockRevolute()
    mid_hinge.Initialize(rod1, rod2, chrono.ChFramed(chrono.ChVector3d(L, 0.0, 0.0), chrono.QUNIT))
    system.AddLink(mid_hinge)

    n_steps = int(round(end_time / dt))

    prev_vcm = chrono.ChVector3d(0.0, 0.0, 0.0)
    prev_t = -1.0

    start = time.perf_counter()
    with state_csv.open("w", newline="", encoding="utf-8") as f_state, reaction_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f_react:
        w_state = csv.writer(f_state)
        w_react = csv.writer(f_react)

        w_state.writerow(
            [
                "t",
                "rod1_x",
                "rod1_y",
                "rod1_z",
                "rod1_vx",
                "rod1_vy",
                "rod1_vz",
                "rod2_x",
                "rod2_y",
                "rod2_z",
                "rod2_vx",
                "rod2_vy",
                "rod2_vz",
                "vcm_x",
                "vcm_y",
                "vcm_z",
            ]
        )
        w_react.writerow(
            [
                "t",
                "fx_joint",
                "fy_joint",
                "fz_joint",
                "f_joint_norm",
                "fx_est",
                "fy_est",
                "fz_est",
                "f_est_norm",
            ]
        )

        for step in range(n_steps + 1):
            t = system.GetChTime()
            p1 = rod1.GetPos()
            p2 = rod2.GetPos()
            v1 = rod1.GetPosDt()
            v2 = rod2.GetPosDt()

            vcm = chrono.ChVector3d(
                (rod1.GetMass() * v1.x + rod2.GetMass() * v2.x) / total_mass,
                (rod1.GetMass() * v1.y + rod2.GetMass() * v2.y) / total_mass,
                (rod1.GetMass() * v1.z + rod2.GetMass() * v2.z) / total_mass,
            )

            fx_est = 0.0
            fy_est = 0.0
            fz_est = 0.0
            if prev_t >= 0.0:
                dtt = t - prev_t
                if dtt > 1e-12:
                    acm_x = (vcm.x - prev_vcm.x) / dtt
                    acm_y = (vcm.y - prev_vcm.y) / dtt
                    acm_z = (vcm.z - prev_vcm.z) / dtt
                    fx_est = total_mass * (acm_x - gravity.x)
                    fy_est = total_mass * (acm_y - gravity.y)
                    fz_est = total_mass * (acm_z - gravity.z)

            react = support_hinge.GetReaction1()
            fx_joint = react.force.x
            fy_joint = react.force.y
            fz_joint = react.force.z

            w_state.writerow(
                [
                    f"{t:.9f}",
                    f"{p1.x:.12g}",
                    f"{p1.y:.12g}",
                    f"{p1.z:.12g}",
                    f"{v1.x:.12g}",
                    f"{v1.y:.12g}",
                    f"{v1.z:.12g}",
                    f"{p2.x:.12g}",
                    f"{p2.y:.12g}",
                    f"{p2.z:.12g}",
                    f"{v2.x:.12g}",
                    f"{v2.y:.12g}",
                    f"{v2.z:.12g}",
                    f"{vcm.x:.12g}",
                    f"{vcm.y:.12g}",
                    f"{vcm.z:.12g}",
                ]
            )
            w_react.writerow(
                [
                    f"{t:.9f}",
                    f"{fx_joint:.12g}",
                    f"{fy_joint:.12g}",
                    f"{fz_joint:.12g}",
                    f"{(fx_joint ** 2 + fy_joint ** 2 + fz_joint ** 2) ** 0.5:.12g}",
                    f"{fx_est:.12g}",
                    f"{fy_est:.12g}",
                    f"{fz_est:.12g}",
                    f"{(fx_est ** 2 + fy_est ** 2 + fz_est ** 2) ** 0.5:.12g}",
                ]
            )

            prev_vcm = chrono.ChVector3d(vcm.x, vcm.y, vcm.z)
            prev_t = t

            if step < n_steps:
                system.DoStepDynamics(dt)

    wall = time.perf_counter() - start
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dt", "end_time", "steps", "wall_time_s"])
        writer.writerow([f"{dt:.12g}", f"{end_time:.12g}", str(n_steps + 1), f"{wall:.6f}"])

    print(f"Wrote: {state_csv}")
    print(f"Wrote: {reaction_csv}")
    print(f"Wrote: {summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyChrono double pendulum benchmark.")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step size.")
    parser.add_argument("--end-time", type=float, default=2.0, help="Simulation end time.")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_double_pendulum(args.dt, args.end_time, args.output_base.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
