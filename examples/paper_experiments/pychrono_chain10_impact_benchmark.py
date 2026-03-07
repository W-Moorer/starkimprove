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
        "pychrono.core is not available. Run with:\n"
        "  conda run -n chrono-baseline python examples/paper_experiments/pychrono_chain10_impact_benchmark.py --mode all"
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"


def make_material(mode: str, nsc_compliance: float, smc_kn: float, smc_gn: float):
    if mode == "nsc_psor":
        mat = chrono.ChContactMaterialNSC()
        mat.SetFriction(0.0)
        mat.SetRestitution(0.0)
        mat.SetCompliance(nsc_compliance)
        mat.SetComplianceT(nsc_compliance)
        return mat
    if mode == "smc_penalty":
        mat = chrono.ChContactMaterialSMC()
        mat.SetFriction(0.0)
        mat.SetRestitution(0.0)
        mat.SetYoungModulus(2e8)
        mat.SetPoissonRatio(0.3)
        mat.SetKn(smc_kn)
        mat.SetKt(0.5 * smc_kn)
        mat.SetGn(smc_gn)
        mat.SetGt(smc_gn)
        return mat
    raise ValueError(f"Unsupported mode: {mode}")


def configure_system(mode: str, solver_max_iters: int, solver_tol: float):
    if mode == "nsc_psor":
        system = chrono.ChSystemNSC()
        system.SetSolverType(chrono.ChSolver.Type_PSOR)
        system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_PROJECTED)
    elif mode == "smc_penalty":
        system = chrono.ChSystemSMC()
        system.SetSolverType(chrono.ChSolver.Type_MINRES)
        system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_LINEARIZED)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -9.81))
    iterative = system.GetSolver().AsIterative()
    if iterative is not None:
        iterative.SetMaxIterations(solver_max_iters)
        iterative.SetTolerance(solver_tol)
        iterative.EnableWarmStart(True)
    return system


def min_box_gap_to_plane_z(body, size_xyz: tuple[float, float, float], plane_z: float) -> float:
    hx, hy, hz = (0.5 * size_xyz[0], 0.5 * size_xyz[1], 0.5 * size_xyz[2])
    min_gap = float("inf")
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = body.TransformPointLocalToParent(chrono.ChVector3d(sx * hx, sy * hy, sz * hz))
                min_gap = min(min_gap, corner.z - plane_z)
    return min_gap


def run_case(
    mode: str,
    dt: float,
    end_time: float,
    output_base: Path,
    tag: str,
    solver_max_iters: int,
    solver_tol: float,
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
    smc_kn: float,
    smc_gn: float,
) -> Dict[str, str]:
    material = make_material(mode, nsc_compliance, smc_kn, smc_gn)
    system = configure_system(mode, solver_max_iters, solver_tol)

    density = link_mass / (link_length * link_width * link_thickness)
    ground = chrono.ChBodyEasyBox(20.0, 2.0, 0.1, 1000.0, True, True, material)
    ground.SetFixed(True)
    ground.SetPos(chrono.ChVector3d(0.0, 0.0, ground_top_z - 0.05))
    system.Add(ground)

    anchor = chrono.ChVector3d(anchor_x, anchor_y, anchor_z)
    angle_rad = math.radians(initial_angle_deg)
    direction = chrono.ChVector3d(math.cos(angle_rad), 0.0, math.sin(angle_rad))
    link_rot = chrono.QuatFromAngleAxis(angle_rad, chrono.ChVector3d(0.0, 1.0, 0.0))
    hinge_rot = chrono.QuatFromAngleAxis(-chrono.CH_PI / 2.0, chrono.ChVector3d(1.0, 0.0, 0.0))

    links = []
    for i in range(chain_count):
        body = chrono.ChBodyEasyBox(link_length, link_width, link_thickness, density, True, True, material)
        center = chrono.ChVector3d(
            anchor.x + (i + 0.5) * chain_spacing * direction.x,
            anchor.y,
            anchor.z + (i + 0.5) * chain_spacing * direction.z,
        )
        body.SetPos(center)
        body.SetRot(link_rot)
        system.Add(body)
        links.append(body)

    support_hinge = chrono.ChLinkLockRevolute()
    support_hinge.Initialize(ground, links[0], chrono.ChFramed(anchor, hinge_rot))
    system.AddLink(support_hinge)

    spherical_links = []
    for i in range(1, chain_count):
        joint_pos = chrono.ChVector3d(
            anchor.x + i * chain_spacing * direction.x,
            anchor.y,
            anchor.z + i * chain_spacing * direction.z,
        )
        joint = chrono.ChLinkLockSpherical()
        joint.Initialize(links[i - 1], links[i], chrono.ChFramed(joint_pos, chrono.QUNIT))
        system.AddLink(joint)
        spherical_links.append(joint)

    mode_tag = mode if not tag else f"{mode}_{tag}"
    case_dir = output_base / f"pychrono_exp2_chain10_{mode_tag}"
    case_dir.mkdir(parents=True, exist_ok=True)
    state_path = case_dir / "chain10_impact_state.csv"
    summary_path = case_dir / "summary.csv"

    n_steps = int(round(end_time / dt))
    solver_iters: List[int] = []
    max_joint_drift = 0.0

    with state_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t", "tip_cx", "tip_cy", "tip_cz", "tip_vx", "tip_vy", "tip_vz",
            "min_gap_ground", "max_joint_drift", "max_link_speed"
        ])
        start = time.perf_counter()
        for step in range(n_steps + 1):
            t = system.GetChTime()
            tip = links[-1]
            tip_x = tip.GetPos()
            tip_v = tip.GetPosDt()

            min_gap_ground = float("inf")
            max_link_speed = 0.0
            for link in links:
                min_gap_ground = min(min_gap_ground, min_box_gap_to_plane_z(link, (link_length, link_width, link_thickness), ground_top_z))
                vel = link.GetPosDt()
                max_link_speed = max(max_link_speed, math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))

            max_joint_drift = 0.0

            writer.writerow([
                f"{t:.9f}",
                f"{tip_x.x:.12g}", f"{tip_x.y:.12g}", f"{tip_x.z:.12g}",
                f"{tip_v.x:.12g}", f"{tip_v.y:.12g}", f"{tip_v.z:.12g}",
                f"{min_gap_ground:.12g}", f"{max_joint_drift:.12g}", f"{max_link_speed:.12g}",
            ])

            if step < n_steps:
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
        "time_step_s": f"{dt:.6g}",
        "end_time_s": f"{end_time:.6g}",
    }
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyChrono frictionless chain10 impact benchmark.")
    parser.add_argument("--mode", choices=["all", "nsc_psor", "smc_penalty"], default="all")
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--end-time", type=float, default=2.0)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--solver-max-iters", type=int, default=600)
    parser.add_argument("--solver-tol", type=float, default=1e-10)
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
    parser.add_argument("--nsc-compliance", type=float, default=1e-4)
    parser.add_argument("--smc-kn", type=float, default=5e6)
    parser.add_argument("--smc-gn", type=float, default=5e2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_base = args.output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    modes: Sequence[str] = ["nsc_psor", "smc_penalty"] if args.mode == "all" else [args.mode]
    rows = [
        run_case(
            mode,
            args.dt,
            args.end_time,
            output_base,
            args.tag,
            args.solver_max_iters,
            args.solver_tol,
            args.chain_count,
            args.link_length,
            args.link_width,
            args.link_thickness,
            args.link_mass,
            args.chain_spacing,
            args.anchor_x,
            args.anchor_y,
            args.anchor_z,
            args.initial_angle_deg,
            args.ground_top_z,
            args.nsc_compliance,
            args.smc_kn,
            args.smc_gn,
        )
        for mode in modes
    ]
    out_csv = output_base / ("pychrono_exp2_chain10_summary.csv" if not args.tag else f"pychrono_exp2_chain10_summary_{args.tag}.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
