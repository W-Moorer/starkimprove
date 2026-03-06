#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import pychrono.core as chrono
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pychrono.core is not available. Run with:\n"
        "  conda run -n chrono-baseline python examples/paper_experiments/pychrono_crank_slider_benchmark.py --mode all"
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


def run_case(mode: str, dt: float, end_time: float, output_base: Path) -> Dict[str, str]:
    material = make_material(mode)
    system = configure_system(mode)

    link_width = 0.05
    link_thickness = 0.05
    crank_length = 0.35
    coupler_length = 0.85
    crank_geom_length = 0.90 * crank_length
    coupler_geom_length = 0.90 * coupler_length
    guide_y = 0.0
    support_y = 0.0
    slider_length = 0.24
    slider_height = 0.18
    stop_face_x = 1.28
    stop_thickness = 0.08
    crank_angle_deg = 60.0
    crank_angle_rad = crank_angle_deg * 3.141592653589793 / 180.0
    # Chrono's motor frame convention is opposite to the STARK hinge motor
    # used in exp2_crank_slider_impact, so we flip the sign to match slider motion.
    motor_w = 3.0
    density = 1000.0

    A = chrono.ChVector3d(0.0, support_y, 0.0)
    B = chrono.ChVector3d(
        crank_length * math.cos(crank_angle_rad),
        support_y + crank_length * math.sin(crank_angle_rad),
        0.0,
    )
    dy = guide_y - B.y
    dx = (coupler_length * coupler_length - dy * dy) ** 0.5
    pin_C = chrono.ChVector3d(B.x + dx, guide_y, 0.0)
    slider_center = chrono.ChVector3d(pin_C.x + 0.5 * slider_length, guide_y, 0.0)

    support = chrono.ChBody()
    support.SetFixed(True)
    support.SetPos(chrono.ChVector3d(0.0, support_y - 0.8, 0.0))
    system.Add(support)

    stop = chrono.ChBodyEasyBox(stop_thickness, 0.6, 0.12, density, True, True, material)
    stop.SetFixed(True)
    stop.SetPos(chrono.ChVector3d(stop_face_x + 0.5 * stop_thickness, guide_y, 0.0))
    system.Add(stop)

    crank = chrono.ChBodyEasyBox(crank_geom_length, link_width, link_thickness, density, True, True, material)
    crank.SetPos(chrono.ChVector3d(0.5 * (A.x + B.x), 0.5 * (A.y + B.y), 0.0))
    crank.SetRot(chrono.QuatFromAngleZ(crank_angle_rad))
    system.Add(crank)

    coupler_angle = math.atan2(pin_C.y - B.y, pin_C.x - B.x)
    coupler = chrono.ChBodyEasyBox(coupler_geom_length, link_width, link_thickness, density, True, True, material)
    coupler.SetPos(chrono.ChVector3d(0.5 * (B.x + pin_C.x), 0.5 * (B.y + pin_C.y), 0.0))
    coupler.SetRot(chrono.QuatFromAngleZ(coupler_angle))
    system.Add(coupler)

    slider = chrono.ChBodyEasyBox(slider_length, slider_height, 0.12, density, True, True, material)
    slider.SetPos(slider_center)
    system.Add(slider)

    motor = chrono.ChLinkMotorRotationSpeed()
    motor.Initialize(support, crank, chrono.ChFramed(A, chrono.QUNIT))
    motor.SetSpeedFunction(chrono.ChFunctionConst(motor_w))
    system.AddLink(motor)

    crank_coupler = chrono.ChLinkLockRevolute()
    crank_coupler.Initialize(crank, coupler, chrono.ChFramed(B, chrono.QUNIT))
    system.AddLink(crank_coupler)

    coupler_slider = chrono.ChLinkLockRevolute()
    coupler_slider.Initialize(coupler, slider, chrono.ChFramed(pin_C, chrono.QUNIT))
    system.AddLink(coupler_slider)

    slider_guide = chrono.ChLinkLockPrismatic()
    slider_guide.Initialize(support, slider, chrono.ChFramed(slider_center, chrono.QUNIT))
    system.AddLink(slider_guide)

    mode_tag = mode
    case_dir = output_base / f"pychrono_exp2_crank_slider_{mode_tag}"
    case_dir.mkdir(parents=True, exist_ok=True)
    state_path = case_dir / "crank_slider_state.csv"
    summary_path = case_dir / "summary.csv"

    n_steps = int(round(end_time / dt))
    solver_iters: List[int] = []
    min_gap = float("inf")
    last_x = 0.0
    last_vx = 0.0

    with state_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "slider_cx", "slider_cy", "slider_vx", "slider_vy", "gap_stop"])
        start = time.perf_counter()
        for i in range(n_steps + 1):
            t = system.GetChTime()
            x = slider.GetPos()
            v = slider.GetPosDt()
            gap = stop_face_x - (x.x + 0.5 * slider_length)
            min_gap = min(min_gap, gap)
            last_x = x.x
            last_vx = v.x
            writer.writerow(
                [
                    f"{t:.9f}",
                    f"{x.x:.9f}",
                    f"{x.y:.9f}",
                    f"{v.x:.9f}",
                    f"{v.y:.9f}",
                    f"{gap:.9f}",
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
        "mode": mode_tag,
        "wall_time_s": f"{wall_time:.6g}",
        "avg_solver_iterations": f"{avg_iters:.6g}",
        "min_gap": f"{min_gap:.9g}",
        "final_slider_x": f"{last_x:.9g}",
        "final_slider_vx": f"{last_vx:.9g}",
        "time_step_s": f"{dt:.6g}",
        "end_time_s": f"{end_time:.6g}",
    }
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyChrono crank-slider benchmark.")
    parser.add_argument("--mode", choices=["all", "nsc_ncp", "smc_penalty"], default="all")
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--end-time", type=float, default=0.4)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_base = args.output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    modes: Sequence[str] = ["nsc_ncp", "smc_penalty"] if args.mode == "all" else [args.mode]
    rows = [run_case(mode, args.dt, args.end_time, output_base) for mode in modes]
    out_csv = output_base / "pychrono_exp2_crank_slider_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
