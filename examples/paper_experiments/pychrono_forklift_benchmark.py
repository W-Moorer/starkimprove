#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    import pychrono.core as chrono
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pychrono.core is not available. Run with:\n"
        "  conda run -n chrono-baseline python examples/paper_experiments/pychrono_forklift_benchmark.py --mode all"
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"
DEFAULT_CHRONO_DATA_DIR = Path(r"E:\Anaconda\envs\chrono-baseline\Library\data")


def resolve_model_path(relative_path: str) -> Path:
    import os

    candidates: List[Path] = []
    override_models = os.environ.get("STARK_MODELS_DIR", "")
    override_chrono = os.environ.get("STARK_CHRONO_DATA_DIR", "")
    if override_models:
        candidates.append(Path(override_models) / relative_path)
    candidates.append(DEFAULT_MODELS_DIR / relative_path)
    if override_chrono:
        candidates.append(Path(override_chrono) / "models" / relative_path)
    candidates.append(DEFAULT_CHRONO_DATA_DIR / "models" / relative_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing model file: {relative_path}")


def parse_obj_bounds(path: Path) -> Tuple[chrono.ChVector3d, chrono.ChVector3d]:
    min_v = [float("inf")] * 3
    max_v = [float("-inf")] * 3
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            _, xs, ys, zs, *_ = line.split()
            vals = [float(xs), float(ys), float(zs)]
            for i, value in enumerate(vals):
                min_v[i] = min(min_v[i], value)
                max_v[i] = max(max_v[i], value)
    return chrono.ChVector3d(*min_v), chrono.ChVector3d(*max_v)


def make_material(
    mode: str,
    friction: float,
    nsc_compliance: float,
    smc_young: float,
    smc_poisson: float,
    smc_kn: float,
    smc_kt: float,
    smc_gn: float,
    smc_gt: float,
):
    if mode == "nsc_psor":
        mat = chrono.ChContactMaterialNSC()
        mat.SetFriction(friction)
        mat.SetRestitution(0.0)
        mat.SetCompliance(nsc_compliance)
        mat.SetComplianceT(nsc_compliance)
        return mat
    if mode == "smc_penalty":
        mat = chrono.ChContactMaterialSMC()
        mat.SetFriction(friction)
        mat.SetRestitution(0.0)
        mat.SetYoungModulus(smc_young)
        mat.SetPoissonRatio(smc_poisson)
        mat.SetKn(smc_kn)
        mat.SetKt(smc_kt)
        mat.SetGn(smc_gn)
        mat.SetGt(smc_gt)
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
    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, -9.81, 0.0))
    iterative = system.GetSolver().AsIterative()
    if iterative is not None:
        iterative.SetMaxIterations(solver_max_iters)
        iterative.SetTolerance(solver_tol)
        iterative.EnableWarmStart(True)
    return system


def add_box_collision(body, material, size_xyz: Tuple[float, float, float], center_xyz: Tuple[float, float, float]):
    shape = chrono.ChCollisionShapeBox(material, *size_xyz)
    body.AddCollisionShape(shape, chrono.ChFramed(chrono.ChVector3d(*center_xyz), chrono.QUNIT))


def run_case(
    mode: str,
    dt: float,
    end_time: float,
    output_base: Path,
    tag: str,
    solver_max_iters: int,
    solver_tol: float,
    lift_start: float,
    lift_end: float,
    lift_speed: float,
    pallet_y: float,
    pallet_z: float,
    friction: float,
    nsc_compliance: float,
    smc_young: float,
    smc_poisson: float,
    smc_kn: float,
    smc_kt: float,
    smc_gn: float,
    smc_gt: float,
) -> Dict[str, str]:
    material = make_material(mode, friction, nsc_compliance, smc_young, smc_poisson, smc_kn, smc_kt, smc_gn, smc_gt)
    system = configure_system(mode, solver_max_iters, solver_tol)

    COG_arm = chrono.ChVector3d(0.0, 1.300, 1.855)
    COG_fork = chrono.ChVector3d(0.0, 0.362, 2.100)
    POS_prismatic = chrono.ChVector3d(0.0, 0.150, 1.855)
    pallet_pos = chrono.ChVector3d(0.0, pallet_y, pallet_z)
    fork_tine_top_local = -0.296

    ground = chrono.ChBodyEasyBox(40.0, 2.0, 40.0, 1000.0, True, True, material)
    ground.SetFixed(True)
    ground.SetPos(chrono.ChVector3d(0.0, -1.0, 0.0))
    system.Add(ground)

    arm = chrono.ChBody()
    arm.SetFixed(True)
    arm.SetPos(COG_arm)
    arm.SetMass(100.0)
    arm.SetInertiaXX(chrono.ChVector3d(30.0, 30.0, 30.0))
    system.Add(arm)

    fork = chrono.ChBody()
    fork.SetPos(COG_fork)
    fork.SetMass(60.0)
    fork.SetInertiaXX(chrono.ChVector3d(15.0, 15.0, 15.0))
    add_box_collision(fork, material, (0.100, 0.032, 1.033), (-0.352, -0.312, 0.613))
    add_box_collision(fork, material, (0.100, 0.032, 1.033), (0.352, -0.312, 0.613))
    add_box_collision(fork, material, (0.344, 1.134, 0.101), (0.0, 0.321, -0.009))
    fork.EnableCollision(True)
    system.Add(fork)

    pallet_path = resolve_model_path("pallet.obj")
    pallet = chrono.ChBodyEasyMesh(str(pallet_path), 300.0, True, True, True, material, 0.001)
    pallet.SetPos(pallet_pos)
    system.Add(pallet)
    pallet_min, _ = parse_obj_bounds(pallet_path)

    lift_prismatic = chrono.ChLinkLockPrismatic()
    lift_prismatic.Initialize(
        fork,
        arm,
        chrono.ChFramed(POS_prismatic, chrono.QuatFromAngleX(chrono.CH_PI / 2.0)),
    )
    system.AddLink(lift_prismatic)

    lift_motor = chrono.ChLinkMotorLinearSpeed()
    lift_motor.Initialize(
        fork,
        arm,
        chrono.ChFramed(POS_prismatic, chrono.QuatFromAngleX(chrono.CH_PI / 2.0)),
    )
    lift_motor.SetGuideConstraint(chrono.ChLinkMotorLinear.GuideConstraint_PRISMATIC)
    speed_fun = chrono.ChFunctionConst(0.0)
    lift_motor.SetSpeedFunction(speed_fun)
    system.AddLink(lift_motor)

    mode_tag = mode if not tag else f"{mode}_{tag}"
    case_dir = output_base / f"pychrono_exp7_forklift_{mode_tag}"
    case_dir.mkdir(parents=True, exist_ok=True)
    state_path = case_dir / "forklift_state.csv"
    summary_path = case_dir / "summary.csv"

    n_steps = int(round(end_time / dt))
    solver_iters: List[int] = []
    last_fork_y = 0.0
    last_pallet_y = 0.0
    last_pallet_z = 0.0
    prev_sample_t: float | None = None
    prev_fork_x: chrono.ChVector3d | None = None
    prev_pallet_x: chrono.ChVector3d | None = None

    with state_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "t", "target_lift_v",
            "fork_cx", "fork_cy", "fork_cz", "fork_vx", "fork_vy", "fork_vz",
            "pallet_cx", "pallet_cy", "pallet_cz", "pallet_vx", "pallet_vy", "pallet_vz",
            "fork_top_y", "pallet_bottom_y", "vertical_gap", "actuator_force_proxy"
        ])
        start = time.perf_counter()
        for i in range(n_steps + 1):
            t = system.GetChTime()
            target_v = lift_speed if (t >= lift_start and t <= lift_end) else 0.0
            speed_fun.SetConstant(target_v)

            fork_frame = fork.GetFrameCOMToAbs()
            pallet_frame = pallet.GetFrameCOMToAbs()
            fork_x = fork_frame.GetPos()
            pallet_x = pallet_frame.GetPos()
            if prev_sample_t is None or prev_fork_x is None or prev_pallet_x is None or t <= prev_sample_t:
                fork_v = chrono.ChVector3d(0.0, 0.0, 0.0)
                pallet_v = chrono.ChVector3d(0.0, 0.0, 0.0)
            else:
                inv_dt = 1.0 / (t - prev_sample_t)
                fork_v = (fork_x - prev_fork_x) * inv_dt
                pallet_v = (pallet_x - prev_pallet_x) * inv_dt
            fork_ref_y = fork.GetPos().y
            pallet_ref_y = pallet.GetPos().y
            fork_top_y = fork_ref_y + fork_tine_top_local
            pallet_bottom_y = pallet_ref_y + pallet_min.y
            vertical_gap = pallet_bottom_y - fork_top_y
            actuator_force = lift_motor.GetMotorForce()
            writer.writerow([
                f"{t:.9f}",
                f"{target_v:.9f}",
                f"{fork_x.x:.9f}", f"{fork_x.y:.9f}", f"{fork_x.z:.9f}",
                f"{fork_v.x:.9f}", f"{fork_v.y:.9f}", f"{fork_v.z:.9f}",
                f"{pallet_x.x:.9f}", f"{pallet_x.y:.9f}", f"{pallet_x.z:.9f}",
                f"{pallet_v.x:.9f}", f"{pallet_v.y:.9f}", f"{pallet_v.z:.9f}",
                f"{fork_top_y:.9f}",
                f"{pallet_bottom_y:.9f}",
                f"{vertical_gap:.9f}",
                f"{actuator_force:.9f}",
            ])
            last_fork_y = fork_x.y
            last_pallet_y = pallet_x.y
            last_pallet_z = pallet_x.z
            prev_sample_t = t
            prev_fork_x = chrono.ChVector3d(fork_x.x, fork_x.y, fork_x.z)
            prev_pallet_x = chrono.ChVector3d(pallet_x.x, pallet_x.y, pallet_x.z)
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
        "final_fork_y": f"{last_fork_y:.9g}",
        "final_pallet_y": f"{last_pallet_y:.9g}",
        "final_pallet_z": f"{last_pallet_z:.9g}",
        "time_step_s": f"{dt:.6g}",
        "end_time_s": f"{end_time:.6g}",
        "lift_start_s": f"{lift_start:.6g}",
        "lift_end_s": f"{lift_end:.6g}",
        "lift_speed": f"{lift_speed:.9g}",
        "pallet_y": f"{pallet_y:.9g}",
        "pallet_z": f"{pallet_z:.9g}",
        "friction": f"{friction:.9g}",
        "solver_max_iters": str(solver_max_iters),
        "solver_tol": f"{solver_tol:.9g}",
        "nsc_compliance": f"{nsc_compliance:.9g}",
        "smc_young": f"{smc_young:.9g}",
        "smc_poisson": f"{smc_poisson:.9g}",
        "smc_kn": f"{smc_kn:.9g}",
        "smc_kt": f"{smc_kt:.9g}",
        "smc_gn": f"{smc_gn:.9g}",
        "smc_gt": f"{smc_gt:.9g}",
    }
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyChrono fixed-base forklift lift benchmark.")
    parser.add_argument("--mode", choices=["all", "nsc_psor", "smc_penalty"], default="all")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--end-time", type=float, default=1.2)
    parser.add_argument("--lift-start", type=float, default=0.3)
    parser.add_argument("--lift-end", type=float, default=1.0)
    parser.add_argument("--lift-speed", type=float, default=-0.10)
    parser.add_argument("--pallet-y", type=float, default=0.15)
    parser.add_argument("--pallet-z", type=float, default=3.02)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--solver-max-iters", type=int, default=800)
    parser.add_argument("--solver-tol", type=float, default=1e-10)
    parser.add_argument("--nsc-compliance", type=float, default=1e-3)
    parser.add_argument("--smc-young", type=float, default=2e8)
    parser.add_argument("--smc-poisson", type=float, default=0.3)
    parser.add_argument("--smc-kn", type=float, default=5e6)
    parser.add_argument("--smc-kt", type=float, default=2e6)
    parser.add_argument("--smc-gn", type=float, default=1e3)
    parser.add_argument("--smc-gt", type=float, default=1e3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_base = args.output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)
    modes: Sequence[str] = ["nsc_psor", "smc_penalty"] if args.mode == "all" else [args.mode]
    rows = []
    for mode in modes:
        rows.append(
            run_case(
                mode=mode,
                dt=args.dt,
                end_time=args.end_time,
                output_base=output_base,
                tag=args.tag,
                solver_max_iters=args.solver_max_iters,
                solver_tol=args.solver_tol,
                lift_start=args.lift_start,
                lift_end=args.lift_end,
                lift_speed=args.lift_speed,
                pallet_y=args.pallet_y,
                pallet_z=args.pallet_z,
                friction=args.friction,
                nsc_compliance=args.nsc_compliance,
                smc_young=args.smc_young,
                smc_poisson=args.smc_poisson,
                smc_kn=args.smc_kn,
                smc_kt=args.smc_kt,
                smc_gn=args.smc_gn,
                smc_gt=args.smc_gt,
            )
        )
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
