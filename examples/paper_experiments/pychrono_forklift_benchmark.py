#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pychrono_bootstrap import configure_windows_dll_search, resolve_chrono_data_dir

configure_windows_dll_search()

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
DEFAULT_CHRONO_DATA_DIR = resolve_chrono_data_dir()


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


def load_obj_vertices(path: Path) -> List[Tuple[float, float, float]]:
    vertices: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            _, xs, ys, zs, *_ = line.split()
            vertices.append((float(xs), float(ys), float(zs)))
    return vertices


def write_localized_obj_copy(src: Path, dst: Path, reference_xyz: Tuple[float, float, float]) -> None:
    rx, ry, rz = reference_xyz
    with src.open("r", encoding="utf-8", errors="ignore") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if line.startswith("v "):
                _, xs, ys, zs, *rest = line.split()
                x = float(xs) - rx
                y = float(ys) - ry
                z = float(zs) - rz
                suffix = (" " + " ".join(rest)) if rest else ""
                fout.write(f"v {x:.9f} {y:.9f} {z:.9f}{suffix}\n")
            else:
                fout.write(line)


def estimate_fork_tine_top_y_local(vertices: Sequence[Tuple[float, float, float]]) -> float:
    candidates = [
        y
        for x, y, z in vertices
        if 0.20 <= abs(x) <= 0.50 and y <= 0.0 and z >= 0.10
    ]
    if not candidates:
        candidates = [y for _, y, _ in vertices if y <= 0.0]
    return max(candidates) if candidates else 0.0


def demo_box_fork_tine_top_y_local() -> float:
    return -0.312 + 0.032


def box_inertia_diag(mass: float, size_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    sx, sy, sz = size_xyz
    ixx = mass * (sy * sy + sz * sz) / 12.0
    iyy = mass * (sx * sx + sz * sz) / 12.0
    izz = mass * (sx * sx + sy * sy) / 12.0
    return ixx, iyy, izz


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
    # Keep benchmark dimensions in full lengths; Chrono expects half-lengths here.
    half_xyz = tuple(0.5 * value for value in size_xyz)
    shape = chrono.ChCollisionShapeBox(material, *half_xyz)
    body.AddCollisionShape(shape, chrono.ChFramed(chrono.ChVector3d(*center_xyz), chrono.QUNIT))


def run_case(
    mode: str,
    scenario: str,
    dt: float,
    end_time: float,
    output_base: Path,
    tag: str,
    solver_max_iters: int,
    solver_tol: float,
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
    smc_young: float,
    smc_poisson: float,
    smc_kn: float,
    smc_kt: float,
    smc_gn: float,
    smc_gt: float,
    velocity_servo_gain: float | None,
) -> Dict[str, str]:
    demo_faithful = scenario == "demo_faithful"
    material = make_material(mode, friction, nsc_compliance, smc_young, smc_poisson, smc_kn, smc_kt, smc_gn, smc_gt)
    system = configure_system(mode, solver_max_iters, solver_tol)

    COG_arm = chrono.ChVector3d(0.0, 1.300, 1.855)
    COG_fork = chrono.ChVector3d(0.0, 0.362, 2.100)
    POS_prismatic = chrono.ChVector3d(0.0, 0.150, 1.855)
    pallet_pos = chrono.ChVector3d(0.0, pallet_y, pallet_z)
    fork_mass = 60.0
    fork_ixx = 15.0
    fork_iyy = 15.0
    fork_izz = 15.0

    scenario_tag = "" if scenario == "benchmark" else f"_{scenario}"
    mode_tag = f"{mode}{scenario_tag}" if not tag else f"{mode}{scenario_tag}_{tag}"
    case_dir = output_base / f"pychrono_exp7_forklift_{mode_tag}"
    case_dir.mkdir(parents=True, exist_ok=True)
    state_path = case_dir / "forklift_state.csv"
    summary_path = case_dir / "summary.csv"

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

    if demo_faithful:
        fork = chrono.ChBody()
        fork.SetPos(COG_fork)
        fork.SetMass(fork_mass)
        fork.SetInertiaXX(chrono.ChVector3d(fork_ixx, fork_iyy, fork_izz))
        fshape1 = chrono.ChCollisionShapeBox(material, 0.100, 0.032, 1.033)
        fshape2 = chrono.ChCollisionShapeBox(material, 0.100, 0.032, 1.033)
        fshape3 = chrono.ChCollisionShapeBox(material, 0.344, 1.134, 0.101)
        fork.AddCollisionShape(fshape1, chrono.ChFramed(chrono.ChVector3d(-0.352, -0.312, 0.613), chrono.QUNIT))
        fork.AddCollisionShape(fshape2, chrono.ChFramed(chrono.ChVector3d(0.352, -0.312, 0.613), chrono.QUNIT))
        fork.AddCollisionShape(fshape3, chrono.ChFramed(chrono.ChVector3d(0.000, 0.321, -0.009), chrono.QUNIT))
        fork.EnableCollision(True)
        system.Add(fork)
        fork_tine_top_local = demo_box_fork_tine_top_y_local()
    else:
        fork_path = resolve_model_path("forklift/forks.obj")
        localized_fork_path = case_dir / "forks_localized.obj"
        write_localized_obj_copy(fork_path, localized_fork_path, (COG_fork.x, COG_fork.y, COG_fork.z))
        fork = chrono.ChBodyEasyMesh(str(localized_fork_path), 300.0, True, True, True, material, 0.001)
        fork.SetPos(COG_fork)
        fork.SetMass(fork_mass)
        fork.SetInertiaXX(chrono.ChVector3d(fork_ixx, fork_iyy, fork_izz))
        system.Add(fork)
        fork_vertices = load_obj_vertices(localized_fork_path)
        fork_tine_top_local = estimate_fork_tine_top_y_local(fork_vertices)

    pallet_path = resolve_model_path("pallet.obj")
    pallet = chrono.ChBodyEasyMesh(str(pallet_path), 300.0, True, True, True, material, 0.001)
    pallet.SetPos(pallet_pos)
    pallet.SetMass(pallet_mass)
    pallet.SetInertiaXX(chrono.ChVector3d(pallet_ixx, pallet_iyy, pallet_izz))
    system.Add(pallet)
    pallet_min, _ = parse_obj_bounds(pallet_path)
    initial_effective_gap = (pallet_pos.y + pallet_min.y) - (COG_fork.y + fork_tine_top_local)

    actuator_marker_offset = 0.01
    actuator_initial_length = actuator_marker_offset
    lift_motor = None
    actuator_fun = None
    servo_gain = 0.0
    if scenario == "benchmark":
        fork.SetFixed(True)
    else:
        lift_prismatic = chrono.ChLinkLockPrismatic()
        lift_prismatic.Initialize(
            fork,
            arm,
            chrono.ChFramed(POS_prismatic, chrono.QuatFromAngleX(chrono.CH_PI / 2.0)),
        )
        system.AddLink(lift_prismatic)

        if not hasattr(chrono, "ChLinkLockLinActuator"):
            raise RuntimeError("PyChrono build does not expose ChLinkLockLinActuator; cannot align exp7 actuator semantics.")
        lift_motor = chrono.ChLinkLockLinActuator()
        lift_motor.Initialize(
            fork,
            arm,
            False,
            chrono.ChFramed(POS_prismatic + chrono.ChVector3d(0.0, actuator_marker_offset, 0.0), chrono.QUNIT),
            chrono.ChFramed(POS_prismatic, chrono.QUNIT),
        )
        # ChLinkLockLinActuator expects the imposed displacement relative to the
        # reference configuration, not the absolute marker distance.
        lift_motor.SetDistanceOffset(actuator_initial_length)
        actuator_fun = chrono.ChFunctionConst(0.0)
        if hasattr(lift_motor, "SetActuatorFunction"):
            lift_motor.SetActuatorFunction(actuator_fun)
        system.AddLink(lift_motor)

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
            "t", "target_fork_vy", "target_fork_y",
            "fork_cx", "fork_cy", "fork_cz", "fork_vx", "fork_vy", "fork_vz",
            "pallet_cx", "pallet_cy", "pallet_cz", "pallet_vx", "pallet_vy", "pallet_vz",
            "fork_top_y", "pallet_bottom_y", "vertical_gap", "effective_vertical_gap", "actuator_force_proxy"
        ])
        start = time.perf_counter()
        for i in range(n_steps + 1):
            t = system.GetChTime()
            target_fork_vy = -lift_speed if (t >= lift_start and t <= lift_end) else 0.0
            active_time = 0.0 if t <= lift_start else (min(t, lift_end) - lift_start)
            target_fork_y = COG_fork.y + (-lift_speed * active_time)
            if scenario == "benchmark":
                fork.SetPos(chrono.ChVector3d(COG_fork.x, target_fork_y, COG_fork.z))
                fork.SetPosDt(chrono.ChVector3d(0.0, target_fork_vy, 0.0))
                fork.SetRot(chrono.QUNIT)
                fork.SetAngVelParent(chrono.ChVector3d(0.0, 0.0, 0.0))
            else:
                target_actuator_extension = max(0.0, -lift_speed * active_time)
                if actuator_fun is not None:
                    actuator_fun.SetConstant(target_actuator_extension)

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
            actuator_force = lift_motor.GetMotorForce() if lift_motor is not None and hasattr(lift_motor, "GetMotorForce") else 0.0
            writer.writerow([
                f"{t:.9f}",
                f"{target_fork_vy:.9f}",
                f"{target_fork_y:.9f}",
                f"{fork_x.x:.9f}", f"{fork_x.y:.9f}", f"{fork_x.z:.9f}",
                f"{fork_v.x:.9f}", f"{fork_v.y:.9f}", f"{fork_v.z:.9f}",
                f"{pallet_x.x:.9f}", f"{pallet_x.y:.9f}", f"{pallet_x.z:.9f}",
                f"{pallet_v.x:.9f}", f"{pallet_v.y:.9f}", f"{pallet_v.z:.9f}",
                f"{fork_top_y:.9f}",
                f"{pallet_bottom_y:.9f}",
                f"{vertical_gap:.9f}",
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
        "scenario": scenario,
        "wall_time_s": f"{wall_time:.6g}",
        "avg_solver_iterations": f"{avg_iters:.6g}",
        "final_fork_y": f"{last_fork_y:.9g}",
        "final_pallet_y": f"{last_pallet_y:.9g}",
        "final_pallet_z": f"{last_pallet_z:.9g}",
        "pallet_mass": f"{pallet_mass:.9g}",
        "pallet_ixx": f"{pallet_ixx:.9g}",
        "pallet_iyy": f"{pallet_iyy:.9g}",
        "pallet_izz": f"{pallet_izz:.9g}",
        "initial_effective_gap": f"{initial_effective_gap:.9g}",
        "time_step_s": f"{dt:.6g}",
        "end_time_s": f"{end_time:.6g}",
        "lift_start_s": f"{lift_start:.6g}",
        "lift_end_s": f"{lift_end:.6g}",
        "lift_speed": f"{lift_speed:.9g}",
        "lift_max_force": f"{lift_max_force:.9g}",
        "velocity_servo_gain": f"{servo_gain:.9g}",
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
    parser.add_argument("--scenario", choices=["benchmark", "demo_faithful"], default="benchmark")
    parser.add_argument("--mode", choices=["all", "nsc_psor", "smc_penalty"], default="all")
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--end-time", type=float, default=1.2)
    parser.add_argument("--lift-start", type=float, default=0.3)
    parser.add_argument("--lift-end", type=float, default=1.0)
    parser.add_argument("--lift-speed", type=float, default=-0.10)
    parser.add_argument("--lift-max-force", type=float, default=2e4)
    parser.add_argument("--pallet-y", type=float, default=0.15)
    parser.add_argument("--pallet-z", type=float, default=3.02)
    parser.add_argument("--pallet-mass", type=float, default=None)
    parser.add_argument("--pallet-ixx", type=float, default=None)
    parser.add_argument("--pallet-iyy", type=float, default=None)
    parser.add_argument("--pallet-izz", type=float, default=None)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--solver-max-iters", type=int, default=800)
    parser.add_argument("--solver-tol", type=float, default=1e-10)
    parser.add_argument("--velocity-servo-gain", type=float, default=None)
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
    pallet_path = resolve_model_path("pallet.obj")
    pallet_min, pallet_max = parse_obj_bounds(pallet_path)
    pallet_size = (pallet_max.x - pallet_min.x, pallet_max.y - pallet_min.y, pallet_max.z - pallet_min.z)
    default_pallet_mass = 300.0 * max(1e-6, pallet_size[0] * pallet_size[1] * pallet_size[2])
    default_ixx, default_iyy, default_izz = box_inertia_diag(default_pallet_mass, pallet_size)
    pallet_mass = args.pallet_mass if args.pallet_mass is not None else default_pallet_mass
    pallet_ixx = args.pallet_ixx if args.pallet_ixx is not None else default_ixx
    pallet_iyy = args.pallet_iyy if args.pallet_iyy is not None else default_iyy
    pallet_izz = args.pallet_izz if args.pallet_izz is not None else default_izz
    modes: Sequence[str] = ["nsc_psor", "smc_penalty"] if args.mode == "all" else [args.mode]
    rows = []
    for mode in modes:
        rows.append(
            run_case(
                mode=mode,
                scenario=args.scenario,
                dt=args.dt,
                end_time=args.end_time,
                output_base=output_base,
                tag=args.tag,
                solver_max_iters=args.solver_max_iters,
                solver_tol=args.solver_tol,
                lift_start=args.lift_start,
                lift_end=args.lift_end,
                lift_speed=args.lift_speed,
                lift_max_force=args.lift_max_force,
                pallet_y=args.pallet_y,
                pallet_z=args.pallet_z,
                pallet_mass=pallet_mass,
                pallet_ixx=pallet_ixx,
                pallet_iyy=pallet_iyy,
                pallet_izz=pallet_izz,
                friction=args.friction,
                nsc_compliance=args.nsc_compliance,
                smc_young=args.smc_young,
                smc_poisson=args.smc_poisson,
                smc_kn=args.smc_kn,
                smc_kt=args.smc_kt,
                smc_gn=args.smc_gn,
                smc_gt=args.smc_gt,
                velocity_servo_gain=args.velocity_servo_gain,
            )
        )
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
