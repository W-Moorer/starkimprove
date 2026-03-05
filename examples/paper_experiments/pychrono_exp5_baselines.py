#!/usr/bin/env python3
"""
Run PyChrono cross-framework baselines for Exp5 (screw-nut model).

Baselines:
- NSC + PSOR (LCP / Moreau-Jean family)
- NSC + APGD (NCP-like)
- SMC + penalty contact
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import pychrono.core as chrono
except ImportError as exc:
    raise SystemExit(
        "pychrono.core is not available. Run with chrono env, e.g.\n"
        "  conda run -n chrono-baseline python examples/paper_experiments/pychrono_exp5_baselines.py --mode all"
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

MODE_TO_TAG = {
    "nsc_lcp": "nsc_lcp",
    "nsc_ncp": "nsc_ncp",
    "smc_penalty": "smc_penalty",
}

MODE_TO_NAME = {
    "nsc_lcp": "PyChrono NSC (LCP/Moreau-Jean family)",
    "nsc_ncp": "PyChrono NSC (NCP-like/APGD)",
    "smc_penalty": "PyChrono SMC (Penalty)",
}


def resolve_model_path(file_name: str) -> Path:
    env_dir = None
    # Optional override for custom model path.
    if "STARK_MODELS_DIR" in __import__("os").environ:
        env_dir = Path(__import__("os").environ["STARK_MODELS_DIR"])

    candidates = [DEFAULT_MODELS_DIR / file_name]
    if env_dir is not None:
        candidates.insert(0, env_dir / file_name)

    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError(
        f"Model file not found: {file_name}. Tried: {', '.join(str(p) for p in candidates)}"
    )


def load_scaled_mesh(path: Path, scale: float):
    mesh = chrono.ChTriangleMeshConnected()
    ok = mesh.LoadWavefrontMesh(str(path), True, False)
    if not ok:
        raise RuntimeError(f"Failed to load mesh: {path}")
    mat = chrono.ChMatrix33d()
    mat[0, 0] = scale
    mat[1, 1] = scale
    mat[2, 2] = scale
    mesh.Transform(chrono.ChVector3d(0.0, 0.0, 0.0), mat)
    mesh.RepairDuplicateVertexes(1e-12)
    return mesh


def make_material(mode: str):
    if mode in {"nsc_lcp", "nsc_ncp"}:
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
    if mode == "nsc_lcp":
        system = chrono.ChSystemNSC()
        system.SetSolverType(chrono.ChSolver.Type_PSOR)
        system.SetTimestepperType(chrono.ChTimestepper.Type_EULER_IMPLICIT_PROJECTED)
    elif mode == "nsc_ncp":
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

    iterative = system.GetSolver().AsIterative()
    if iterative is not None:
        if mode == "nsc_lcp":
            iterative.SetMaxIterations(120)
            iterative.SetTolerance(1e-8)
        elif mode == "nsc_ncp":
            iterative.SetMaxIterations(200)
            iterative.SetTolerance(1e-8)
        else:
            iterative.SetMaxIterations(200)
            iterative.SetTolerance(1e-10)
        iterative.EnableWarmStart(True)

    return system


def add_mesh_body(system, mesh, density: float, material, fixed: bool, swept_sphere_radius: float = 1e-4):
    body = chrono.ChBodyEasyMesh(mesh, density, True, True, True, material, swept_sphere_radius)
    body.SetFixed(fixed)
    system.Add(body)
    return body


def run_case(
    mode: str,
    dt: float,
    end_time: float,
    output_base: Path,
    density: float = 8050.0,
    scale: float = 0.01,
) -> Dict[str, str]:
    system = configure_system(mode)
    material = make_material(mode)

    nut_mesh = load_scaled_mesh(resolve_model_path("nut-big.obj"), scale)
    screw_mesh = load_scaled_mesh(resolve_model_path("screw-big.obj"), scale)

    nut = add_mesh_body(system, nut_mesh, density, material, fixed=True)
    nut.SetPos(chrono.ChVector3d(0.0, 0.0, 0.0))

    screw = add_mesh_body(system, screw_mesh, density, material, fixed=False)
    screw.SetPos(chrono.ChVector3d(0.0, 0.03, 0.0))

    system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, -9.81, 0.0))

    mode_tag = MODE_TO_TAG[mode]
    out_dir = output_base / f"pychrono_exp5_{mode_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_steps = int(round(end_time / dt))
    state_path = out_dir / "screw_state.csv"
    summary_path = out_dir / "summary.csv"

    solver_iters: List[int] = []
    last_y = 0.0
    last_vy = 0.0

    with state_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "x", "y", "z", "vx", "vy", "vz"])

        start = time.perf_counter()
        for i in range(n_steps + 1):
            t = system.GetChTime()
            x = screw.GetPos()
            v = screw.GetPosDt()

            writer.writerow(
                [
                    f"{t:.9f}",
                    f"{x.x:.9f}",
                    f"{x.y:.9f}",
                    f"{x.z:.9f}",
                    f"{v.x:.9f}",
                    f"{v.y:.9f}",
                    f"{v.z:.9f}",
                ]
            )
            last_y = x.y
            last_vy = v.y

            if i < n_steps:
                system.DoStepDynamics(dt)
                iterative = system.GetSolver().AsIterative()
                iters = int(iterative.GetIterations()) if iterative is not None else 0
                solver_iters.append(iters)
        wall_time_s = time.perf_counter() - start

    avg_iters = (sum(solver_iters) / len(solver_iters)) if solver_iters else 0.0
    row = {
        "method_id": f"pychrono_exp5_{mode_tag}",
        "method_name": MODE_TO_NAME[mode],
        "framework": "PyChrono",
        "contact_formulation": (
            "LCP/NSC" if mode == "nsc_lcp" else ("NCP-like/NSC" if mode == "nsc_ncp" else "Penalty/SMC")
        ),
        "time_step_s": f"{dt:.6g}",
        "simulated_time_s": f"{end_time:.6g}",
        "step_count": str(n_steps + 1),
        "wall_time_s": f"{wall_time_s:.6g}",
        "avg_solver_iterations": f"{avg_iters:.6g}",
        "final_y": f"{last_y:.9f}",
        "final_vy": f"{last_vy:.9f}",
    }

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyChrono Exp5 baselines.")
    parser.add_argument(
        "--mode",
        choices=["all", "nsc_lcp", "nsc_ncp", "smc_penalty"],
        default="all",
        help="Baseline mode.",
    )
    parser.add_argument("--dt", type=float, default=0.01, help="Time step.")
    parser.add_argument("--end-time", type=float, default=5.0, help="Simulation end time.")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    modes: Sequence[str] = (
        ["nsc_lcp", "nsc_ncp", "smc_penalty"] if args.mode == "all" else [args.mode]
    )

    output_base = args.output_base.resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for mode in modes:
        try:
            row = run_case(mode, args.dt, args.end_time, output_base)
            rows.append(row)
            print(
                f"[{mode}] wall={float(row['wall_time_s']):.3f}s, "
                f"final_y={float(row['final_y']):.6f}, final_vy={float(row['final_vy']):.6f}"
            )
        except Exception as exc:
            print(f"[{mode}] failed: {exc}")

    if rows:
        out_csv = output_base / "pychrono_exp5_summary.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote summary: {out_csv}")


if __name__ == "__main__":
    main()
