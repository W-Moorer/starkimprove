#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_utils import FIGS_DIR, OUTPUT_BASE, latest_logger, parse_logger_metrics, resolve_executable, sanitize_curve, save_fig, setup_axes


def parse_list(values: str, cast=float) -> List[float]:
    out: List[float] = []
    for token in values.split(","):
        token = token.strip()
        if token:
            out.append(cast(token))
    if not out:
        raise ValueError("Empty list.")
    return out


def stark_env(run_name: str, dt: float, end_time: float) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP2_RUN_NAME"] = run_name
    env["STARK_EXP2_DT"] = f"{dt:.12g}"
    env["STARK_EXP2_END_TIME"] = f"{end_time:.12g}"
    return env


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["t", "slider_cx", "slider_cy", "slider_vx", "slider_vy", "gap_stop"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", "slider_cx", "slider_vx", "gap_stop"])
    df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t").reset_index(drop=True)
    return df


def error_against_reference(ref: pd.DataFrame, cur: pd.DataFrame) -> Dict[str, float]:
    t = cur["t"].to_numpy()
    x = cur["slider_cx"].to_numpy()
    vx = cur["slider_vx"].to_numpy()
    gap = cur["gap_stop"].to_numpy()

    ref_x = np.interp(t, ref["t"].to_numpy(), ref["slider_cx"].to_numpy())
    ref_vx = np.interp(t, ref["t"].to_numpy(), ref["slider_vx"].to_numpy())
    ref_gap = np.interp(t, ref["t"].to_numpy(), ref["gap_stop"].to_numpy())

    rmse_x = float(np.sqrt(np.mean((x - ref_x) ** 2)))
    rmse_vx = float(np.sqrt(np.mean((vx - ref_vx) ** 2)))
    rmse_gap = float(np.sqrt(np.mean((gap - ref_gap) ** 2)))
    min_gap_err = float(abs(float(np.min(gap)) - float(np.min(ref["gap_stop"].to_numpy()))))

    x_scale = max(1e-3, float(ref["slider_cx"].max() - ref["slider_cx"].min()))
    vx_scale = max(1e-2, float(np.abs(ref["slider_vx"]).max()))
    gap_scale = max(1e-3, float(np.abs(ref["gap_stop"]).max()))
    composite = rmse_x / x_scale + rmse_vx / vx_scale + rmse_gap / gap_scale + min_gap_err / gap_scale
    return {
        "rmse_x_to_ref": rmse_x,
        "rmse_vx_to_ref": rmse_vx,
        "rmse_gap_to_ref": rmse_gap,
        "min_gap_error_to_ref": min_gap_err,
        "composite_error": composite,
    }


def run_stark_case(exe: Path, run_name: str, dt: float, end_time: float, force_run: bool) -> Dict[str, object]:
    case_dir = OUTPUT_BASE / run_name
    logger = latest_logger(case_dir)
    if force_run or logger is None:
        cmd = [str(exe), "exp2_slider"]
        env = stark_env(run_name, dt, end_time)
        print(f"[a2] run STARK {run_name}")
        ret = subprocess.run(cmd, cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"STARK A2 failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")

    metrics = parse_logger_metrics(logger)
    state_csv = case_dir / "crank_slider_state.csv"
    state = load_state(state_csv)
    return {
        "framework": "STARK",
        "method": "STARK IPC",
        "run_name": run_name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
        "min_gap": float(state["gap_stop"].min()),
        "state_csv": str(state_csv),
    }


def _format_tag(mode: str, params: Dict[str, float]) -> str:
    if mode == "nsc_ncp":
        return f"iters{int(params['solver_max_iters'])}_comp{params['nsc_compliance']:.0e}".replace("+", "")
    return (
        f"iters{int(params['solver_max_iters'])}_kn{params['smc_kn']:.0e}_gn{params['smc_gn']:.0e}"
        .replace("+", "")
    )


def run_pychrono_case(mode: str, dt: float, end_time: float, params: Dict[str, float], force_run: bool) -> Dict[str, object]:
    tag = _format_tag(mode, params)
    case_dir = OUTPUT_BASE / f"pychrono_exp2_crank_slider_{mode}_{tag}"
    summary_csv = case_dir / "summary.csv"
    state_csv = case_dir / "crank_slider_state.csv"
    if force_run or not summary_csv.exists() or not state_csv.exists():
        script_path = Path(__file__).resolve().parent / "pychrono_crank_slider_benchmark.py"
        cmd = (
            "conda activate chrono-baseline; "
            f"python '{script_path}' "
            f"--mode {mode} "
            f"--dt {dt:.12g} "
            f"--end-time {end_time:.12g} "
            f"--output-base '{OUTPUT_BASE}' "
            f"--tag '{tag}' "
            f"--solver-max-iters {int(params['solver_max_iters'])} "
            f"--solver-tol {params['solver_tol']:.12g} "
            f"--nsc-compliance {params['nsc_compliance']:.12g} "
            f"--smc-kn {params['smc_kn']:.12g} "
            f"--smc-kt {params['smc_kt']:.12g} "
            f"--smc-gn {params['smc_gn']:.12g} "
            f"--smc-gt {params['smc_gt']:.12g} "
            f"--smc-young {params['smc_young']:.12g} "
            f"--smc-poisson {params['smc_poisson']:.12g}"
        )
        print(f"[a2] run PyChrono {mode} {tag}")
        ret = subprocess.run(
            ["powershell", "-NoLogo", "-Command", cmd],
            cwd=Path(__file__).resolve().parents[2],
        )
        if ret.returncode != 0:
            raise RuntimeError(f"PyChrono A2 benchmark failed: {mode} {tag}")

    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    state = load_state(state_csv)
    return {
        "framework": "PyChrono",
        "method": "NSC/APGD" if mode == "nsc_ncp" else "SMC",
        "run_name": case_dir.name,
        "logger_file": "",
        "total": float(row["wall_time_s"]),
        "newton_iterations": None,
        "linear_iterations": float(row["avg_solver_iterations"]),
        "min_gap": float(state["gap_stop"].min()),
        "state_csv": str(state_csv),
        "solver_max_iters": int(row["solver_max_iters"]),
        "solver_tol": float(row["solver_tol"]),
        "nsc_compliance": float(row["nsc_compliance"]),
        "smc_kn": float(row["smc_kn"]),
        "smc_kt": float(row["smc_kt"]),
        "smc_gn": float(row["smc_gn"]),
        "smc_gt": float(row["smc_gt"]),
        "smc_young": float(row["smc_young"]),
        "smc_poisson": float(row["smc_poisson"]),
    }


def choose_matched_case(candidates: List[Dict[str, object]], target_score: float, score_band: float) -> Dict[str, object]:
    candidates = sorted(candidates, key=lambda row: (float(row["composite_error"]), float(row["total"])))
    eligible = [row for row in candidates if float(row["composite_error"]) <= score_band * target_score]
    if eligible:
        chosen = min(eligible, key=lambda row: float(row["total"]))
        chosen["matched_target_hit"] = True
        return chosen
    chosen = candidates[0]
    chosen["matched_target_hit"] = False
    return chosen


def default_pychrono_params(mode: str, solver_max_iters: int) -> Dict[str, float]:
    params = {
        "solver_max_iters": float(solver_max_iters),
        "solver_tol": 1e-8 if mode == "nsc_ncp" else 1e-10,
        "nsc_compliance": 1e-9,
        "smc_young": 2e8,
        "smc_poisson": 0.3,
        "smc_kn": 5e6,
        "smc_kt": 2e6,
        "smc_gn": 1e3,
        "smc_gt": 1e3,
    }
    return params


def build_pychrono_param_grid(nsc_iters: List[int], nsc_compliances: List[float], smc_iters: List[int], smc_kns: List[float], smc_gns: List[float]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    nsc_grid: List[Dict[str, float]] = []
    for max_iters in nsc_iters:
        for compliance in nsc_compliances:
            params = default_pychrono_params("nsc_ncp", max_iters)
            params["nsc_compliance"] = compliance
            nsc_grid.append(params)

    smc_grid: List[Dict[str, float]] = []
    for max_iters in smc_iters:
        for kn in smc_kns:
            for gn in smc_gns:
                params = default_pychrono_params("smc_penalty", max_iters)
                params["smc_kn"] = kn
                params["smc_kt"] = 0.4 * kn
                params["smc_gn"] = gn
                params["smc_gt"] = gn
                smc_grid.append(params)
    return nsc_grid, smc_grid


def write_csv(rows: List[Dict[str, object]], path: Path):
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


def plot_curves(rows: List[Dict[str, object]], fig_dir: Path):
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8), sharex=True)
    for ax in axs:
        setup_axes(ax)
    palette = {"STARK IPC": "#1f77b4", "NSC/APGD": "#2ca02c", "SMC": "#d62728"}
    for row in rows:
        state = pd.read_csv(Path(row["state_csv"]))
        x_curve = sanitize_curve(state, "t", "slider_cx")
        v_curve = sanitize_curve(state, "t", "slider_vx")
        axs[0].plot(x_curve["t"], x_curve["slider_cx"], color=palette[row["method"]], label=row["method"])
        axs[1].plot(v_curve["t"], v_curve["slider_vx"], color=palette[row["method"]], label=row["method"])
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Slider x (m)")
    axs[0].set_title("A2: Slider Displacement")
    axs[0].legend()
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Slider vx (m/s)")
    axs[1].set_title("A2: Slider Velocity")
    save_fig(fig, fig_dir, "a2_crank_slider_compare")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/collect A2 crank-slider benchmark under the contact-centric paper framing.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--ref-dt", type=float, default=0.001)
    parser.add_argument("--end-time", type=float, default=0.4)
    parser.add_argument("--nsc-iters", type=str, default="100,200,400")
    parser.add_argument("--nsc-compliances", type=str, default="1e-7,1e-8,1e-9")
    parser.add_argument("--smc-iters", type=str, default="100,200")
    parser.add_argument("--smc-kns", type=str, default="1e6,5e6,1e7,5e7")
    parser.add_argument("--smc-gns", type=str, default="5e2,1e3,5e3")
    parser.add_argument("--score-band", type=float, default=1.1)
    parser.add_argument("--force-run", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=OUTPUT_BASE / "a2_crank_slider_summary.csv")
    parser.add_argument("--sweep-csv", type=Path, default=OUTPUT_BASE / "a2_pychrono_matched_sweep.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)

    ref_row = run_stark_case(exe, "exp2_crank_slider_a2_reference", args.ref_dt, args.end_time, args.force_run)
    ref_state = load_state(Path(ref_row["state_csv"]))

    stark_row = run_stark_case(exe, "exp2_crank_slider_a2_stark", args.dt, args.end_time, args.force_run)
    stark_row.update(error_against_reference(ref_state, load_state(Path(stark_row["state_csv"]))))
    stark_row["matched_target_hit"] = True
    stark_row["selection_target"] = "reference"
    stark_row["solver_max_iters"] = ""
    stark_row["solver_tol"] = ""
    stark_row["nsc_compliance"] = ""
    stark_row["smc_kn"] = ""
    stark_row["smc_kt"] = ""
    stark_row["smc_gn"] = ""
    stark_row["smc_gt"] = ""
    stark_row["smc_young"] = ""
    stark_row["smc_poisson"] = ""

    target_score = float(stark_row["composite_error"])

    nsc_grid, smc_grid = build_pychrono_param_grid(
        parse_list(args.nsc_iters, int),
        parse_list(args.nsc_compliances, float),
        parse_list(args.smc_iters, int),
        parse_list(args.smc_kns, float),
        parse_list(args.smc_gns, float),
    )

    sweep_rows: List[Dict[str, object]] = []
    for params in nsc_grid:
        row = run_pychrono_case("nsc_ncp", args.dt, args.end_time, params, args.force_run)
        row.update(error_against_reference(ref_state, load_state(Path(row["state_csv"]))))
        row["selection_target"] = "stark_score_band"
        sweep_rows.append(row)
    for params in smc_grid:
        row = run_pychrono_case("smc_penalty", args.dt, args.end_time, params, args.force_run)
        row.update(error_against_reference(ref_state, load_state(Path(row["state_csv"]))))
        row["selection_target"] = "stark_score_band"
        sweep_rows.append(row)

    nsc_best = choose_matched_case([row for row in sweep_rows if row["method"] == "NSC/APGD"], target_score, args.score_band)
    smc_best = choose_matched_case([row for row in sweep_rows if row["method"] == "SMC"], target_score, args.score_band)

    final_rows = [stark_row, nsc_best, smc_best]
    write_csv(sweep_rows, args.sweep_csv.resolve())
    write_csv(final_rows, args.out_csv.resolve())
    plot_curves(final_rows, args.fig_dir.resolve())
    print(f"Wrote {args.out_csv.resolve()}")
    print(f"Wrote {args.sweep_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
