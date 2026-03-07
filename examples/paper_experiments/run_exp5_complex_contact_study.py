#!/usr/bin/env python3
"""
Run Exp5 as a complex-contact supplement with RecurDyn-aligned matched-error selection.

Outputs:
- output/paper_experiments/exp5_complex_contact_summary.csv
- output/paper_experiments/exp5_pychrono_matched_sweep.csv
- output/paper_experiments/exp5_complex_contact_summary.md
- documents/local/paper1/figs/exp5_bolt_vs_ref.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_utils import FIGS_DIR, OUTPUT_BASE, latest_logger, parse_logger_metrics, resolve_executable, save_fig, setup_axes


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]


def parse_list(values: str, cast=float) -> List[float]:
    out: List[float] = []
    for token in values.split(","):
        token = token.strip()
        if token:
            out.append(cast(token))
    if not out:
        raise ValueError("Empty list.")
    return out


def load_state(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["t", "x", "y", "z", "vx", "vy", "vz"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["t", "x", "y", "z", "vx", "vy", "vz"])
    df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t").reset_index(drop=True)
    return df


def load_recurdyn_curve(path: Path, scale: float = 0.01) -> pd.DataFrame:
    df = pd.read_csv(path)
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce") * scale
    mask = x.notna() & y.notna()
    return pd.DataFrame({"t": x[mask], "v": y[mask]}).sort_values("t").reset_index(drop=True)


def error_against_recurdyn(
    state: pd.DataFrame,
    ref_pos: pd.DataFrame,
    ref_vel: pd.DataFrame,
) -> Dict[str, float]:
    t = state["t"].to_numpy()
    y = state["y"].to_numpy()
    vy = state["vy"].to_numpy()
    x = state["x"].to_numpy()
    z = state["z"].to_numpy()

    y0_sim = float(y[0])
    y0_ref = float(ref_pos["v"].iloc[0])
    y_aligned = y - y0_sim
    ref_y = np.interp(t, ref_pos["t"].to_numpy(), ref_pos["v"].to_numpy()) - y0_ref
    ref_vy = np.interp(t, ref_vel["t"].to_numpy(), ref_vel["v"].to_numpy())

    rmse_y = float(np.sqrt(np.mean((y_aligned - ref_y) ** 2)))
    rmse_vy = float(np.sqrt(np.mean((vy - ref_vy) ** 2)))

    y_scale = max(1e-4, float(ref_pos["v"].max() - ref_pos["v"].min()))
    vy_scale = max(1e-4, float(np.abs(ref_vel["v"]).max()))
    composite_error = rmse_y / y_scale + rmse_vy / vy_scale

    radial = np.sqrt(x**2 + z**2)
    return {
        "final_y_sim_aligned": float(y_aligned[-1]),
        "final_y_recurdyn_aligned": float(ref_y[-1]),
        "final_vy_sim": float(vy[-1]),
        "final_vy_recurdyn": float(ref_vy[-1]),
        "rmse_y_to_recurdyn": rmse_y,
        "rmse_vy_to_recurdyn": rmse_vy,
        "composite_error": composite_error,
        "max_transverse_offset": float(np.max(radial)),
        "final_transverse_offset": float(radial[-1]),
        "stable": 1.0,
    }


def stark_env(run_name: str, dt: float, end_time: float) -> Dict[str, str]:
    env = dict(os.environ)
    env["STARK_EXP5_RUN_NAME"] = run_name
    env["STARK_EXP5_DT"] = f"{dt:.12g}"
    env["STARK_EXP5_END_TIME"] = f"{end_time:.12g}"
    return env


def run_stark_case(exe: Path, run_name: str, dt: float, end_time: float, force_run: bool) -> Dict[str, object]:
    case_dir = OUTPUT_BASE / run_name
    state_csv = case_dir / "screw_state.csv"
    logger = latest_logger(case_dir)
    if force_run or logger is None or not state_csv.exists():
        cmd = [str(exe), "exp5"]
        env = stark_env(run_name, dt, end_time)
        ret = subprocess.run(cmd, cwd=exe.parents[3], env=env)
        if ret.returncode != 0:
            raise RuntimeError(f"STARK exp5 failed: {run_name}")
        logger = latest_logger(case_dir)
    if logger is None or not state_csv.exists():
        raise FileNotFoundError(f"Missing STARK exp5 outputs in {case_dir}")

    metrics = parse_logger_metrics(logger)
    state = load_state(state_csv)
    return {
        "label": "STARK IPC vs RecurDyn",
        "method_id": "exp5_stark_vs_recurdyn",
        "method_name": "STARK IPC",
        "framework": "STARK",
        "contact_formulation": "Barrier/IPC threaded contact",
        "run_name": run_name,
        "logger_file": logger.name,
        "time_step_s": dt,
        "simulated_time_s": end_time,
        "step_count": len(state),
        "wall_time_s": float(metrics.get("total", float("nan"))),
        "iter_per_step": (
            float(metrics["newton_iterations"]) / float(metrics["time_steps"])
            if metrics.get("newton_iterations") and metrics.get("time_steps")
            else float("nan")
        ),
        "state_csv": str(state_csv),
        "mode": "stark",
        "solver_max_iters": "",
        "solver_tol": "",
        "nsc_compliance": "",
        "smc_kn": "",
        "smc_kt": "",
        "smc_gn": "",
        "smc_gt": "",
        "smc_young": "",
        "smc_poisson": "",
        "matched_target_hit": True,
        "selection_role": "reference-aligned target",
        "selection_target": "RecurDyn score band",
    }


def make_pychrono_tag(mode: str, dt: float, params: Dict[str, float]) -> str:
    if mode in {"nsc_lcp", "nsc_ncp"}:
        return (
            f"dt{dt:.0e}_iters{int(params['solver_max_iters'])}_comp{params['nsc_compliance']:.0e}"
            .replace("+", "")
            .replace(".", "p")
        )
    return (
        f"dt{dt:.0e}_iters{int(params['solver_max_iters'])}_kn{params['smc_kn']:.0e}_gn{params['smc_gn']:.0e}"
        .replace("+", "")
        .replace(".", "p")
    )


def default_pychrono_params(mode: str, solver_max_iters: int) -> Dict[str, float]:
    return {
        "solver_max_iters": float(solver_max_iters),
        "solver_tol": 1e-8 if mode != "smc_penalty" else 1e-10,
        "nsc_compliance": 1e-9,
        "smc_kn": 5e6,
        "smc_kt": 2e6,
        "smc_gn": 1e3,
        "smc_gt": 1e3,
        "smc_young": 2e8,
        "smc_poisson": 0.3,
    }


def build_pychrono_grid(
    lcp_dts: Iterable[float],
    lcp_iters: Iterable[int],
    lcp_compliances: Iterable[float],
    ncp_dts: Iterable[float],
    ncp_iters: Iterable[int],
    ncp_compliances: Iterable[float],
    smc_dts: Iterable[float],
    smc_iters: Iterable[int],
    smc_kns: Iterable[float],
    smc_gns: Iterable[float],
) -> List[tuple[str, float, Dict[str, float]]]:
    grid: List[tuple[str, float, Dict[str, float]]] = []
    for dt in lcp_dts:
        for max_iters in lcp_iters:
            for compliance in lcp_compliances:
                params = default_pychrono_params("nsc_lcp", max_iters)
                params["nsc_compliance"] = compliance
                grid.append(("nsc_lcp", float(dt), params))
    for dt in ncp_dts:
        for max_iters in ncp_iters:
            for compliance in ncp_compliances:
                params = default_pychrono_params("nsc_ncp", max_iters)
                params["nsc_compliance"] = compliance
                grid.append(("nsc_ncp", float(dt), params))
    for dt in smc_dts:
        for max_iters in smc_iters:
            for kn in smc_kns:
                for gn in smc_gns:
                    params = default_pychrono_params("smc_penalty", max_iters)
                    params["smc_kn"] = kn
                    params["smc_kt"] = 0.4 * kn
                    params["smc_gn"] = gn
                    params["smc_gt"] = gn
                    grid.append(("smc_penalty", float(dt), params))
    return grid


def run_pychrono_case(
    mode: str,
    dt: float,
    end_time: float,
    params: Dict[str, float],
    force_run: bool,
) -> Dict[str, object]:
    tag = make_pychrono_tag(mode, dt, params)
    case_dir = OUTPUT_BASE / f"pychrono_exp5_{mode}_{tag}"
    summary_csv = case_dir / "summary.csv"
    state_csv = case_dir / "screw_state.csv"
    legacy_case_dir = OUTPUT_BASE / f"pychrono_exp5_{mode}"
    legacy_summary_csv = legacy_case_dir / "summary.csv"
    legacy_state_csv = legacy_case_dir / "screw_state.csv"

    def matches_legacy_defaults(legacy_dt: float) -> bool:
        if mode == "nsc_lcp":
            return (
                abs(legacy_dt - 0.05) < 1e-12
                and abs(dt - 0.05) < 1e-12
                and int(params["solver_max_iters"]) == 120
                and abs(params["nsc_compliance"] - 1e-9) < 1e-20
            )
        if mode == "nsc_ncp":
            return (
                abs(legacy_dt - 0.05) < 1e-12
                and abs(dt - 0.05) < 1e-12
                and int(params["solver_max_iters"]) == 200
                and abs(params["nsc_compliance"] - 1e-9) < 1e-20
            )
        if mode == "smc_penalty":
            return (
                abs(legacy_dt - 0.01) < 1e-12
                and abs(dt - 0.01) < 1e-12
                and int(params["solver_max_iters"]) == 200
                and abs(params["smc_kn"] - 5e6) < 1e-12
                and abs(params["smc_gn"] - 1e3) < 1e-12
            )
        return False

    def candidate_is_invalid(path: Path) -> bool:
        if not path.exists():
            return True
        row = pd.read_csv(path).iloc[0]
        invalid_state = int(float(row.get("invalid_state", 0)))
        final_y = pd.to_numeric([row.get("final_y", float("nan"))], errors="coerce")[0]
        final_vy = pd.to_numeric([row.get("final_vy", float("nan"))], errors="coerce")[0]
        return invalid_state != 0 or not np.isfinite(final_y) or not np.isfinite(final_vy)

    if not force_run and legacy_summary_csv.exists() and legacy_state_csv.exists():
        legacy_row = pd.read_csv(legacy_summary_csv).iloc[0]
        legacy_dt = float(legacy_row.get("time_step_s", dt))
        current_invalid = candidate_is_invalid(summary_csv) if summary_csv.exists() else True
        use_legacy = matches_legacy_defaults(legacy_dt) and current_invalid
        if use_legacy:
            case_dir = legacy_case_dir
            summary_csv = legacy_summary_csv
            state_csv = legacy_state_csv

    if force_run or not summary_csv.exists() or not state_csv.exists():
        script_path = REPO_ROOT / "examples" / "paper_experiments" / "pychrono_exp5_baselines.py"
        cmd = (
            "conda activate chrono-baseline; "
            f"python \"{script_path}\" "
            f"--mode {mode} "
            f"--dt {dt:.12g} "
            f"--end-time {end_time:.12g} "
            f"--output-base \"{OUTPUT_BASE}\" "
            f"--tag {tag} "
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
        ret = subprocess.run(
            ["powershell", "-NoLogo", "-Command", cmd],
            cwd=str(REPO_ROOT),
        )
        if ret.returncode != 0:
            raise RuntimeError(f"PyChrono exp5 failed: {mode} {tag}")

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing PyChrono summary: {summary_csv}")

    row = pd.read_csv(summary_csv).iloc[0].to_dict()
    out: Dict[str, object] = {
        "label": row.get("method_name", row.get("method_id", mode)),
        "method_id": row.get("method_id", f"pychrono_exp5_{mode}_{tag}"),
        "method_name": row.get("method_name", mode),
        "framework": "PyChrono",
        "contact_formulation": row.get("contact_formulation", mode),
        "run_name": case_dir.name,
        "logger_file": "",
        "time_step_s": float(row.get("time_step_s", dt)),
        "simulated_time_s": float(row.get("simulated_time_s", end_time)),
        "step_count": int(float(row.get("step_count", 0))),
        "wall_time_s": float(row.get("wall_time_s", float("nan"))),
        "iter_per_step": float(row.get("avg_solver_iterations", float("nan"))),
        "state_csv": str(state_csv),
        "mode": mode,
        "solver_max_iters": int(float(row.get("solver_max_iters", params["solver_max_iters"]))),
        "solver_tol": float(row.get("solver_tol", params["solver_tol"])),
        "nsc_compliance": float(row.get("nsc_compliance", params["nsc_compliance"])),
        "smc_kn": float(row.get("smc_kn", params["smc_kn"])),
        "smc_kt": float(row.get("smc_kt", params["smc_kt"])),
        "smc_gn": float(row.get("smc_gn", params["smc_gn"])),
        "smc_gt": float(row.get("smc_gt", params["smc_gt"])),
        "smc_young": float(row.get("smc_young", params["smc_young"])),
        "smc_poisson": float(row.get("smc_poisson", params["smc_poisson"])),
        "matched_target_hit": False,
        "selection_role": "",
        "selection_target": "RecurDyn score band",
    }
    return out


def choose_matched_case(candidates: List[Dict[str, object]], target_score: float, score_band: float) -> Dict[str, object]:
    if not candidates:
        raise RuntimeError("No PyChrono candidates available for selection.")
    stable = [row for row in candidates if bool(row.get("stable", 0.0))]
    if not stable:
        chosen = dict(sorted(candidates, key=lambda row: float(row.get("wall_time_s", float("inf"))))[0])
        chosen["matched_target_hit"] = False
        chosen["selection_role"] = "no stable candidate"
        return chosen
    stable = sorted(stable, key=lambda row: (float(row["composite_error"]), float(row["wall_time_s"])))
    eligible = [row for row in stable if float(row["composite_error"]) <= score_band * target_score]
    chosen = min(eligible, key=lambda row: float(row["wall_time_s"])) if eligible else stable[0]
    chosen["matched_target_hit"] = bool(eligible)
    return chosen


def write_csv(rows: List[Dict[str, object]], path: Path):
    if not rows:
        return
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


def write_md(rows: List[Dict[str, object]], path: Path):
    lines = [
        "# Exp5 Complex Contact Matched-Error Summary",
        "",
        "Reference curves come from RecurDyn (`mu=0`).",
        "",
    ]
    for row in rows:
        lines.append(f"## {row.get('label', row.get('method_name', 'Unknown'))}")
        for key, value in row.items():
            if key in {"state_csv"}:
                continue
            lines.append(f"- `{key}`: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(rows: List[Dict[str, object]], ref_pos: pd.DataFrame, ref_vel: pd.DataFrame, fig_dir: Path):
    fig, axs = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
    for ax in axs:
        setup_axes(ax)

    palette = {
        "STARK": "#1f77b4",
        "PyChrono-LCP": "#2ca02c",
        "PyChrono-APGD": "#ff7f0e",
        "PyChrono-SMC": "#d62728",
    }

    y0_ref = float(ref_pos["v"].iloc[0])
    axs[0].plot(ref_pos["t"], ref_pos["v"] - y0_ref, color="black", linestyle="--", linewidth=1.6, label="RecurDyn")
    axs[1].plot(ref_vel["t"], ref_vel["v"], color="black", linestyle="--", linewidth=1.6, label="RecurDyn")

    for row in rows:
        if row.get("framework") == "RecurDyn":
            continue
        state = load_state(Path(str(row["state_csv"])))
        if state.empty:
            continue
        y0 = float(state["y"].iloc[0])
        if row["framework"] == "STARK":
            label = "STARK IPC"
            color = palette["STARK"]
        else:
            mode = str(row["mode"])
            label = "PyChrono NSC-LCP" if mode == "nsc_lcp" else ("PyChrono NSC-APGD" if mode == "nsc_ncp" else "PyChrono SMC")
            color = palette["PyChrono-LCP"] if mode == "nsc_lcp" else (palette["PyChrono-APGD"] if mode == "nsc_ncp" else palette["PyChrono-SMC"])
        axs[0].plot(state["t"], state["y"] - y0, color=color, label=label)
        axs[1].plot(state["t"], state["vy"], color=color, label=label)

    axs[0].set_ylabel("Aligned axial y (m)")
    axs[0].set_title("Exp5: Screw Response vs RecurDyn")
    axs[0].legend(loc="upper right")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel(r"Axial velocity $v_y$ (m/s)")
    axs[1].legend(loc="lower right")
    save_fig(fig, fig_dir, "exp5_bolt_vs_ref")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp5 complex-contact study with matched-error PyChrono selection.")
    parser.add_argument("--exe", type=Path, default=None)
    parser.add_argument("--stark-run-name", type=str, default="exp5_bolt")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--end-time", type=float, default=5.0)
    parser.add_argument("--score-band", type=float, default=1.1)
    parser.add_argument("--force-run-stark", action="store_true")
    parser.add_argument("--force-run-pychrono", action="store_true")
    parser.add_argument("--lcp-dts", type=str, default="0.05")
    parser.add_argument("--lcp-iters", type=str, default="120")
    parser.add_argument("--lcp-compliances", type=str, default="1e-9")
    parser.add_argument("--ncp-dts", type=str, default="0.05,0.02")
    parser.add_argument("--ncp-iters", type=str, default="200")
    parser.add_argument("--ncp-compliances", type=str, default="1e-9")
    parser.add_argument("--smc-dts", type=str, default="0.01")
    parser.add_argument("--smc-iters", type=str, default="200")
    parser.add_argument("--smc-kns", type=str, default="5e6")
    parser.add_argument("--smc-gns", type=str, default="1e3")
    parser.add_argument("--summary-csv", type=Path, default=OUTPUT_BASE / "exp5_complex_contact_summary.csv")
    parser.add_argument("--sweep-csv", type=Path, default=OUTPUT_BASE / "exp5_pychrono_matched_sweep.csv")
    parser.add_argument("--summary-md", type=Path, default=OUTPUT_BASE / "exp5_complex_contact_summary.md")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)

    ref_pos = load_recurdyn_curve(OUTPUT_BASE / "exp5_ref" / "Pos_Ty_mu0.csv")
    ref_vel = load_recurdyn_curve(OUTPUT_BASE / "exp5_ref" / "Vel_Ty_mu0.csv")

    stark_row = run_stark_case(exe, args.stark_run_name, args.dt, args.end_time, args.force_run_stark)
    stark_row.update(error_against_recurdyn(load_state(Path(str(stark_row["state_csv"]))), ref_pos, ref_vel))

    grid = build_pychrono_grid(
        parse_list(args.lcp_dts, float),
        parse_list(args.lcp_iters, int),
        parse_list(args.lcp_compliances, float),
        parse_list(args.ncp_dts, float),
        parse_list(args.ncp_iters, int),
        parse_list(args.ncp_compliances, float),
        parse_list(args.smc_dts, float),
        parse_list(args.smc_iters, int),
        parse_list(args.smc_kns, float),
        parse_list(args.smc_gns, float),
    )

    sweep_rows: List[Dict[str, object]] = []
    for mode, dt, params in grid:
        row = run_pychrono_case(mode, dt, args.end_time, params, args.force_run_pychrono)
        state_path = Path(str(row["state_csv"]))
        try:
            state = load_state(state_path)
            if state.empty:
                raise RuntimeError("empty state")
            row.update(error_against_recurdyn(state, ref_pos, ref_vel))
        except Exception:
            row.update(
                {
                    "final_y_sim_aligned": float("nan"),
                    "final_y_recurdyn_aligned": float("nan"),
                    "final_vy_sim": float("nan"),
                    "final_vy_recurdyn": float("nan"),
                    "rmse_y_to_recurdyn": float("nan"),
                    "rmse_vy_to_recurdyn": float("nan"),
                    "composite_error": float("inf"),
                    "max_transverse_offset": float("nan"),
                    "final_transverse_offset": float("nan"),
                    "stable": 0.0,
                }
            )
        sweep_rows.append(row)

    lcp_best = choose_matched_case([row for row in sweep_rows if row["mode"] == "nsc_lcp"], float(stark_row["composite_error"]), args.score_band)
    lcp_best["selection_role"] = "nearest NSC-LCP candidate"
    ncp_best = choose_matched_case([row for row in sweep_rows if row["mode"] == "nsc_ncp"], float(stark_row["composite_error"]), args.score_band)
    ncp_best["selection_role"] = "nearest NSC-APGD candidate"
    smc_best = choose_matched_case([row for row in sweep_rows if row["mode"] == "smc_penalty"], float(stark_row["composite_error"]), args.score_band)
    smc_best["selection_role"] = "nearest SMC candidate"

    final_rows = [stark_row, lcp_best, ncp_best, smc_best]
    write_csv(sweep_rows, args.sweep_csv.resolve())
    write_csv(final_rows, args.summary_csv.resolve())
    write_md(final_rows, args.summary_md.resolve())

    plot_rows = [stark_row] + [row for row in [lcp_best, ncp_best, smc_best] if bool(row.get("stable")) and float(row.get("composite_error", float("inf"))) < 5.0]
    plot_curves(plot_rows, ref_pos, ref_vel, args.fig_dir.resolve())

    print(f"Wrote {args.summary_csv.resolve()}")
    print(f"Wrote {args.sweep_csv.resolve()}")
    print(f"Wrote {args.summary_md.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
