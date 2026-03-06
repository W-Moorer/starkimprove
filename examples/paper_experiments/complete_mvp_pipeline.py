#!/usr/bin/env python3
"""
One-command MVP completion pipeline for paper1 (A1/A2/D1 + D2):
1) Collect (or run+collect) baseline cases from phase0 config.
2) Run D1 parameter sensitivity sweep.
3) Generate figures.
4) Export MVP completion summary (json + markdown).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
FIG_OUT_DIR = REPO_ROOT / "documents" / "local" / "paper1" / "figs"

PHASE0_RUNNER = SCRIPT_PATH.with_name("phase0_baseline_runner.py")
D1_RUNNER = SCRIPT_PATH.with_name("run_d1_parameter_sensitivity.py")
PLOTTER = SCRIPT_PATH.with_name("plot_results.py")

PHASE0_CSV = OUTPUT_BASE / "phase0_baseline_minlog.csv"
PHASE0_SUMMARY_JSON = OUTPUT_BASE / "phase0_baseline_summary.json"
D1_CSV = OUTPUT_BASE / "d1_parameter_sensitivity.csv"
MVP_SUMMARY_JSON = OUTPUT_BASE / "mvp_completion_summary.json"
MVP_SUMMARY_MD = OUTPUT_BASE / "mvp_completion_summary.md"

DEFAULT_EXE_CANDIDATES = [
    REPO_ROOT / "build" / "examples" / "Release" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "Debug" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "examples.exe",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_executable(path_arg: Optional[Path]) -> Path:
    if path_arg is not None:
        p = path_arg.resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Executable not found: {p}")
    for candidate in DEFAULT_EXE_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find examples executable. Build target `examples` first or pass --exe.")


def run_cmd(cmd: List[str], cwd: Path):
    print(f"[cmd] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def summarize_baseline(csv_path: Path) -> Dict[str, Dict]:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    summary: Dict[str, Dict] = {}
    for case_id, g in df.groupby("case_id"):
        summary[str(case_id)] = {
            "rows": int(len(g)),
            "output_dirs": sorted(g["output_dir"].astype(str).unique().tolist()),
            "total_time_sum_s": float(pd.to_numeric(g["total"], errors="coerce").fillna(0.0).sum()),
            "failed_step_time_sum_s": float(pd.to_numeric(g["failed_step_time"], errors="coerce").fillna(0.0).sum()),
            "max_joint_drift": float(pd.to_numeric(g.get("final_max_drift"), errors="coerce").dropna().max())
            if "final_max_drift" in g.columns and pd.to_numeric(g.get("final_max_drift"), errors="coerce").notna().any()
            else None,
        }
    return summary


def summarize_d1(csv_path: Path) -> Dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    total = pd.to_numeric(df.get("total"), errors="coerce")
    err = pd.to_numeric(df.get("joint_error_max_l2"), errors="coerce")
    valid = df[total.notna() & err.notna()].copy()
    if valid.empty:
        return {"rows": int(len(df))}

    valid["total_num"] = pd.to_numeric(valid["total"], errors="coerce")
    valid["err_num"] = pd.to_numeric(valid["joint_error_max_l2"], errors="coerce")
    best_runtime = valid.loc[valid["total_num"].idxmin()]
    best_error = valid.loc[valid["err_num"].idxmin()]

    return {
        "rows": int(len(df)),
        "valid_rows": int(len(valid)),
        "best_runtime_case": {
            "case_idx": int(best_runtime["case_idx"]) if pd.notna(best_runtime.get("case_idx")) else None,
            "total_s": float(best_runtime["total_num"]),
            "joint_error_max_l2": float(best_runtime["err_num"]),
            "rho0": float(best_runtime["rho0"]) if pd.notna(best_runtime.get("rho0")) else None,
            "rho_update_ratio": float(best_runtime["rho_update_ratio"]) if pd.notna(best_runtime.get("rho_update_ratio")) else None,
        },
        "best_error_case": {
            "case_idx": int(best_error["case_idx"]) if pd.notna(best_error.get("case_idx")) else None,
            "total_s": float(best_error["total_num"]),
            "joint_error_max_l2": float(best_error["err_num"]),
            "rho0": float(best_error["rho0"]) if pd.notna(best_error.get("rho0")) else None,
            "rho_update_ratio": float(best_error["rho_update_ratio"]) if pd.notna(best_error.get("rho_update_ratio")) else None,
        },
    }


def resolve_exp4_run_selection(phase0_csv: Path) -> Dict[str, str]:
    if not phase0_csv.exists():
        return {}
    df = pd.read_csv(phase0_csv)
    if df.empty or "output_dir" not in df.columns or "logger_file" not in df.columns:
        return {}

    out: Dict[str, str] = {}
    for case_id, logger_key, dir_key in (
        ("A1_seed_joint_contact", "exp4_base_logger", "exp4_base_dir"),
        ("A1_seed_joint_contact_al", "exp4_al_logger", "exp4_al_dir"),
    ):
        subset = df[df["case_id"].astype(str) == case_id].copy()
        if subset.empty:
            continue
        if "logger_mtime_utc" in subset.columns:
            subset = subset.sort_values("logger_mtime_utc")
        out[logger_key] = str(subset.iloc[-1]["logger_file"])
        out[dir_key] = str(subset.iloc[-1]["output_dir"])
    return out


def list_generated_figures(fig_dir: Path) -> List[str]:
    if not fig_dir.exists():
        return []
    keys = {
        "exp1_settling",
        "exp2_impact",
        "exp4_drift",
        "a1_joint_drift_compare",
        "a1_chain10_joint_drift_compare",
        "exp4_chain10_drift",
        "exp5_bolt_vs_ref",
        "d2_runtime_breakdown",
        "d1_pareto_total_vs_error",
    }
    out: List[str] = []
    for stem in sorted(keys):
        for ext in ("pdf", "svg"):
            p = fig_dir / f"{stem}.{ext}"
            if p.exists():
                out.append(str(p))
    return out


def write_markdown_summary(path: Path, payload: Dict):
    lines: List[str] = []
    lines.append("# MVP Completion Summary")
    lines.append("")
    lines.append(f"- Generated UTC: `{payload.get('generated_utc')}`")
    lines.append(f"- Status: `{payload.get('status')}`")
    lines.append("")
    lines.append("## Required Scope (A1/A2/D1 + D2)")
    lines.append("")
    lines.append("- `A1`: represented by refreshed isolated four-bar runs `exp4_fourbar_a1fix_soft` (+ `exp4_fourbar_a1fix_al` comparison)")
    lines.append("- `A2`: represented by `exp2_v10/v100/v500`")
    lines.append("- `D2`: represented by `exp1_adaptive/gap_adaptive/fixed_soft`")
    lines.append("- `D1`: parameter sensitivity from `d1_parameter_sensitivity.csv`")
    lines.append("")
    lines.append("## Core Artifacts")
    lines.append("")
    for key in ("phase0_csv", "phase0_summary_json", "d1_csv", "mvp_summary_json"):
        v = payload.get("artifacts", {}).get(key)
        if v:
            lines.append(f"- `{key}`: `{v}`")
    fig_paths = payload.get("artifacts", {}).get("figures", [])
    if fig_paths:
        lines.append("- `figures`:")
        for p in fig_paths:
            lines.append(f"  - `{p}`")
    lines.append("")
    lines.append("## Metrics Snapshot")
    lines.append("")
    baseline = payload.get("baseline_summary", {})
    if baseline:
        for case_id, info in baseline.items():
            lines.append(
                f"- `{case_id}`: rows={info.get('rows')}, total_time_sum_s={info.get('total_time_sum_s'):.6g}, "
                f"failed_step_time_sum_s={info.get('failed_step_time_sum_s'):.6g}"
            )
    d1 = payload.get("d1_summary", {})
    if d1:
        lines.append(
            f"- `D1`: rows={d1.get('rows')}, valid_rows={d1.get('valid_rows')}, "
            f"best_runtime_total_s={d1.get('best_runtime_case', {}).get('total_s')}"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Complete paper1 MVP experiment pipeline.")
    parser.add_argument("--exe", type=Path, default=None, help="Path to examples executable.")
    parser.add_argument("--run-baseline", action="store_true", help="Run baseline simulations before collecting.")
    parser.add_argument("--run-d1", action="store_true", help="Run D1 sweep before plotting.")
    parser.add_argument("--d1-max-cases", type=int, default=8, help="Max D1 cases (default: 8).")
    parser.add_argument("--d1-rho0", type=str, default="1e2,1e3", help="D1 rho0 list.")
    parser.add_argument("--d1-rho-update", type=str, default="1.2,1.5", help="D1 rho update ratio list.")
    parser.add_argument("--d1-newton-tol", type=str, default="1e-3,1e-4", help="D1 Newton tolerance list.")
    parser.add_argument("--d1-linear-tol", type=str, default="1.0,0.1", help="D1 linear tolerance list.")
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE, help=f"Output base dir (default: {OUTPUT_BASE})")
    parser.add_argument("--fig-out", type=Path, default=FIG_OUT_DIR, help=f"Figure output dir (default: {FIG_OUT_DIR})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exe = resolve_executable(args.exe)
    output_base = args.output_base.resolve()
    fig_out = args.fig_out.resolve()

    baseline_cmd = [
        "python",
        str(PHASE0_RUNNER),
        "--output-base",
        str(output_base),
    ]
    if args.run_baseline:
        baseline_cmd += ["--run", "--exe", str(exe)]
    run_cmd(baseline_cmd, cwd=REPO_ROOT)

    d1_csv = output_base / D1_CSV.name
    if args.run_d1:
        d1_cmd = [
            "python",
            str(D1_RUNNER),
            "--exe",
            str(exe),
            "--out-csv",
            str(d1_csv),
            "--rho0",
            args.d1_rho0,
            "--rho-update",
            args.d1_rho_update,
            "--newton-tol",
            args.d1_newton_tol,
            "--linear-tol",
            args.d1_linear_tol,
            "--max-cases",
            str(args.d1_max_cases),
        ]
        run_cmd(d1_cmd, cwd=REPO_ROOT)

    phase0_csv = output_base / PHASE0_CSV.name
    exp4_loggers = resolve_exp4_run_selection(phase0_csv)
    plot_cmd = [
        "python",
        str(PLOTTER),
        "--data-dir",
        str(output_base),
        "--out-dir",
        str(fig_out),
        "--baseline-csv",
        str(phase0_csv),
        "--d1-csv",
        str(d1_csv),
    ]
    if exp4_loggers.get("exp4_base_logger"):
        plot_cmd += ["--exp4-base-logger", exp4_loggers["exp4_base_logger"]]
    if exp4_loggers.get("exp4_al_logger"):
        plot_cmd += ["--exp4-al-logger", exp4_loggers["exp4_al_logger"]]
    if exp4_loggers.get("exp4_base_dir"):
        plot_cmd += ["--exp4-base-dir", exp4_loggers["exp4_base_dir"]]
    if exp4_loggers.get("exp4_al_dir"):
        plot_cmd += ["--exp4-al-dir", exp4_loggers["exp4_al_dir"]]
    run_cmd(plot_cmd, cwd=REPO_ROOT)

    phase0_summary = output_base / PHASE0_SUMMARY_JSON.name
    baseline_summary = summarize_baseline(phase0_csv)
    d1_summary = summarize_d1(d1_csv)
    figure_list = list_generated_figures(fig_out)

    required_cases = {"D2_seed_stack20", "A2_seed_contact", "A1_seed_joint_contact"}
    baseline_ready = required_cases.issubset(set(baseline_summary.keys()))
    d1_ready = bool(d1_summary.get("rows", 0))
    status = "completed" if baseline_ready and d1_ready else "incomplete"

    payload = {
        "generated_utc": utc_now(),
        "status": status,
        "requirements": {
            "baseline_ready": baseline_ready,
            "d1_ready": d1_ready,
            "required_case_ids": sorted(required_cases),
        },
        "baseline_summary": baseline_summary,
        "d1_summary": d1_summary,
        "artifacts": {
            "phase0_csv": str(phase0_csv),
            "phase0_summary_json": str(phase0_summary),
            "d1_csv": str(d1_csv),
            "figures": figure_list,
            "mvp_summary_json": str(MVP_SUMMARY_JSON),
            "mvp_summary_md": str(MVP_SUMMARY_MD),
        },
    }

    MVP_SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_summary(MVP_SUMMARY_MD, payload)

    print(f"Wrote: {MVP_SUMMARY_JSON}")
    print(f"Wrote: {MVP_SUMMARY_MD}")
    print(f"MVP status: {status}")
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[complete_mvp_pipeline] ERROR: {exc}")
        raise SystemExit(1)
