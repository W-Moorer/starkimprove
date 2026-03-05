#!/usr/bin/env python3
"""
Collect cross-framework baseline for Exp5.

Outputs:
- output/paper_experiments/cross_framework_baseline_exp5.csv
- output/paper_experiments/cross_framework_baseline_exp5.tex
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"


def parse_numeric(text: str) -> Optional[float]:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def dedup_by_time(rows: List[Dict[str, str]], tkey: str = "t") -> List[Dict[str, str]]:
    latest: Dict[float, Dict[str, str]] = {}
    for row in rows:
        t = parse_numeric(row.get(tkey, ""))
        if t is None:
            continue
        latest[t] = row
    return [latest[t] for t in sorted(latest.keys())]


def parse_latest_logger(case_dir: Path) -> Dict[str, Optional[float]]:
    logger_files = sorted(case_dir.glob("logger_*.txt"), key=lambda p: p.stat().st_mtime)
    if not logger_files:
        return {"wall_time_s": None, "time_steps": None, "newton_iterations": None}

    metrics = {"wall_time_s": None, "time_steps": None, "newton_iterations": None}
    pattern = re.compile(r"^(total|time_steps|newton_iterations):\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)$")
    with logger_files[-1].open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m = pattern.match(line)
            if not m:
                continue
            key, val = m.group(1), float(m.group(2))
            if key == "total":
                metrics["wall_time_s"] = val
            elif key == "time_steps":
                metrics["time_steps"] = val
            elif key == "newton_iterations":
                metrics["newton_iterations"] = val
    return metrics


def format_num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def load_ref_curve(path: Path, scale: float = 1.0) -> Tuple[List[float], List[float]]:
    ts: List[float] = []
    vs: List[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            t = parse_numeric(row[0])
            v = parse_numeric(row[1])
            if t is None or v is None:
                continue
            ts.append(t)
            vs.append(v * scale)
    return ts, vs


def interp_linear(xs: List[float], ys: List[float], xq: float) -> Optional[float]:
    if not xs or xq < xs[0] or xq > xs[-1]:
        return None
    lo, hi = 0, len(xs) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= xq:
            lo = mid
        else:
            hi = mid
    x0, x1 = xs[lo], xs[hi]
    y0, y1 = ys[lo], ys[hi]
    if x1 == x0:
        return y0
    a = (xq - x0) / (x1 - x0)
    return y0 * (1.0 - a) + y1 * a


def rmse(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return math.sqrt(sum(v * v for v in values) / len(values))


def collect_exp5_rows(output_base: Path) -> List[Dict[str, str]]:
    case = output_base / "exp5_bolt"
    state_path = case / "screw_state.csv"
    if not state_path.exists():
        return []

    state = dedup_by_time(read_csv_rows(state_path))
    if not state:
        return []

    t_sim = [parse_numeric(r["t"]) for r in state]
    y_sim = [parse_numeric(r["y"]) for r in state]
    vy_sim = [parse_numeric(r["vy"]) for r in state]
    data = [(t, y, vy) for t, y, vy in zip(t_sim, y_sim, vy_sim) if t is not None and y is not None and vy is not None]
    t_sim = [d[0] for d in data]
    y_sim = [d[1] for d in data]
    vy_sim = [d[2] for d in data]

    pos_ref_t, pos_ref_y = load_ref_curve(output_base / "exp5_ref" / "Pos_Ty_mu0.csv", scale=0.01)
    vel_ref_t, vel_ref_v = load_ref_curve(output_base / "exp5_ref" / "Vel_Ty_mu0.csv", scale=0.01)

    y0_sim = y_sim[0]
    y0_ref = pos_ref_y[0] if pos_ref_y else 0.0
    y_sim_aligned = [y - y0_sim for y in y_sim]

    y_err: List[float] = []
    vy_err: List[float] = []
    y_ref_at_t_end: Optional[float] = None
    vy_ref_at_t_end: Optional[float] = None
    for t, ys, vys in zip(t_sim, y_sim_aligned, vy_sim):
        yr = interp_linear(pos_ref_t, pos_ref_y, t)
        vr = interp_linear(vel_ref_t, vel_ref_v, t)
        if yr is not None:
            y_err.append(ys - (yr - y0_ref))
            y_ref_at_t_end = yr - y0_ref
        if vr is not None:
            vy_err.append(vys - vr)
            vy_ref_at_t_end = vr

    logm = parse_latest_logger(case)
    iter_per_step = None
    if logm["time_steps"] and logm["newton_iterations"] and logm["time_steps"] > 0:
        iter_per_step = logm["newton_iterations"] / logm["time_steps"]

    t_end = t_sim[-1] if t_sim else None
    return [
        {
            "method_id": "exp5_stark_vs_ref",
            "method_name": "STARK IPC vs Reference",
            "framework": "STARK / Reference",
            "contact_formulation": "Threaded contact (mu=0)",
            "simulated_time_s": format_num(t_end, 2),
            "step_count": format_num(logm["time_steps"], 0),
            "wall_time_s": format_num(logm["wall_time_s"], 3),
            "iter_per_step": format_num(iter_per_step, 2),
            "final_y_sim_aligned": format_num(y_sim_aligned[-1] if y_sim_aligned else None, 6),
            "final_y_ref_aligned": format_num(y_ref_at_t_end, 6),
            "final_vy_sim": format_num(vy_sim[-1] if vy_sim else None, 6),
            "final_vy_ref": format_num(vy_ref_at_t_end, 6),
            "rmse_y": format_num(rmse(y_err), 6),
            "rmse_vy": format_num(rmse(vy_err), 6),
        }
    ]


def write_csv(rows: List[Dict[str, str]], path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_exp5_tex(rows: List[Dict[str, str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    r = rows[0]
    lines = [
        "% Auto-generated by collect_cross_framework_baseline.py",
        "\\begin{tabular}{cccccc}",
        "\\toprule",
        "$y_{\\mathrm{end}}^{\\mathrm{sim}}$ (m) & $y_{\\mathrm{end}}^{\\mathrm{ref}}$ (m) & "
        "$v_{y,\\mathrm{end}}^{\\mathrm{sim}}$ (m/s) & $v_{y,\\mathrm{end}}^{\\mathrm{ref}}$ (m/s) & "
        "RMSE$_y$ (m) & RMSE$_{v_y}$ (m/s) \\\\",
        "\\midrule",
        f"{r['final_y_sim_aligned']} & {r['final_y_ref_aligned']} & {r['final_vy_sim']} & {r['final_vy_ref']} & "
        f"{r['rmse_y']} & {r['rmse_vy']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Exp5 cross-framework baseline.")
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_base = args.output_base.resolve()
    exp5_rows = collect_exp5_rows(output_base)

    exp5_csv = output_base / "cross_framework_baseline_exp5.csv"
    exp5_tex = output_base / "cross_framework_baseline_exp5.tex"
    write_csv(
        exp5_rows,
        exp5_csv,
        [
            "method_id",
            "method_name",
            "framework",
            "contact_formulation",
            "simulated_time_s",
            "step_count",
            "wall_time_s",
            "iter_per_step",
            "final_y_sim_aligned",
            "final_y_ref_aligned",
            "final_vy_sim",
            "final_vy_ref",
            "rmse_y",
            "rmse_vy",
        ],
    )
    write_exp5_tex(exp5_rows, exp5_tex)

    print(f"Wrote: {exp5_csv}")
    print(f"Wrote: {exp5_tex}")


if __name__ == "__main__":
    main()
