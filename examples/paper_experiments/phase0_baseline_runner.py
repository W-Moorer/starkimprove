#!/usr/bin/env python3
"""
Phase0 baseline runner/collector for STARK paper experiments.

Outputs:
- output/paper_experiments/phase0_baseline_minlog.csv
- output/paper_experiments/phase0_baseline_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_CONFIG = SCRIPT_PATH.with_name("phase0_experiments_config.json")
DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
DEFAULT_EXE_CANDIDATES = [
    REPO_ROOT / "build" / "examples" / "Release" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "Debug" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "examples.exe",
]
LOGGER_GLOB = "logger_*.txt"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (TypeError, ValueError):
        return None


def normalize_key(raw: str) -> str:
    key = raw.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_executable(path_arg: Optional[Path]) -> Path:
    if path_arg is not None:
        if path_arg.exists():
            return path_arg
        raise FileNotFoundError(f"Executable not found: {path_arg}")

    for candidate in DEFAULT_EXE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find examples executable. Build target `examples` first or pass --exe."
    )


def latest_logger(case_dir: Path) -> Optional[Path]:
    logger_files = sorted(case_dir.glob(LOGGER_GLOB), key=lambda p: p.stat().st_mtime)
    if not logger_files:
        return None
    return logger_files[-1]


def parse_logger_metrics(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if ":" not in line:
                continue
            left, right = line.split(":", 1)
            value = parse_float(right)
            if value is None:
                continue
            metrics[normalize_key(left)] = value
    return metrics


def read_last_row(path: Path) -> Optional[Dict[str, str]]:
    last: Optional[Dict[str, str]] = None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last


def maybe_assign_numeric(dst: Dict[str, object], key: str, row: Dict[str, str], *candidates: str):
    for col in candidates:
        if col in row:
            value = parse_float(row[col])
            if value is not None:
                dst[key] = value
                return


def collect_snapshot_fields(case_dir: Path) -> Dict[str, object]:
    out: Dict[str, object] = {}
    csv_candidates = [
        "min_z.csv",
        "joint_drift.csv",
        "velocity.csv",
        "impact_state.csv",
        "state.csv",
        "screw_state.csv",
    ]

    for name in csv_candidates:
        csv_path = case_dir / name
        if not csv_path.exists():
            continue
        row = read_last_row(csv_path)
        if not row:
            continue
        if "final_time" not in out:
            maybe_assign_numeric(out, "final_time", row, "t")
        maybe_assign_numeric(out, "final_min_z", row, "min_z")
        maybe_assign_numeric(out, "final_max_drift", row, "max_drift")
        maybe_assign_numeric(out, "final_v_x", row, "v_x", "vx")
        maybe_assign_numeric(out, "final_v_y", row, "v_y", "vy")
        maybe_assign_numeric(out, "final_v_z", row, "v_z", "vz")
    return out


def filter_cases(cases: List[Dict], cases_arg: Optional[str]) -> List[Dict]:
    if not cases_arg:
        return cases
    requested = {c.strip() for c in cases_arg.split(",") if c.strip()}
    selected = [c for c in cases if c.get("id") in requested]
    missing = sorted(requested - {c.get("id") for c in selected})
    if missing:
        raise ValueError(f"Unknown case id(s): {', '.join(missing)}")
    return selected


def run_cases(exe: Path, cases: Iterable[Dict]):
    for case in cases:
        case_id = case.get("id", "<unknown>")
        arg = case.get("experiment_arg")
        if not arg:
            raise ValueError(f"Case {case_id} missing experiment_arg")
        cmd = [str(exe), str(arg)]
        print(f"[run] {case_id}: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"Case {case_id} failed with exit code {result.returncode}")


def collect_records(
    output_base: Path,
    cases: Iterable[Dict],
    minimal_fields: List[str],
    strict: bool,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    records: List[Dict[str, object]] = []
    missing: List[Dict[str, str]] = []
    min_fields_norm = [normalize_key(f) for f in minimal_fields]

    for case in cases:
        case_id = str(case.get("id", "unknown_case"))
        arg = str(case.get("experiment_arg", ""))
        out_dirs = case.get("expected_output_dirs", [])
        for out_dir_name in out_dirs:
            case_dir = output_base / out_dir_name
            if not case_dir.exists():
                miss = {"case_id": case_id, "output_dir": out_dir_name, "reason": "missing_output_dir"}
                missing.append(miss)
                if strict:
                    raise FileNotFoundError(f"{miss}")
                continue

            logger = latest_logger(case_dir)
            if logger is None:
                miss = {"case_id": case_id, "output_dir": out_dir_name, "reason": "missing_logger"}
                missing.append(miss)
                if strict:
                    raise FileNotFoundError(f"{miss}")
                continue

            log_metrics = parse_logger_metrics(logger)
            row: Dict[str, object] = {
                "case_id": case_id,
                "experiment_arg": arg,
                "output_dir": out_dir_name,
                "logger_file": logger.name,
                "logger_mtime_utc": datetime.fromtimestamp(logger.stat().st_mtime, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "status": "ok",
            }
            for field in min_fields_norm:
                row[field] = log_metrics.get(field)
            row.update(collect_snapshot_fields(case_dir))
            records.append(row)
    return records, missing


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "case_id",
        "experiment_arg",
        "output_dir",
        "logger_file",
        "logger_mtime_utc",
        "status",
        "total",
        "time_steps",
        "newton_iterations",
        "linear_iterations",
        "cg_iterations",
        "hardening_count",
        "failed_step_count",
        "line_search_iterations",
        "failed_step_time",
        "failed_steps",
        "line_search",
        "before_energy_evaluation",
        "evaluate_e_grad_hess",
        "write_frame",
        "final_time",
        "final_min_z",
        "final_max_drift",
        "final_v_x",
        "final_v_y",
        "final_v_z",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and/or collect Phase0 STARK baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to phase0 config JSON (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE}).",
    )
    parser.add_argument(
        "--exe",
        type=Path,
        default=None,
        help="Path to examples executable. If omitted, common build paths are checked.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run configured experiments before collecting logs.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated case ids to run/collect (default: all configured cases).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected output folder or logger file is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config.resolve())
    cases = filter_cases(cfg.get("stark_baseline_runset", []), args.cases)
    if not cases:
        raise ValueError("No cases selected.")

    if args.run:
        exe = resolve_executable(args.exe.resolve() if args.exe else None)
        run_cases(exe, cases)

    output_base = args.output_base.resolve()
    records, missing = collect_records(
        output_base=output_base,
        cases=cases,
        minimal_fields=cfg.get("minimal_log_fields", []),
        strict=args.strict,
    )

    out_csv = output_base / "phase0_baseline_minlog.csv"
    out_json = output_base / "phase0_baseline_summary.json"
    write_csv(records, out_csv)

    summary = {
        "timestamp_utc": now_utc_iso(),
        "phase": cfg.get("phase"),
        "frozen_on": cfg.get("frozen_on"),
        "run_mode": "run_and_collect" if args.run else "collect_only",
        "selected_cases": [c.get("id") for c in cases],
        "records_count": len(records),
        "missing_count": len(missing),
        "missing": missing,
        "output_csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    if missing:
        print("Missing entries detected:")
        for miss in missing:
            print(f"  - {miss}")
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[phase0_baseline_runner] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
