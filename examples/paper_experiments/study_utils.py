#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import shutil
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
FIGS_DIR = REPO_ROOT / "documents" / "local" / "paper1" / "figs"
DEFAULT_EXE_CANDIDATES = [
    REPO_ROOT / "build" / "examples" / "Release" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "Debug" / "examples.exe",
    REPO_ROOT / "build" / "examples" / "examples.exe",
]


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["lines.linewidth"] = 1.7


def setup_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.4)


def resolve_executable(path_arg: Optional[Path] = None) -> Path:
    if path_arg is not None:
        if path_arg.exists():
            return path_arg.resolve()
        raise FileNotFoundError(f"Executable not found: {path_arg}")
    for candidate in DEFAULT_EXE_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find examples executable.")


def resolve_conda_executable() -> Path:
    env_conda = Path(__import__("os").environ.get("CONDA_EXE", ""))
    candidates = []
    if str(env_conda):
        candidates.append(env_conda)
    which_conda = shutil.which("conda")
    if which_conda:
        candidates.append(Path(which_conda))
    candidates.extend(
        [
            Path(r"C:\ProgramData\anaconda3\Scripts\conda.exe"),
            Path(r"E:\Anaconda\Scripts\conda.exe"),
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Could not find conda executable.")


def build_conda_python_command(script_path: Path, *args: object, env_name: str = "chrono-baseline") -> list[str]:
    return [
        str(resolve_conda_executable()),
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(script_path),
        *[str(arg) for arg in args],
    ]


def latest_logger(case_dir: Path) -> Optional[Path]:
    logger_files = sorted(case_dir.glob("logger_*.txt"), key=lambda p: p.stat().st_mtime)
    return logger_files[-1] if logger_files else None


def parse_float(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (TypeError, ValueError, AttributeError):
        return None


def normalize_key(raw: str) -> str:
    key = raw.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


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


def sanitize_curve(df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 4000) -> pd.DataFrame:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    out = pd.DataFrame({x_col: x, y_col: y}).dropna()
    out = out.drop_duplicates(subset=[x_col], keep="last").sort_values(x_col)
    if len(out) > max_points:
        step = int(math.ceil(len(out) / max_points))
        out = out.iloc[::step].copy()
    return out


def resolve_run_curve(case_dir: Path, curve_stem: str, logger_name: str | None) -> Path:
    if logger_name:
        name = Path(logger_name).name
        if name.startswith("logger_") and name.endswith(".txt"):
            run_file = curve_stem + "_" + name[len("logger_") : -4] + ".csv"
            run_path = case_dir / run_file
            if run_path.exists():
                return run_path
    fallback = case_dir / f"{curve_stem}.csv"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing curve file: {fallback}")
    return fallback


def save_fig(fig: plt.Figure, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.pdf", format="pdf")
    fig.savefig(out_dir / f"{stem}.svg", format="svg")
    plt.close(fig)
