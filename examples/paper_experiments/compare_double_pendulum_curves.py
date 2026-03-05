#!/usr/bin/env python3
"""
Compare STARK and PyChrono double-pendulum curves:
- COM displacement curves
- COM velocity curves
- support reaction force curves
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]

DEFAULT_OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
DEFAULT_STARK_DIR = DEFAULT_OUTPUT_BASE / "exp6_double_pendulum_stark"
DEFAULT_CHRONO_DIR = DEFAULT_OUTPUT_BASE / "exp6_double_pendulum_pychrono"
DEFAULT_COMPARE_DIR = DEFAULT_OUTPUT_BASE / "exp6_double_pendulum_compare"
DEFAULT_FIG_DIR = REPO_ROOT / "documents" / "local" / "paper1" / "figs"


@dataclass
class ErrorStats:
    rmse: float
    max_abs: float
    rel_rmse: float

    def to_dict(self) -> Dict[str, float]:
        return {"rmse": self.rmse, "max_abs": self.max_abs, "rel_rmse": self.rel_rmse}


def setup_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.45)


def read_time_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "t" not in df.columns:
        raise ValueError(f"Missing `t` column in {path}")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    df = df.drop_duplicates(subset=["t"], keep="last").sort_values("t")
    return df.reset_index(drop=True)


def interp_on_grid(df: pd.DataFrame, t_grid: np.ndarray, cols: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    t = df["t"].to_numpy()
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        mask = np.isfinite(t) & np.isfinite(v)
        if np.count_nonzero(mask) < 2:
            raise ValueError(f"Not enough valid points to interpolate column `{c}`.")
        out[c] = np.interp(t_grid, t[mask], v[mask])
    return out


def compute_error(a: np.ndarray, b: np.ndarray) -> ErrorStats:
    e = a - b
    rmse = float(np.sqrt(np.mean(e * e)))
    max_abs = float(np.max(np.abs(e)))
    amp = float(max(1e-12, np.max(np.abs(b))))
    return ErrorStats(rmse=rmse, max_abs=max_abs, rel_rmse=rmse / amp)


def save_fig(fig: plt.Figure, out_dirs: List[Path], stem: str):
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def compare(stark_dir: Path, chrono_dir: Path, compare_dir: Path, fig_dir: Path) -> Dict:
    stark_state = read_time_series(stark_dir / "double_pendulum_state.csv")
    chrono_state = read_time_series(chrono_dir / "double_pendulum_state.csv")
    stark_react = read_time_series(stark_dir / "support_reaction_est.csv")
    chrono_react = read_time_series(chrono_dir / "support_reaction.csv")

    t_min = max(float(stark_state["t"].min()), float(chrono_state["t"].min()))
    t_max = min(float(stark_state["t"].max()), float(chrono_state["t"].max()))
    stark_t = stark_state["t"].to_numpy()
    t_grid = stark_t[(stark_t >= t_min) & (stark_t <= t_max)]
    if len(t_grid) < 10:
        raise RuntimeError("Insufficient overlapping time window between STARK and PyChrono.")

    state_cols = [
        "rod1_x",
        "rod1_y",
        "rod1_vx",
        "rod1_vy",
        "rod2_x",
        "rod2_y",
        "rod2_vx",
        "rod2_vy",
    ]
    s = interp_on_grid(stark_state, t_grid, state_cols)
    c = interp_on_grid(chrono_state, t_grid, state_cols)

    # Displacement relative to each framework's initial COM state.
    for tag in ("rod1_x", "rod1_y", "rod2_x", "rod2_y"):
        s[tag + "_disp"] = s[tag] - s[tag][0]
        c[tag + "_disp"] = c[tag] - c[tag][0]

    # Speed magnitude.
    s["rod1_speed"] = np.sqrt(s["rod1_vx"] ** 2 + s["rod1_vy"] ** 2)
    s["rod2_speed"] = np.sqrt(s["rod2_vx"] ** 2 + s["rod2_vy"] ** 2)
    c["rod1_speed"] = np.sqrt(c["rod1_vx"] ** 2 + c["rod1_vy"] ** 2)
    c["rod2_speed"] = np.sqrt(c["rod2_vx"] ** 2 + c["rod2_vy"] ** 2)

    react_cols_stark = ["fx_est", "fy_est", "fz_est", "f_norm"]
    react_cols_chrono = ["fx_joint", "fy_joint", "fz_joint", "f_joint_norm"]
    sr = interp_on_grid(stark_react, t_grid, react_cols_stark)
    cr = interp_on_grid(chrono_react, t_grid, react_cols_chrono)

    # Possible sign convention mismatch for joint reaction vector.
    sign_options = [1.0, -1.0]
    best_sign = 1.0
    best_rmse = float("inf")
    for sign in sign_options:
        rmse_try = np.sqrt(np.mean((sr["fy_est"] - sign * cr["fy_joint"]) ** 2))
        if rmse_try < best_rmse:
            best_rmse = float(rmse_try)
            best_sign = sign
    cr_fx = best_sign * cr["fx_joint"]
    cr_fy = best_sign * cr["fy_joint"]
    cr_fz = best_sign * cr["fz_joint"]

    metrics = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_window": {"t_min": float(t_grid[0]), "t_max": float(t_grid[-1]), "num_samples": int(len(t_grid))},
        "reaction_sign_applied_to_pychrono": best_sign,
        "displacement": {
            "rod1_x": compute_error(s["rod1_x_disp"], c["rod1_x_disp"]).to_dict(),
            "rod1_y": compute_error(s["rod1_y_disp"], c["rod1_y_disp"]).to_dict(),
            "rod2_x": compute_error(s["rod2_x_disp"], c["rod2_x_disp"]).to_dict(),
            "rod2_y": compute_error(s["rod2_y_disp"], c["rod2_y_disp"]).to_dict(),
        },
        "speed": {
            "rod1": compute_error(s["rod1_speed"], c["rod1_speed"]).to_dict(),
            "rod2": compute_error(s["rod2_speed"], c["rod2_speed"]).to_dict(),
        },
        "support_reaction": {
            "fx": compute_error(sr["fx_est"], cr_fx).to_dict(),
            "fy": compute_error(sr["fy_est"], cr_fy).to_dict(),
            "fz": compute_error(sr["fz_est"], cr_fz).to_dict(),
            "norm": compute_error(sr["f_norm"], cr["f_joint_norm"]).to_dict(),
        },
    }

    pass_checks = {
        "rod1_disp_rel_rmse": metrics["displacement"]["rod1_y"]["rel_rmse"] < 0.03,
        "rod2_disp_rel_rmse": metrics["displacement"]["rod2_y"]["rel_rmse"] < 0.03,
        "rod1_speed_rel_rmse": metrics["speed"]["rod1"]["rel_rmse"] < 0.05,
        "rod2_speed_rel_rmse": metrics["speed"]["rod2"]["rel_rmse"] < 0.05,
        "reaction_norm_rel_rmse": metrics["support_reaction"]["norm"]["rel_rmse"] < 0.12,
    }
    metrics["consistency_pass"] = bool(all(pass_checks.values()))
    metrics["consistency_checks"] = pass_checks

    out_dirs = [compare_dir, fig_dir]
    # Figure 1: link1 displacement.
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax in axs:
        setup_axes(ax)
    axs[0].plot(t_grid, s["rod1_x_disp"], label="STARK")
    axs[0].plot(t_grid, c["rod1_x_disp"], "--", label="PyChrono")
    axs[0].set_ylabel("dx (m)")
    axs[0].set_title("Double Pendulum Link1 COM Displacement")
    axs[0].legend()
    axs[1].plot(t_grid, s["rod1_y_disp"], label="STARK")
    axs[1].plot(t_grid, c["rod1_y_disp"], "--", label="PyChrono")
    axs[1].set_ylabel("dy (m)")
    axs[1].set_xlabel("Time (s)")
    save_fig(fig, out_dirs, "exp6_link1_displacement")

    # Figure 2: link2 displacement.
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax in axs:
        setup_axes(ax)
    axs[0].plot(t_grid, s["rod2_x_disp"], label="STARK")
    axs[0].plot(t_grid, c["rod2_x_disp"], "--", label="PyChrono")
    axs[0].set_ylabel("dx (m)")
    axs[0].set_title("Double Pendulum Link2 COM Displacement")
    axs[0].legend()
    axs[1].plot(t_grid, s["rod2_y_disp"], label="STARK")
    axs[1].plot(t_grid, c["rod2_y_disp"], "--", label="PyChrono")
    axs[1].set_ylabel("dy (m)")
    axs[1].set_xlabel("Time (s)")
    save_fig(fig, out_dirs, "exp6_link2_displacement")

    # Figure 3: speed curves.
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax in axs:
        setup_axes(ax)
    axs[0].plot(t_grid, s["rod1_speed"], label="STARK")
    axs[0].plot(t_grid, c["rod1_speed"], "--", label="PyChrono")
    axs[0].set_ylabel("|v1| (m/s)")
    axs[0].set_title("Double Pendulum COM Speed")
    axs[0].legend()
    axs[1].plot(t_grid, s["rod2_speed"], label="STARK")
    axs[1].plot(t_grid, c["rod2_speed"], "--", label="PyChrono")
    axs[1].set_ylabel("|v2| (m/s)")
    axs[1].set_xlabel("Time (s)")
    save_fig(fig, out_dirs, "exp6_speed")

    # Figure 4: support reaction.
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax in axs:
        setup_axes(ax)
    axs[0].plot(t_grid, sr["fy_est"], label="STARK est Fy")
    axs[0].plot(t_grid, cr_fy, "--", label="PyChrono hinge Fy")
    axs[0].set_ylabel("Fy (N)")
    axs[0].set_title("Support Reaction Force")
    axs[0].legend()
    axs[1].plot(t_grid, sr["f_norm"], label="STARK est |F|")
    axs[1].plot(t_grid, cr["f_joint_norm"], "--", label="PyChrono hinge |F|")
    axs[1].set_ylabel("|F| (N)")
    axs[1].set_xlabel("Time (s)")
    axs[1].legend()
    save_fig(fig, out_dirs, "exp6_support_reaction")

    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "t": t_grid,
            "rod1_x_disp_stark": s["rod1_x_disp"],
            "rod1_x_disp_pychrono": c["rod1_x_disp"],
            "rod1_y_disp_stark": s["rod1_y_disp"],
            "rod1_y_disp_pychrono": c["rod1_y_disp"],
            "rod2_x_disp_stark": s["rod2_x_disp"],
            "rod2_x_disp_pychrono": c["rod2_x_disp"],
            "rod2_y_disp_stark": s["rod2_y_disp"],
            "rod2_y_disp_pychrono": c["rod2_y_disp"],
            "rod1_speed_stark": s["rod1_speed"],
            "rod1_speed_pychrono": c["rod1_speed"],
            "rod2_speed_stark": s["rod2_speed"],
            "rod2_speed_pychrono": c["rod2_speed"],
            "support_fy_stark": sr["fy_est"],
            "support_fy_pychrono": cr_fy,
            "support_fnorm_stark": sr["f_norm"],
            "support_fnorm_pychrono": cr["f_joint_norm"],
        }
    ).to_csv(compare_dir / "aligned_curves.csv", index=False)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare STARK vs PyChrono double-pendulum curves.")
    parser.add_argument("--stark-dir", type=Path, default=DEFAULT_STARK_DIR, help=f"STARK output dir (default: {DEFAULT_STARK_DIR})")
    parser.add_argument(
        "--chrono-dir", type=Path, default=DEFAULT_CHRONO_DIR, help=f"PyChrono output dir (default: {DEFAULT_CHRONO_DIR})"
    )
    parser.add_argument(
        "--compare-dir", type=Path, default=DEFAULT_COMPARE_DIR, help=f"Comparison output dir (default: {DEFAULT_COMPARE_DIR})"
    )
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR, help=f"Figure output dir (default: {DEFAULT_FIG_DIR})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = compare(
        stark_dir=args.stark_dir.resolve(),
        chrono_dir=args.chrono_dir.resolve(),
        compare_dir=args.compare_dir.resolve(),
        fig_dir=args.fig_dir.resolve(),
    )
    print(f"Wrote: {args.compare_dir.resolve() / 'summary.json'}")
    print(f"Wrote: {args.compare_dir.resolve() / 'aligned_curves.csv'}")
    print(f"consistency_pass: {metrics['consistency_pass']}")
    return 0 if metrics["consistency_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
