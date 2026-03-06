#!/usr/bin/env python3
"""
Generate paper figures for retained paper1 experiments (exp1/exp2/exp4/exp5) and D1/D2 MVP artifacts.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "output" / "paper_experiments"
DEFAULT_OUT_DIR = REPO_ROOT / "documents" / "local" / "paper1" / "figs"
DEFAULT_BASELINE_CSV = DEFAULT_DATA_DIR / "phase0_baseline_minlog.csv"
DEFAULT_D1_CSV = DEFAULT_DATA_DIR / "d1_parameter_sensitivity.csv"


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 13
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["legend.fontsize"] = 11
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["axes.linewidth"] = 1.1
plt.rcParams["lines.linewidth"] = 1.8


def setup_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.45)


def load_ref_curve(csv_path: Path, scale: float = 1.0) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    x = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    y = pd.to_numeric(df.iloc[:, 1], errors="coerce") * scale
    mask = x.notna() & y.notna()
    return pd.DataFrame({"t": x[mask], "v": y[mask]}).sort_values("t")


def sanitize_curve(df: pd.DataFrame, x_col: str, y_col: str, max_points: int = 2000) -> pd.DataFrame:
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
        logger_file = Path(logger_name).name
        if not logger_file.startswith("logger_") or not logger_file.endswith(".txt"):
            raise ValueError(f"Invalid logger name for {case_dir.name}: {logger_name}")
        run_file = curve_stem + "_" + logger_file[len("logger_") : -4] + ".csv"
        run_path = case_dir / run_file
        if run_path.exists():
            return run_path
        raise FileNotFoundError(f"Missing run-specific curve file: {run_path}")

    fallback = case_dir / f"{curve_stem}.csv"
    if not fallback.exists():
        raise FileNotFoundError(f"Missing fallback curve file: {fallback}")
    return fallback


def save_fig(fig: plt.Figure, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.svg", format="svg")
    fig.savefig(out_dir / f"{stem}.pdf", format="pdf")
    plt.close(fig)
    print(f"Saved {stem}.svg/.pdf")


def plot_exp1_settling(data_dir: Path, out_dir: Path):
    centers_path = data_dir / "exp1_adaptive" / "centers_z.csv"
    minz_path = data_dir / "exp1_adaptive" / "min_z.csv"
    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)

    if centers_path.exists():
        df = pd.read_csv(centers_path)
        if "t" in df.columns:
            t = pd.to_numeric(df["t"], errors="coerce")
            z_cols = [c for c in df.columns if c.startswith("z_")]
            for c in z_cols:
                z = pd.to_numeric(df[c], errors="coerce")
                mask = t.notna() & z.notna()
                ax.plot(t[mask], z[mask], alpha=0.6, linewidth=1.0)
            ax.set_ylabel("Center Height z (m)")
            ax.set_title("Exp1: Box Center-z Trajectories")
    elif minz_path.exists():
        df = pd.read_csv(minz_path)
        curve = sanitize_curve(df, "t", "min_z")
        ax.plot(curve["t"], curve["min_z"], color="#1f77b4", label="Minimum Object Z")
        ax.legend()
        ax.set_ylabel("Vertical Position (m)")
        ax.set_title("Exp1: Settling Dynamics")
    else:
        raise FileNotFoundError("Missing exp1 input csv.")

    ax.set_xlabel("Time (s)")
    save_fig(fig, out_dir, "exp1_settling")


def plot_exp2_impact(data_dir: Path, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    speeds = [10, 100, 500]
    plotted = False
    for color, v0 in zip(colors, speeds):
        path = data_dir / f"exp2_v{v0}" / "impact_state.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        curve = sanitize_curve(df, "t", "x")
        ax.plot(curve["t"], curve["x"], color=color, label=f"v0 = {v0} m/s")
        plotted = True

    if not plotted:
        raise FileNotFoundError("Missing exp2 impact csv.")

    ax.axhspan(1.5, 2.5, color="gray", alpha=0.25, label="Wall extent")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Horizontal Position x (m)")
    ax.set_title("A2 Seed: High-Speed Impact")
    ax.legend(loc="lower right")
    save_fig(fig, out_dir, "exp2_impact")


def plot_joint_drift_compare(
    data_dir: Path,
    out_dir: Path,
    base_logger: str | None = None,
    al_logger: str | None = None,
    base_dir_name: str = "exp4_fourbar_a1fix_soft",
    al_dir_name: str = "exp4_fourbar_a1fix_al",
    title: str = "A1 Four-Bar: Joint Drift Baseline vs AL",
    secondary_title: str = "Four-Bar Joint Constraint Accuracy",
    primary_stem: str = "a1_joint_drift_compare",
    secondary_stem: str = "exp4_drift",
):
    base_dir = data_dir / base_dir_name
    al_dir = data_dir / al_dir_name
    base_path = resolve_run_curve(base_dir, "joint_drift", base_logger)
    al_path = resolve_run_curve(al_dir, "joint_drift", al_logger) if al_dir.exists() else None

    base_df = sanitize_curve(pd.read_csv(base_path), "t", "max_drift")
    al_df = sanitize_curve(pd.read_csv(al_path), "t", "max_drift") if al_path.exists() else None

    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)
    ax.plot(base_df["t"], 1e3 * base_df["max_drift"], color="#e377c2", label="Soft-constraint baseline")
    if al_df is not None and not al_df.empty:
        ax.plot(al_df["t"], 1e3 * al_df["max_drift"], color="#1f77b4", label="AL-IPC")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Drift (mm)")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, out_dir, primary_stem)

    # Backward-compatible file names
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    setup_axes(ax2)
    ax2.plot(base_df["t"], 1e3 * base_df["max_drift"], color="#e377c2", label="Max Constraint Violation")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Joint Drift (mm)")
    ax2.set_title(secondary_title)
    ax2.legend()
    save_fig(fig2, out_dir, secondary_stem)


def plot_exp4_drift_compare(
    data_dir: Path,
    out_dir: Path,
    base_logger: str | None = None,
    al_logger: str | None = None,
    base_dir_name: str = "exp4_fourbar_a1fix_soft",
    al_dir_name: str = "exp4_fourbar_a1fix_al",
):
    plot_joint_drift_compare(
        data_dir,
        out_dir,
        base_logger,
        al_logger,
        base_dir_name,
        al_dir_name,
        title="A1 Four-Bar: Joint Drift Baseline vs AL",
        secondary_title="Four-Bar Joint Constraint Accuracy",
        primary_stem="a1_joint_drift_compare",
        secondary_stem="exp4_drift",
    )


def plot_exp4_chain10_supplement(
    data_dir: Path,
    out_dir: Path,
    base_logger: str | None = None,
    al_logger: str | None = None,
    base_dir_name: str = "exp4_chain10_supp_soft",
    al_dir_name: str = "exp4_chain10_supp_al",
):
    plot_joint_drift_compare(
        data_dir,
        out_dir,
        base_logger,
        al_logger,
        base_dir_name,
        al_dir_name,
        title="A1 Supplement: 10-Link Chain Drift",
        secondary_title="10-Link Chain Constraint Accuracy",
        primary_stem="a1_chain10_joint_drift_compare",
        secondary_stem="exp4_chain10_drift",
    )


def plot_exp5_bolt_vs_ref(data_dir: Path, out_dir: Path):
    summary_csv = data_dir / "exp5_complex_contact_summary.csv"
    sim_path = data_dir / "exp5_bolt" / "screw_state.csv"
    ref_pos_path = data_dir / "exp5_ref" / "Pos_Ty_mu0.csv"
    ref_vel_path = data_dir / "exp5_ref" / "Vel_Ty_mu0.csv"
    if not sim_path.exists() or not ref_pos_path.exists() or not ref_vel_path.exists():
        raise FileNotFoundError("Missing exp5 simulation/reference csv.")

    ref_pos = load_ref_curve(ref_pos_path, scale=0.01)
    ref_vel = load_ref_curve(ref_vel_path, scale=0.01)
    y0_ref = float(ref_pos["v"].iloc[0])
    ref_y_aligned = ref_pos["v"] - y0_ref

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    setup_axes(axs[0])
    setup_axes(axs[1])

    axs[0].plot(ref_pos["t"], ref_y_aligned, color="black", linestyle="--", linewidth=1.5, label="RecurDyn (aligned)")
    axs[1].plot(ref_vel["t"], ref_vel["v"], color="black", linestyle="--", linewidth=1.5, label="RecurDyn")

    palette = {
        "STARK": "#1f77b4",
        "nsc_lcp": "#2ca02c",
        "nsc_ncp": "#ff7f0e",
        "smc_penalty": "#d62728",
    }
    labels = {
        "STARK": "STARK IPC",
        "nsc_lcp": "PyChrono NSC-LCP",
        "nsc_ncp": "PyChrono NSC-APGD",
        "smc_penalty": "PyChrono SMC",
    }

    plotted = False
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        for _, row in summary.iterrows():
            framework = str(row.get("framework", ""))
            if framework == "PyChrono":
                stable = pd.to_numeric(pd.Series([row.get("stable", np.nan)]), errors="coerce").iloc[0]
                comp = pd.to_numeric(pd.Series([row.get("composite_error", np.nan)]), errors="coerce").iloc[0]
                if not np.isfinite(stable) or stable < 0.5 or not np.isfinite(comp) or comp >= 5.0:
                    continue
            state_path = Path(str(row.get("state_csv", "")))
            if not state_path.exists():
                continue
            sim_df = pd.read_csv(state_path)
            sim_df = sim_df.drop_duplicates(subset=["t"], keep="last").sort_values("t")
            y0_sim = float(pd.to_numeric(sim_df["y"], errors="coerce").dropna().iloc[0])
            y_aligned = pd.to_numeric(sim_df["y"], errors="coerce") - y0_sim
            mode = "STARK" if framework == "STARK" else str(row.get("mode", ""))
            axs[0].plot(sim_df["t"], y_aligned, color=palette.get(mode, "#7f7f7f"), label=labels.get(mode, mode))
            axs[1].plot(sim_df["t"], sim_df["vy"], color=palette.get(mode, "#7f7f7f"), label=labels.get(mode, mode))
            plotted = True

    if not plotted:
        sim_df = pd.read_csv(sim_path)
        sim_df = sim_df.drop_duplicates(subset=["t"], keep="last").sort_values("t")
        y0_sim = float(sim_df["y"].iloc[0])
        y_aligned = sim_df["y"] - y0_sim
        axs[0].plot(sim_df["t"], y_aligned, color="#1f77b4", label="STARK IPC (aligned)")
        axs[1].plot(sim_df["t"], sim_df["vy"], color="#1f77b4", label="STARK IPC")

    axs[0].set_ylabel("Position y (m)")
    axs[0].set_title("Exp5: Screw y / vy vs RecurDyn")
    axs[0].legend(loc="upper right")

    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel(r"Velocity $v_y$ (m/s)")
    axs[1].legend(loc="lower right")
    save_fig(fig, out_dir, "exp5_bolt_vs_ref")


def plot_d2_runtime(baseline_csv: Path, out_dir: Path):
    if not baseline_csv.exists():
        raise FileNotFoundError("Missing phase0_baseline_minlog.csv.")
    df = pd.read_csv(baseline_csv)
    d2 = df[df["case_id"] == "D2_seed_stack20"].copy()
    if d2.empty:
        raise ValueError("No D2_seed_stack20 rows in baseline csv.")

    d2 = d2.sort_values("output_dir")
    labels = d2["output_dir"].astype(str).tolist()
    total = pd.to_numeric(d2["total"], errors="coerce").fillna(0.0).to_numpy()
    failed = pd.to_numeric(d2["failed_step_time"], errors="coerce").fillna(0.0).to_numpy()

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    setup_axes(ax)
    ax.bar(x - w / 2, total, width=w, color="#4c78a8", label="Total Runtime (s)")
    ax.bar(x + w / 2, failed, width=w, color="#f58518", label="Failed-Step Time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("Time (s)")
    ax.set_title("D2: Runtime Breakdown by Stiffness Strategy")
    ax.legend()
    save_fig(fig, out_dir, "d2_runtime_breakdown")


def plot_d1_pareto(d1_csv: Path, out_dir: Path):
    if not d1_csv.exists():
        raise FileNotFoundError("Missing d1_parameter_sensitivity.csv.")
    df = pd.read_csv(d1_csv)
    if df.empty:
        raise ValueError("D1 csv is empty.")
    for col in ("total", "joint_error_max_l2", "rho0"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["total", "joint_error_max_l2", "rho0"])
    if df.empty:
        raise ValueError("D1 csv has no valid rows for Pareto plotting.")

    fig, ax = plt.subplots(figsize=(6, 4))
    setup_axes(ax)
    sc = ax.scatter(df["joint_error_max_l2"], df["total"], c=df["rho0"], cmap="viridis", s=36, alpha=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Joint Error Max L2 (m)")
    ax.set_ylabel("Total Runtime (s)")
    ax.set_title("D1: Error-Cost Pareto Seeds")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("rho0")
    save_fig(fig, out_dir, "d1_pareto_total_vs_error")


def run_all(
    data_dir: Path,
    out_dir: Path,
    baseline_csv: Path,
    d1_csv: Path,
    exp4_base_logger: str | None = None,
    exp4_al_logger: str | None = None,
    exp4_base_dir: str = "exp4_coupled_joints",
    exp4_al_dir: str = "exp4_coupled_joints_al",
    exp4_chain_base_logger: str | None = None,
    exp4_chain_al_logger: str | None = None,
    exp4_chain_base_dir: str = "exp4_chain10_supp_soft",
    exp4_chain_al_dir: str = "exp4_chain10_supp_al",
):
    tasks: List[Tuple[str, Callable[[], None]]] = [
        ("exp1_settling", lambda: plot_exp1_settling(data_dir, out_dir)),
        ("exp2_impact", lambda: plot_exp2_impact(data_dir, out_dir)),
        (
            "exp4_drift_compare",
            lambda: plot_exp4_drift_compare(
                data_dir,
                out_dir,
                exp4_base_logger,
                exp4_al_logger,
                exp4_base_dir,
                exp4_al_dir,
            ),
        ),
        (
            "exp4_chain10_supplement",
            lambda: plot_exp4_chain10_supplement(
                data_dir,
                out_dir,
                exp4_chain_base_logger,
                exp4_chain_al_logger,
                exp4_chain_base_dir,
                exp4_chain_al_dir,
            ),
        ),
        ("exp5_bolt_vs_ref", lambda: plot_exp5_bolt_vs_ref(data_dir, out_dir)),
        ("d2_runtime_breakdown", lambda: plot_d2_runtime(baseline_csv, out_dir)),
        ("d1_pareto_total_vs_error", lambda: plot_d1_pareto(d1_csv, out_dir)),
    ]
    errors = []
    for name, fn in tasks:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - best effort for partial outputs
            errors.append((name, str(exc)))
            print(f"[plot_results] {name} failed: {exc}")
    if errors:
        print("[plot_results] completed with warnings.")
        for name, msg in errors:
            print(f"  - {name}: {msg}")
    else:
        print("[plot_results] all figures generated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper1 experiment figures.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help=f"Data directory (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"Figure output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=DEFAULT_BASELINE_CSV,
        help=f"Phase0 baseline CSV (default: {DEFAULT_BASELINE_CSV})",
    )
    parser.add_argument(
        "--d1-csv",
        type=Path,
        default=DEFAULT_D1_CSV,
        help=f"D1 sweep CSV (default: {DEFAULT_D1_CSV})",
    )
    parser.add_argument("--exp4-base-logger", type=str, default=None, help="Use a specific exp4 baseline logger file name.")
    parser.add_argument("--exp4-al-logger", type=str, default=None, help="Use a specific exp4 AL logger file name.")
    parser.add_argument("--exp4-base-dir", type=str, default="exp4_fourbar_a1fix_soft", help="Use a specific exp4 baseline output directory name.")
    parser.add_argument("--exp4-al-dir", type=str, default="exp4_fourbar_a1fix_al", help="Use a specific exp4 AL output directory name.")
    parser.add_argument("--exp4-chain-base-logger", type=str, default=None, help="Use a specific exp4 10-link baseline logger file name.")
    parser.add_argument("--exp4-chain-al-logger", type=str, default=None, help="Use a specific exp4 10-link AL logger file name.")
    parser.add_argument("--exp4-chain-base-dir", type=str, default="exp4_chain10_supp_soft", help="Use a specific exp4 10-link baseline output directory name.")
    parser.add_argument("--exp4-chain-al-dir", type=str, default="exp4_chain10_supp_al", help="Use a specific exp4 10-link AL output directory name.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_all(
        data_dir=args.data_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        baseline_csv=args.baseline_csv.resolve(),
        d1_csv=args.d1_csv.resolve(),
        exp4_base_logger=args.exp4_base_logger,
        exp4_al_logger=args.exp4_al_logger,
        exp4_base_dir=args.exp4_base_dir,
        exp4_al_dir=args.exp4_al_dir,
        exp4_chain_base_logger=args.exp4_chain_base_logger,
        exp4_chain_al_logger=args.exp4_chain_al_logger,
        exp4_chain_base_dir=args.exp4_chain_base_dir,
        exp4_chain_al_dir=args.exp4_chain_al_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
