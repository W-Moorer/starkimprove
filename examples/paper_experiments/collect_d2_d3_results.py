#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from study_utils import FIGS_DIR, OUTPUT_BASE, latest_logger, parse_logger_metrics, save_fig, setup_axes

DISPLAY_NAME = {
    "full": "Contact-consistent IPC",
    "gap_only": "Gap-ratio-only schedule",
    "fixed_rate": "Fixed-rate hardening baseline",
    "fixed_soft": "Low fixed-kappa baseline",
}


def collect_case(case_dir: Path, label: str, extra: Dict[str, object] | None = None) -> Dict[str, object]:
    logger = latest_logger(case_dir)
    if logger is None:
        raise FileNotFoundError(f"Missing logger in {case_dir}")
    metrics = parse_logger_metrics(logger)
    row: Dict[str, object] = {
        "label": DISPLAY_NAME.get(label, label),
        "label_id": label,
        "case_dir": case_dir.name,
        "logger_file": logger.name,
        "total": metrics.get("total"),
        "failed_step_time": metrics.get("failed_step_time"),
        "failed_step_count": metrics.get("failed_step_count"),
        "hardening_count": metrics.get("hardening_count"),
        "newton_iterations": metrics.get("newton_iterations"),
        "linear_iterations": metrics.get("linear_iterations"),
    }
    if extra:
        row.update(extra)
    return row


def format_d3_label(variant: str, mass_ratio: int) -> str:
    return f"{DISPLAY_NAME[variant]} (MR={mass_ratio})"


def write_csv(rows: List[Dict[str, object]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_d2(rows: List[Dict[str, object]], fig_dir: Path):
    df = pd.DataFrame(rows)
    x = np.arange(len(df))
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax in axs:
        setup_axes(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], rotation=18, ha="right")

    axs[0].bar(x, pd.to_numeric(df["total"], errors="coerce"), color="#4c78a8")
    axs[0].set_ylabel("Runtime (s)")
    axs[0].set_title("D2: Total Runtime")

    axs[1].bar(x - 0.18, pd.to_numeric(df["failed_step_time"], errors="coerce"), width=0.36, color="#f58518", label="Failed-step time")
    axs[1].bar(x + 0.18, pd.to_numeric(df["hardening_count"], errors="coerce"), width=0.36, color="#54a24b", label="Hardening count")
    axs[1].set_ylabel("Count / Time")
    axs[1].set_title("D2: Failure and Hardening")
    axs[1].legend()
    save_fig(fig, fig_dir, "d2_complete_ablation")


def plot_d3(rows: List[Dict[str, object]], fig_dir: Path):
    df = pd.DataFrame(rows)
    fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.8))
    for ax in axs:
        setup_axes(ax)
        ax.set_xscale("log")
    for variant, color in [("full", "#1f77b4"), ("fixed_rate", "#d62728")]:
        sub = df[df["variant"] == variant].sort_values("mass_ratio")
        axs[0].plot(sub["mass_ratio"], sub["total"], marker="o", color=color, label=DISPLAY_NAME[variant])
        axs[1].plot(sub["mass_ratio"], sub["hardening_count"], marker="o", color=color, label=DISPLAY_NAME[variant])
    axs[0].set_xlabel("Mass ratio")
    axs[0].set_ylabel("Runtime (s)")
    axs[0].set_title("D3: Runtime vs Mass Ratio")
    axs[0].legend()
    axs[1].set_xlabel("Mass ratio")
    axs[1].set_ylabel("Hardening count")
    axs[1].set_title("D3: Hardening vs Mass Ratio")
    save_fig(fig, fig_dir, "d3_mass_ratio_sweep")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect D2/D3 study outputs from Exp1 directories.")
    parser.add_argument("--out-d2-csv", type=Path, default=OUTPUT_BASE / "d2_complete_ablation.csv")
    parser.add_argument("--out-d3-csv", type=Path, default=OUTPUT_BASE / "d3_mass_ratio_sweep.csv")
    parser.add_argument("--fig-dir", type=Path, default=FIGS_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    d2_rows = [
        collect_case(OUTPUT_BASE / "exp1_mass_adaptive", "full", {"variant": "full"}),
        collect_case(OUTPUT_BASE / "exp1_gap_adaptive", "gap_only", {"variant": "gap_only"}),
        collect_case(OUTPUT_BASE / "exp1_adaptive", "fixed_rate", {"variant": "fixed_rate"}),
        collect_case(OUTPUT_BASE / "exp1_fixed_soft", "fixed_soft", {"variant": "fixed_soft"}),
    ]
    write_csv(d2_rows, args.out_d2_csv.resolve())
    plot_d2(d2_rows, args.fig_dir.resolve())

    d3_rows: List[Dict[str, object]] = []
    for ratio in [1, 10, 100, 1000]:
        d3_rows.append(
            collect_case(
                OUTPUT_BASE / f"exp1_mr{ratio}_full",
                f"mr{ratio}_full",
                {"mass_ratio": ratio, "variant": "full", "method": DISPLAY_NAME["full"], "label": format_d3_label("full", ratio)},
            )
        )
        d3_rows.append(
            collect_case(
                OUTPUT_BASE / f"exp1_mr{ratio}_fixed_rate",
                f"mr{ratio}_fixed_rate",
                {
                    "mass_ratio": ratio,
                    "variant": "fixed_rate",
                    "method": DISPLAY_NAME["fixed_rate"],
                    "label": format_d3_label("fixed_rate", ratio),
                },
            )
        )
    write_csv(d3_rows, args.out_d3_csv.resolve())
    plot_d3(d3_rows, args.fig_dir.resolve())

    print(f"Wrote {args.out_d2_csv.resolve()}")
    print(f"Wrote {args.out_d3_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
