#!/usr/bin/env python3
"""
Contact-track paper asset summary for paper1.

This replaces the legacy A1/A2/D1 + joint-AL MVP pipeline. The current paper
line is contact-centric, so this script only summarizes the canonical
contact-dominant studies and the remaining verification artifacts.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
OUTPUT_BASE = REPO_ROOT / "output" / "paper_experiments"
FIG_OUT_DIR = REPO_ROOT / "documents" / "local" / "paper1" / "figs"

SUMMARY_JSON = OUTPUT_BASE / "contact_track_summary.json"
SUMMARY_MD = OUTPUT_BASE / "contact_track_summary.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def maybe(path: Path) -> str | None:
    return str(path) if path.exists() else None


def list_existing(paths: List[Path]) -> List[str]:
    return [str(p) for p in paths if p.exists()]


def build_payload() -> Dict:
    figures = list_existing(
        [
            FIG_OUT_DIR / "a1_fourbar_dt_sweep.pdf",
            FIG_OUT_DIR / "a1_fourbar_force_torque.pdf",
            FIG_OUT_DIR / "a1_chain10_joint_drift_compare.pdf",
            FIG_OUT_DIR / "a2_crank_slider_compare.pdf",
            FIG_OUT_DIR / "a3_limit_stop_compare.pdf",
            FIG_OUT_DIR / "exp5_bolt_vs_ref.pdf",
            FIG_OUT_DIR / "exp7_forklift_compare.pdf",
            FIG_OUT_DIR / "d1_pareto_total_vs_error.pdf",
            FIG_OUT_DIR / "d2_complete_ablation.pdf",
            FIG_OUT_DIR / "d3_mass_ratio_sweep.pdf",
        ]
    )
    artifacts = {
        "a1_fourbar_dt_sweep_csv": maybe(OUTPUT_BASE / "a1_fourbar_dt_sweep.csv"),
        "a2_crank_slider_summary_csv": maybe(OUTPUT_BASE / "a2_crank_slider_summary.csv"),
        "a3_limit_stop_summary_csv": maybe(OUTPUT_BASE / "a3_limit_stop_summary.csv"),
        "d1_parameter_sensitivity_csv": maybe(OUTPUT_BASE / "d1_parameter_sensitivity.csv"),
        "d2_complete_ablation_csv": maybe(OUTPUT_BASE / "d2_complete_ablation.csv"),
        "d3_mass_ratio_sweep_csv": maybe(OUTPUT_BASE / "d3_mass_ratio_sweep.csv"),
        "exp5_complex_contact_summary_csv": maybe(OUTPUT_BASE / "exp5_complex_contact_summary.csv"),
        "exp7_forklift_summary_csv": maybe(OUTPUT_BASE / "exp7_forklift_summary.csv"),
        "figures": figures,
    }
    present = [k for k, v in artifacts.items() if v]
    required = [
        "a2_crank_slider_summary_csv",
        "a3_limit_stop_summary_csv",
        "d2_complete_ablation_csv",
        "d3_mass_ratio_sweep_csv",
        "exp5_complex_contact_summary_csv",
        "exp7_forklift_summary_csv",
    ]
    status = "completed" if all(artifacts.get(k) for k in required) else "partial"
    return {
        "generated_utc": utc_now(),
        "status": status,
        "paper_line": "contact_consistent_ipc",
        "primary_claims": [
            "inertia-aware contact stiffness initialization",
            "active-set-stable contact scheduling",
            "contact-driven time-step acceptance",
        ],
        "supporting_experiments": {
            "verification_only": ["A1_four_bar", "A1_chain10"],
            "main_evidence": ["A2_crank_slider", "A3_limit_stop", "S1_exp5", "S2_exp7", "D1", "D2", "D3"],
            "negative_or_boundary": ["B1_preconditioner_ablation"],
        },
        "artifacts": artifacts,
        "available_artifact_count": len(present),
    }


def write_markdown(path: Path, payload: Dict):
    lines = [
        "# Contact-Track Summary",
        "",
        f"- Generated UTC: `{payload['generated_utc']}`",
        f"- Status: `{payload['status']}`",
        f"- Paper line: `{payload['paper_line']}`",
        "",
        "## Primary Claims",
        "",
    ]
    for item in payload["primary_claims"]:
        lines.append(f"- `{item}`")
    lines += ["", "## Experiment Roles", ""]
    for bucket, entries in payload["supporting_experiments"].items():
        lines.append(f"- `{bucket}`: {', '.join(entries)}")
    lines += ["", "## Artifacts", ""]
    for key, value in payload["artifacts"].items():
        if isinstance(value, list):
            lines.append(f"- `{key}`:")
            for item in value:
                lines.append(f"  - `{item}`")
        elif value:
            lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    payload = build_payload()
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(SUMMARY_MD, payload)
    print(f"Wrote: {SUMMARY_JSON}")
    print(f"Wrote: {SUMMARY_MD}")
    print(f"Status: {payload['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
