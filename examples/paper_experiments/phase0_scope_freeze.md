# Phase0 Scope Freeze

Baseline date: `2026-03-05`

This document records the Phase0 freeze decision and links to executable artifacts for baseline collection.

## Freeze Decision

- Freeze MVP targets: `A1`, `A2`, `D1` (defined in `phase0_experiments_config.json`).
- Freeze backup migration scope: `double_stiffness_rule + exp1 stack20 + exp2 impact + exp4 joints + exp5 threaded contact`.
- Freeze logging contract and metric vocabulary in `phase0_metrics_definition.md`.
- Exclude `exp1_fixed_stiff` from Phase0 baseline runset (cannot finish in acceptable time under current setup).

## Deliverables Implemented

- Experiment configuration list:
  - `examples/paper_experiments/phase0_experiments_config.json`
- Metric definition document:
  - `examples/paper_experiments/phase0_metrics_definition.md`
- One-click baseline runner and collector:
  - `examples/paper_experiments/run_phase0_baseline.ps1`
  - `examples/paper_experiments/phase0_baseline_runner.py`
- D1 parameter sensitivity runner (AL + solver sweep):
  - `examples/paper_experiments/run_d1_parameter_sensitivity.py`
- One-command MVP pipeline (A1/A2/D1 + D2):
  - `examples/paper_experiments/complete_mvp_pipeline.py`
- Figure generator:
  - `examples/paper_experiments/plot_results.py`

## Baseline Modes

- Collect existing outputs only:
  - `python examples/paper_experiments/phase0_baseline_runner.py`
- Build + run configured baseline set + collect:
  - `powershell -ExecutionPolicy Bypass -File examples/paper_experiments/run_phase0_baseline.ps1 -Run`
- End-to-end MVP (baseline + D1 + figures + summary):
  - `python examples/paper_experiments/complete_mvp_pipeline.py --run-d1`
  - `python examples/paper_experiments/complete_mvp_pipeline.py --run-baseline --run-d1`

## Phase0 Completion Gate

Phase0 is considered complete when:

1. The runner can execute configured STARK baseline cases in one command.
2. The collector exports `phase0_baseline_minlog.csv` and `phase0_baseline_summary.json`.
3. The exported fields satisfy the minimal log contract in `phase0_metrics_definition.md`.

## MVP Completion Snapshot (`2026-03-05`)

Required 8.1 scope (`A1/A2/D1 + D2`) is completed using retained seed scenes plus the refreshed four-bar A1 formal case:

- `A1` -> `exp4_fourbar_a1fix_soft` (+ `exp4_fourbar_a1fix_al` comparison)
- `A1` seed stress case -> `exp4_coupled_joints` / `exp4_coupled_joints_al`
- `A2` -> `exp2_v10/v100/v500`
- `D2` -> `exp1_adaptive/exp1_gap_adaptive/exp1_fixed_soft`
- `D1` -> `output/paper_experiments/d1_parameter_sensitivity.csv`

Completion artifacts:

- `output/paper_experiments/mvp_completion_summary.json`
- `output/paper_experiments/mvp_completion_summary.md`
- `documents/local/paper1/figs/a1_joint_drift_compare.pdf`
- `documents/local/paper1/figs/d2_runtime_breakdown.pdf`
- `documents/local/paper1/figs/d1_pareto_total_vs_error.pdf`
