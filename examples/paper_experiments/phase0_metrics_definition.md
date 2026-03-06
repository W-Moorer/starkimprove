# Phase0 Metrics Definition

This file freezes the metric vocabulary used by Phase0 so that later A1/A2/D1 runs use a consistent interpretation.

## Canonical Paper Metrics

| Metric ID | Symbol | Definition | Unit | Phase0 status |
| --- | --- | --- | --- | --- |
| `joint_error_max_l2` | `e_g` | `max_t ||g(x_t)||_2` over all accepted time steps. | m | Frozen (A1/A2 completed via seed scenes) |
| `loop_drift` | `e_drift` | Closed-loop geometric drift at end time or end cycle. | m | Frozen (A1 completed via seed scene) |
| `contact_gap_min` | `d_min` | Minimum contact gap over time (`>= 0` means no penetration under current convention). | m | Frozen |
| `hardening_count` | `N_hard` | Number of contact or joint stiffness hardening events. | count | Frozen |
| `failed_step_time` | `T_fail` | Wall-time spent in failed/retried time steps. | s | Frozen |
| `backtrack_count` | `N_backtrack` | Number of line-search backtracks (or equivalent rollback counter). | count | Frozen |
| `wall_time_total` | `T_total` | End-to-end runtime from logger field `total`. | s | Frozen |
| `newton_iterations` | `N_newton` | Total Newton iterations over all accepted steps. | count | Frozen |
| `linear_iterations` | `N_lin` | Total linear solver iterations (`linear_iterations`, aliased from `CG_iterations`). | count | Frozen |

## Minimal Baseline Log Contract (Phase0)

The Phase0 baseline collector exports these fields from `logger_*.txt`:

| Export key | Logger key | Meaning |
| --- | --- | --- |
| `total` | `total` | Total wall-time for one run. |
| `time_steps` | `time_steps` | Accepted time-step count. |
| `newton_iterations` | `newton_iterations` | Newton iteration count. |
| `linear_iterations` | `linear_iterations` | Canonical linear iteration count alias. |
| `cg_iterations` | `CG_iterations` | Backward-compatible linear iteration count. |
| `hardening_count` | `hardening_count` | Total hardening events (contact + joint paths). |
| `failed_step_count` | `failed_step_count` | Failed/retried step count. |
| `failed_step_time` | `failed_step_time` | Time spent in failed steps (seconds, canonical alias). |
| `line_search_iterations` | `line_search_iterations` | Line-search iteration count. |
| `failed_steps` | `failed_steps` | Time spent in failed steps (seconds). |
| `line_search` | `line_search` | Time spent in line-search (seconds). |
| `before_energy_evaluation` | `before_energy_evaluation` | Runtime section timing. |
| `evaluate_e_grad_hess` | `evaluate_E_grad_hess` | Runtime section timing. |
| `write_frame` | `write_frame` | Output time. |

## Current Phase0 Derived Snapshot Fields

The collector also exports last-row snapshot values from case CSVs when available:

- `final_time`
- `final_min_z` (from `min_z.csv`)
- `final_max_drift` (from the `joint_drift_<simulation>__<timestamp>.csv` matched to `logger_file`, falling back to `joint_drift.csv`)
- `final_v_x`, `final_v_y`, `final_v_z` (from `velocity.csv`, `impact_state.csv`, `state.csv`, or `screw_state.csv`)

## Notes

- Phase0 freezes names and formulas, not final performance numbers.
- As of `2026-03-05`, MVP-required `A1/A2/D1 + D2` are completed with retained seed scenes (`exp1/exp2/exp4`) and exported artifacts (`phase0_baseline_minlog.csv`, `d1_parameter_sensitivity.csv`, figure set in `documents/local/paper1/figs`).
