# Contact-Track Scope Freeze

Baseline date: `2026-03-07`

This note replaces the legacy joint-AL MVP freeze. The current paper line is
contact-centric and keeps soft joints as the structural baseline.

## Freeze Decision

- Freeze the paper line as `contact_consistent_ipc`.
- Keep `A1` only for structural verification and chain-stress illustration.
- Move main evidence to `A2`, `A3`, `S1`, `S2`, `D1`, `D2`, and `D3`.
- Treat `B1` as a negative or boundary result, not as a core innovation point.

## Canonical Entry Points

- `examples/paper_experiments/run_a1_fourbar_study.py`
- `examples/paper_experiments/run_a2_crank_slider_benchmark.py`
- `examples/paper_experiments/run_a3_limit_stop_study.py`
- `examples/paper_experiments/run_d1_parameter_sensitivity.py`
- `examples/paper_experiments/run_exp5_complex_contact_study.py`
- `examples/paper_experiments/run_exp7_forklift_benchmark.py`
- `examples/paper_experiments/complete_mvp_pipeline.py`

## Current Figure Set

- `documents/local/paper1/figs/a1_fourbar_dt_sweep.pdf`
- `documents/local/paper1/figs/a1_chain10_joint_drift_compare.pdf`
- `documents/local/paper1/figs/a2_crank_slider_compare.pdf`
- `documents/local/paper1/figs/a3_limit_stop_compare.pdf`
- `documents/local/paper1/figs/exp5_bolt_vs_ref.pdf`
- `documents/local/paper1/figs/exp7_forklift_compare.pdf`
- `documents/local/paper1/figs/d1_pareto_total_vs_error.pdf`
- `documents/local/paper1/figs/d2_complete_ablation.pdf`
- `documents/local/paper1/figs/d3_mass_ratio_sweep.pdf`

## Interpretation Guardrails

1. Do not reintroduce joint-specific method lines as the main contribution.
2. Do not use A1 as the primary innovation proof.
3. Do not claim a solver gain from `B1`.
4. Frame all primary claims around contact-stiffness organization, active-set
   stability, and contact-driven step control.
